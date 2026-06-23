"""Script to play a checkpoint of an RL agent from RSL-RL MoE+CTS."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL MoE+CTS.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import os
import time
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.rl.on_policy_runner_cts import OnPolicyRunnerCTS


class MoECTSInferenceWrapper(nn.Module):
    """Wrapper for exporting MoE+CTS inference policy (student encoder + actor).

    The standard isaaclab_rl exporter only copies the actor MLP, which expects
    (latent + obs) as input. This wrapper includes the full inference pipeline:
    obs → history buffer → student_moe_encoder → latent → concat(latent, obs) → actor → actions.
    """

    def __init__(self, model, num_obs, history_length, device="cpu"):
        super().__init__()
        self.num_obs = num_obs
        self.history_length = history_length
        # Deep copy the student encoder and actor for export
        self.student_moe_encoder = copy.deepcopy(model.student_moe_encoder)
        self.actor = copy.deepcopy(model.actor)
        # Register history buffer (non-persistent, initialized at inference time)
        self.register_buffer("history", torch.zeros(1, history_length, num_obs), persistent=False)

    def forward(self, obs):
        # obs shape: [batch, num_obs] or [num_obs]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # Expand history buffer to match batch size if needed
        if self.history.shape[0] != obs.shape[0]:
            self.history = self.history.expand(obs.shape[0], -1, -1).clone()
        # Roll history buffer
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        # Student encoder: history → latent
        latent, _ = self.student_moe_encoder(self.history.flatten(1))
        # Concat latent + obs → actor
        x = torch.cat([latent, obs], dim=1)
        return self.actor(x)

    def reset(self):
        """Reset history buffer to zeros."""
        self.history.zero_()


class MoECTSOnnxWrapper(nn.Module):
    """Stateless wrapper for ONNX export: history is an explicit input/output.

    Unlike MoECTSInferenceWrapper which maintains internal state, this wrapper
    takes history as an explicit input and returns the updated history as an output.
    This is necessary because ONNX is stateless and cannot preserve internal
    buffers between inference calls.
    """

    def __init__(self, model, num_obs, history_length):
        super().__init__()
        self.num_obs = num_obs
        self.history_length = history_length
        self.student_moe_encoder = copy.deepcopy(model.student_moe_encoder)
        self.actor = copy.deepcopy(model.actor)

    def forward(self, obs, history):
        # obs: [batch, num_obs], history: [batch, history_length, num_obs]
        new_history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent, _ = self.student_moe_encoder(new_history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        actions = self.actor(x)
        return actions, new_history


def export_moe_cts_policy_as_jit(model, num_obs, history_length, path, filename="policy.pt"):
    """Export MoE+CTS inference policy as Torch JIT.

    Args:
        model: The full ActorCriticMoECTS model.
        num_obs: Number of policy observation dimensions.
        history_length: History buffer length.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    os.makedirs(path, exist_ok=True)
    wrapper = MoECTSInferenceWrapper(model, num_obs, history_length, device="cpu")
    wrapper.to("cpu")
    wrapper.eval()
    traced = torch.jit.script(wrapper)
    traced.save(os.path.join(path, filename))
    print(f"[INFO] Exported JIT policy to {os.path.join(path, filename)}")


def export_moe_cts_policy_as_onnx(model, num_obs, history_length, path, filename="policy.onnx", verbose=False):
    """Export MoE+CTS inference policy as ONNX with explicit history input/output.

    The ONNX model has two inputs (obs, history) and two outputs (actions, new_history).
    The caller must maintain the history buffer externally and pass it as input on each call.

    Args:
        model: The full ActorCriticMoECTS model.
        num_obs: Number of policy observation dimensions.
        history_length: History buffer length.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    wrapper = MoECTSOnnxWrapper(model, num_obs, history_length)
    wrapper.to("cpu")
    wrapper.eval()
    dummy_obs = torch.zeros(1, num_obs)
    dummy_history = torch.zeros(1, history_length, num_obs)
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_history),
        os.path.join(path, filename),
        export_params=True,
        opset_version=11,
        verbose=verbose,
        input_names=["obs", "history"],
        output_names=["actions", "new_history"],
        dynamic_axes={},
    )
    print(f"[INFO] Exported ONNX policy to {os.path.join(path, filename)}")
    print(f"[INFO]   Input:  obs [1, {num_obs}], history [1, {history_length}, {num_obs}]")
    print(f"[INFO]   Output: actions [1, 12], new_history [1, {history_length}, {num_obs}]")


def main():
    """Play with RSL-RL MoE+CTS agent."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    cfg_dict = agent_cfg.to_dict()
    runner = OnPolicyRunnerCTS(env, cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.model

    # Get observation dimensions for export
    obs_dict = env.unwrapped.observation_manager.compute()
    num_obs = obs_dict["policy"].shape[-1]

    # Extract history_length from config
    cfg_dict = agent_cfg.to_dict()
    if "runner" in cfg_dict:
        history_length = cfg_dict["runner"].get("history_length", 5)
    else:
        history_length = cfg_dict.get("history_length", 5)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_moe_cts_policy_as_jit(policy_nn, num_obs, history_length, path=export_model_dir, filename="policy.pt")
    except Exception as e:
        print(f"[WARNING] JIT export failed: {e}. Skipping JIT export.")
    try:
        export_moe_cts_policy_as_onnx(policy_nn, num_obs, history_length, path=export_model_dir, filename="policy.onnx")
    except Exception as e:
        print(f"[WARNING] ONNX export failed: {e}. Skipping ONNX export.")

    dt = env.unwrapped.step_dt

    obs, _ = env.unwrapped.reset()
    obs = obs["policy"]
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs_dict, _, _, _, _ = env.unwrapped.step(actions)
            obs = obs_dict["policy"]
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
