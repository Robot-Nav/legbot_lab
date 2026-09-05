"""Run a 16-DOF Legbot CTS policy in MuJoCo.

The first 12 policy actions are leg position offsets.  The final four actions
are wheel velocity commands, matching ``robot_lab.tasks.legbot.env_cfg``.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from utils import (
    MujocoRenderUtils,
    build_delay_buffers,
    display_current_command,
    gravity_from_quat,
    init_joystick,
    load_config,
    open_video_writer,
    pd_control,
    read_joystick_command,
    sample_delayed_targets,
    set_initial_state,
    setup_tracking_camera,
)


CONFIG_NAME = "legbot.yaml"
VIDEO_DIR = Path(__file__).with_name("videos")
ROOT_DIR = Path(__file__).resolve().parents[2]
TERRAIN_XMLS = {
    "flat": ROOT_DIR / "resources" / "legbot_wf" / "legbot.xml",
    "stairs": ROOT_DIR / "resources" / "legbot_wf" / "legbot_stairs.xml",
    "rough": ROOT_DIR / "resources" / "legbot_wf" / "legbot_rough.xml",
    "mixed": ROOT_DIR / "resources" / "legbot_wf" / "legbot_mixed.xml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_NAME, help="Config filename in deploy_mujoco/configs.")
    parser.add_argument("--policy", type=Path, help="Override the policy_path from the config.")
    parser.add_argument(
        "--terrain",
        choices=tuple(TERRAIN_XMLS),
        help="Override the MuJoCo scene: flat, stairs, rough, or mixed.",
    )
    parser.add_argument("--validate", action="store_true", help="Validate model/config without opening a viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer or real-time sleeping.")
    parser.add_argument("--duration", type=float, help="Override simulation_duration in seconds.")
    return parser.parse_args()


def validate_config(cfg, model: mujoco.MjModel) -> None:
    expected = cfg.num_actions
    arrays = {
        "kps": cfg.kps,
        "kds": cfg.kds,
        "torque_limit": cfg.torque_limit,
        "default_angles": cfg.default_angles,
    }
    for name, value in arrays.items():
        if value.shape != (expected,):
            raise ValueError(f"{name} must have shape ({expected},), got {value.shape}.")
    if model.nq != 7 + expected or model.nv != 6 + expected or model.nu != expected:
        raise ValueError(
            f"Unexpected MuJoCo dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}; "
            f"expected {7 + expected}, {6 + expected}, {expected}."
        )
    expected_obs = 9 + cfg.num_leg_joints + 2 * cfg.num_actions
    if cfg.num_obs != expected_obs:
        raise ValueError(f"num_obs={cfg.num_obs}, but Legbot observation layout requires {expected_obs}.")
    if cfg.history_len != 10:
        raise ValueError(f"history_len must match training history length 10, got {cfg.history_len}.")


def build_single_obs(data, action_model: np.ndarray, cmd: np.ndarray, cfg) -> np.ndarray:
    joint_pos_mj = (data.qpos[7:] - cfg.default_angles) * cfg.dof_pos_scale
    joint_vel_mj = data.qvel[6:] * cfg.dof_vel_scale
    joint_pos_model = joint_pos_mj[cfg.idx_mj2model]
    joint_vel_model = joint_vel_mj[cfg.idx_mj2model]
    obs = np.concatenate(
        (
            data.qvel[3:6] * cfg.ang_vel_scale,
            gravity_from_quat(data.qpos[3:7]),
            cmd * cfg.cmd_scale,
            joint_pos_model[: cfg.num_leg_joints],
            joint_vel_model,
            action_model,
        )
    ).astype(np.float32, copy=False)
    if obs.shape != (cfg.num_obs,):
        raise RuntimeError(f"Built observation has shape {obs.shape}; expected ({cfg.num_obs},).")
    return obs


def action_to_targets(action_model: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Convert model-order actions to MuJoCo-order position/velocity targets."""
    default_model = cfg.default_angles[cfg.idx_mj2model]
    target_pos_model = default_model.copy()
    target_vel_model = np.zeros(cfg.num_actions, dtype=np.float32)
    target_pos_model[: cfg.num_leg_joints] += (
        action_model[: cfg.num_leg_joints] * cfg.action_pos_scale
    )
    target_vel_model[cfg.num_leg_joints :] = (
        action_model[cfg.num_leg_joints :] * cfg.action_vel_scale
    )
    return target_pos_model[cfg.idx_model2mj], target_vel_model[cfg.idx_model2mj]


def load_policy(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Policy not found: {path}. Run play.py once to export policy.pt, then pass it with --policy."
        )
    policy = torch.jit.load(str(path), map_location="cpu")
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.policy is not None:
        cfg.policy_path = args.policy.expanduser().resolve()
    if args.terrain is not None:
        cfg.xml_path = TERRAIN_XMLS[args.terrain]
    if args.duration is not None:
        if args.duration <= 0:
            raise ValueError("--duration must be positive.")
        cfg.duration = args.duration
    if args.headless and cfg.save_video:
        raise ValueError("Headless video recording is not supported; set save_video: false.")

    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    model.opt.timestep = cfg.dt
    validate_config(cfg, model)
    if args.validate:
        print(
            f"Legbot Sim2Sim configuration OK: scene={cfg.xml_path.name}, nq={model.nq}, nv={model.nv}, "
            f"nu={model.nu}, obs={cfg.num_obs}, actions={cfg.num_actions}."
        )
        return

    data = mujoco.MjData(model)
    set_initial_state(data, cfg.base_init_pos, cfg.base_init_quat, cfg.default_angles)
    mujoco.mj_forward(model, data)
    policy = load_policy(cfg.policy_path)
    joystick = None if args.headless else init_joystick()
    cmd = cfg.cmd_init.copy()
    display_current_command(cmd)

    action_model = np.zeros(cfg.num_actions, dtype=np.float32)
    target_pos = cfg.default_angles.copy()
    target_vel = np.zeros(cfg.num_actions, dtype=np.float32)
    pos_history, vel_history = build_delay_buffers(target_pos, target_vel, delay_max=cfg.delay_max)
    delay_rng = np.random.default_rng(cfg.delay_seed)
    # One independently delayed group per physical leg, including its wheel.
    actuator_groups = tuple(np.arange(i, i + 4, dtype=np.int64) for i in range(0, cfg.num_actions, 4))

    writer, frame_skip, video_path = open_video_writer(
        cfg.save_video,
        policy_path=cfg.policy_path,
        cmd=cmd,
        dt=cfg.dt,
        video_dir=VIDEO_DIR,
        video_fps=cfg.video_fps,
    )
    renderer = mujoco.Renderer(model, height=360, width=640) if writer else None
    render_substeps = max(1, int((1.0 / cfg.render_fps) / cfg.dt))
    render_utils = MujocoRenderUtils()

    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    with viewer_context as viewer:
        if viewer is not None:
            setup_tracking_camera(viewer)
        counter = 0
        while data.time < cfg.duration and (viewer is None or viewer.is_running()):
            step_start = time.time()
            if joystick and counter % cfg.decimation == 0:
                cmd = read_joystick_command(joystick, cfg.max_cmd)

            torque = pd_control(target_pos, data.qpos[7:], cfg.kps, target_vel, data.qvel[6:], cfg.kds)
            data.ctrl[:] = np.clip(torque, -cfg.torque_limit, cfg.torque_limit)
            mujoco.mj_step(model, data)
            render_utils.update(cmd, data)

            if writer and renderer is not None and counter % frame_skip == 0:
                renderer.update_scene(data, camera=viewer.cam)
                render_utils.update_external_rendering(renderer, ctype="renderer")
                writer.append_data(renderer.render())

            counter += 1
            if counter % cfg.decimation == 0:
                single_obs = build_single_obs(data, action_model, cmd, cfg)
                with torch.inference_mode():
                    action_tensor = policy(torch.from_numpy(single_obs).unsqueeze(0))
                action_model = action_tensor.cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                if action_model.shape != (cfg.num_actions,):
                    raise RuntimeError(f"Policy returned shape {action_model.shape}; expected ({cfg.num_actions},).")
                next_pos, next_vel = action_to_targets(action_model, cfg)
                pos_history.append(next_pos.copy())
                vel_history.append(next_vel.copy())
                target_pos, target_vel = sample_delayed_targets(
                    (pos_history, vel_history), actuator_groups, cfg.delay_min, cfg.delay_max, delay_rng
                )
                if viewer is not None:
                    display_current_command(cmd)

            if viewer is not None and counter % render_substeps == 0:
                render_utils.update_external_rendering(viewer, ctype="viewer")
                viewer.sync()
            if viewer is not None:
                sleep_time = cfg.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    print()
    if writer:
        writer.close()
        print(f"Video saved: {video_path}")
    if args.headless:
        print(
            f"Headless Sim2Sim finished: sim_time={data.time:.3f}s, steps={counter}, "
            f"base_xyz={np.array2string(data.qpos[:3], precision=4)}."
        )


if __name__ == "__main__":
    main()
