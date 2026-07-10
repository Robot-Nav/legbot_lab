# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 加载检查点并运行策略推理的入口脚本。"""

"""首先启动 Isaac Sim 仿真器。"""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip

# 构建命令行参数解析器
parser = argparse.ArgumentParser(description="使用 RSL-RL 运行已训练的策略。")
parser.add_argument("--video", action="store_true", default=False, help="推理期间录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="每段录制视频的步数。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用 Fabric，使用标准 USD I/O。"
)
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="使用 Nucleus 上发布的预训练检查点。",
)
parser.add_argument("--real-time", action="store_true", default=False, help="尽可能按真实时间运行。")
# 追加 RSL-RL 专有参数
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 若需要录制视频则强制启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""后续逻辑。"""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """RSL-RL 推理主流程。"""
    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 指定实验日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 从目录加载实验: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] 当前任务暂无可用的预训练检查点。")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 若环境为多智能体，则转换为单智能体形式
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装环境以录制视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 推理期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 使用 RSL-RL 的环境包装器
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: 从检查点加载模型: {resume_path}")
    # 加载已训练模型
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"不支持的训练器类型: {agent_cfg.class_name}")
    runner.load(resume_path)

    # 获取推理策略
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 提取神经网络模块
    # 使用 try-except 保持向后兼容
    try:
        # rsl-rl 2.3 及以上版本
        policy_nn = runner.alg.policy
    except AttributeError:
        # rsl-rl 2.2 及以下版本
        policy_nn = runner.alg.actor_critic

    # 提取观测归一化器
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # 导出策略为 JIT 与 ONNX 格式
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # 重置环境
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # 运行仿真循环
    while simulation_app.is_running():
        start_time = time.time()
        # 推理模式下执行策略
        with torch.inference_mode():
            # 策略输出动作
            actions = policy(obs)
            # 环境步进
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # 录制完一段视频后退出
            if timestep == args_cli.video_length:
                break

        # 真实时间评估的延时
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭仿真器
    env.close()


if __name__ == "__main__":
    # 执行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
