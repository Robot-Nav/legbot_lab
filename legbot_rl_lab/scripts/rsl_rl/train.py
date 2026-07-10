# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 训练强化学习策略的入口脚本。"""

"""首先启动 Isaac Sim 仿真器。"""


import gymnasium as gym
import pathlib
import sys

# 将上级目录加入模块搜索路径，以便导入 list_envs
sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

# 收集所有以 Unitree 命名且非 Isaac 前缀的任务
tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip

# 构建命令行参数解析器
parser = argparse.ArgumentParser(description="使用 RSL-RL 训练强化学习策略。")
parser.add_argument("--video", action="store_true", default=False, help="训练期间录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="每段录制视频的步数。")
parser.add_argument("--video_interval", type=int, default=2000, help="视频录制间隔步数。")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量。")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="任务名称。")
parser.add_argument("--seed", type=int, default=None, help="环境随机种子。")
parser.add_argument("--max_iterations", type=int, default=None, help="策略训练迭代次数。")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="使用多 GPU 或多节点分布式训练。"
)
# 追加 RSL-RL 专有参数
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# 若需要录制视频则强制启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 清空 sys.argv，仅保留脚本名与 Hydra 参数
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""检查 RSL-RL 最低版本要求。"""

import importlib.metadata as metadata
import platform

from packaging import version

# 分布式训练所需的最低 RSL-RL 版本
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"请安装正确版本的 RSL-RL。\n当前版本: '{installed_version}'"
        f" 所需版本: '{RSL_RL_VERSION}'。\n安装命令:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""后续逻辑。"""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: 考虑在终端打印实验名称。

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

# 设置 CUDA 计算精度与性能选项
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """RSL-RL 训练主流程。"""
    # 使用非 Hydra 命令行参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境种子
    # 注意：环境初始化中存在随机化操作，因此在此设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 多 GPU 分布式训练配置
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 为不同进程设置不同种子以保证多样性
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 指定实验日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 实验日志目录: {log_root_path}")
    # 单次运行的日志目录：时间戳_运行名
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Ray Tune 可据此提取实验名称
    print(f"命令行请求的精确实验名: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 若环境为多智能体，则转换为单智能体形式
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 保存恢复路径，因为后续创建 log_dir 会改变路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 包装环境以录制视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 训练期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 使用 RSL-RL 的环境包装器
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建 RSL-RL 训练器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # 将当前 Git 状态写入日志
    runner.add_git_repo_to_log(__file__)
    # 加载检查点
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: 从检查点加载模型: {resume_path}")
        # 加载已训练模型
        runner.load(resume_path)

    # 将配置导出到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # 将环境配置文件复制到日志目录
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # 启动训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 关闭仿真器
    env.close()


if __name__ == "__main__":
    # 执行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
