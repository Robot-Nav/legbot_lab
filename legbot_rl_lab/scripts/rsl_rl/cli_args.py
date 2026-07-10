# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 训练与推理脚本的命令行参数解析与配置更新工具。"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """向参数解析器添加 RSL-RL 相关参数。"""
    # 新建参数分组
    arg_group = parser.add_argument_group("rsl_rl", description="RSL-RL 智能体参数。")
    # 实验参数
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="实验文件夹名称，用于存放日志。"
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="日志目录的运行名后缀。")
    # 加载参数
    arg_group.add_argument("--resume", action="store_true", default=False, help="是否从检查点恢复训练。")
    arg_group.add_argument("--load_run", type=str, default=None, help="要恢复的运行文件夹名称。")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="要恢复的检查点文件。")
    # 日志参数
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="使用的日志后端。"
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="使用 wandb 或 neptune 时的项目名称。"
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """根据任务名与命令行参数解析 RSL-RL 配置。"""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 加载默认配置
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if rslrl_cfg.experiment_name == "":
        rslrl_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """根据命令行参数更新 RSL-RL 配置。"""
    # 使用命令行参数覆盖默认配置
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # seed 为 -1 时随机采样
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # 为 wandb 与 neptune 设置项目名称
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    return agent_cfg
