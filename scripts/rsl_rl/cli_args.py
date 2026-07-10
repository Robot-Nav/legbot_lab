# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL 命令行参数解析与配置更新工具。"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """向解析器添加 RSL-RL 相关命令行参数。

    Args:
        parser: 要添加参数的解析器。
    """
    # 创建新的参数组
    arg_group = parser.add_argument_group('rsl_rl', description='Arguments for RSL-RL agent.')
    # -- 实验相关参数
    arg_group.add_argument(
        '--experiment_name', type=str, default=None, help='Name of the experiment folder where logs will be stored.'
    )
    arg_group.add_argument('--run_name', type=str, default=None, help='Run name suffix to the log directory.')
    # -- 加载相关参数
    arg_group.add_argument('--resume', action='store_true', default=False, help='Whether to resume from a checkpoint.')
    arg_group.add_argument('--load_run', type=str, default=None, help='Name of the run folder to resume from.')
    arg_group.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file to resume from.')
    # -- 日志相关参数
    arg_group.add_argument(
        '--logger', type=str, default=None, choices={'wandb', 'tensorboard', 'neptune'}, help='Logger module to use.'
    )
    arg_group.add_argument(
        '--log_project_name', type=str, default=None, help='Name of the logging project when using wandb or neptune.'
    )
    arg_group.add_argument(
        '--robogauge', action='store_true', default=False, help='Enable robogauge evaluation interface.'
    )
    arg_group.add_argument(
        '--robogauge_port', type=int, default=9973, help='Port for robogauge evaluation interface.'
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> 'RslRlOnPolicyRunnerCfg':
    """基于输入解析 RSL-RL 智能体配置。

    Args:
        task_name: 环境任务名。
        args_cli: 命令行参数。

    Returns:
        基于输入解析得到的 RSL-RL 智能体配置。
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 加载默认配置
    rslrl_cfg: 'RslRlOnPolicyRunnerCfg' = load_cfg_from_registry(task_name, 'rsl_rl_cfg_entry_point')
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: 'RslRlOnPolicyRunnerCfg', args_cli: argparse.Namespace):
    """基于命令行参数更新 RSL-RL 智能体配置。

    Args:
        agent_cfg: RSL-RL 智能体配置。
        args_cli: 命令行参数。

    Returns:
        更新后的 RSL-RL 智能体配置。
    """
    # 使用命令行参数覆盖默认配置
    if hasattr(args_cli, 'seed') and args_cli.seed is not None:
        # seed = -1 时随机采样一个种子
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
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
    # 为 wandb 与 neptune 设置项目名
    if agent_cfg.logger in {'wandb', 'neptune'} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
