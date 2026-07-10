# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""LegBot 四足机器人任务包。

复用 Go2 的 MDP 模块（奖励、观测、事件等），因为 LegBot 具有相同的
运动学结构与关节命名约定。
"""

import os
import toml
import gymnasium as gym

from isaaclab_tasks.utils import import_packages

gym.register(
    id='RobotLab-Legbot-v0',
    entry_point='robot_lab.tasks.legbot.env.legbot_env:LegbotEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.env_cfg:LegbotEnvCfg',
        'rsl_rl_cfg_entry_point': f'{__name__}.rsl_rl_cfg:MoECTSRunnerCfg',
    },
)

# 防止从子包导入配置的黑名单
_BLACKLIST_PKGS = ['utils']
import_packages(__name__, _BLACKLIST_PKGS)
