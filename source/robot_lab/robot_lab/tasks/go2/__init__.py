# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 机器人任务包，负责注册 Gym 环境并导入子包配置。"""

import os
import toml
import gymnasium as gym

from isaaclab_tasks.utils import import_packages

gym.register(
    id='RobotLab-Go2-v0',
    entry_point='robot_lab.tasks.go2.env.go2_env:Go2Env',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.env_cfg:Go2EnvCfg',
        'rsl_rl_cfg_entry_point': f'{__name__}.rsl_rl_cfg:MoECTSRunnerCfg',
    },
)

# 防止从子包导入配置的黑名单
_BLACKLIST_PKGS = ['utils']
import_packages(__name__, _BLACKLIST_PKGS)
