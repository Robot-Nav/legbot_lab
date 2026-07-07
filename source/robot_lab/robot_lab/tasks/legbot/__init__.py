# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""LegBot quadruped robot task registration.

Reuses the Go2 MDP module (rewards, observations, events, etc.) since LegBot
shares the same kinematic structure and joint naming convention.
"""

import os
import toml
import gymnasium as gym

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##
gym.register(
    id="RobotLab-Legbot-v0",
    entry_point="robot_lab.tasks.legbot.env.legbot_env:LegbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LegbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:MoECTSRunnerCfg",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
