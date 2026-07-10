# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""机器人任务实现包，负责注册并导入各子任务配置。"""

import os
import toml

from isaaclab_tasks.utils import import_packages

# 防止从子包导入配置的黑名单
_BLACKLIST_PKGS = ['utils']
import_packages(__name__, _BLACKLIST_PKGS)
