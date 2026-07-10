# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""存放机器人模型、传感器等资源的配置。"""

import os
import toml

ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
"""扩展源码目录路径。"""

ISAACLAB_ASSETS_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../resources'))
"""扩展数据资源目录路径。"""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, 'config', 'extension.toml'))
"""从 extension.toml 解析得到的扩展元数据字典。"""

__version__ = ISAACLAB_ASSETS_METADATA['package']['version']
