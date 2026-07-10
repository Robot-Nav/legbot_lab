# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""对称性增强配置解析。"""

from __future__ import annotations

from rsl_rl.env import VecEnv


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """解析对称性增强配置。

    若算法配置中启用对称性增强，则将环境对象注入配置，
    供对称性函数根据观测项进行左右镜像处理。

    参数:
        alg_cfg: 算法配置字典。
        env: 环境对象。

    返回:
        解析后的算法配置字典。
    """
    # 启用对称性增强时，将环境对象写入配置
    if 'symmetry_cfg' in alg_cfg and alg_cfg['symmetry_cfg'] is not None:
        alg_cfg['symmetry_cfg']['_env'] = env
    else:
        alg_cfg['symmetry_cfg'] = None
    return alg_cfg
