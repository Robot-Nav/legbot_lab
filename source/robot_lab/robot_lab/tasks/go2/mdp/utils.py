# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""地形感知操作的工具函数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:
    """计算某地形类型所占列范围的辅助函数。

    参数:
        terrain_cfg: 地形生成器配置。
        terrain_name: 地形名称。
        device: Torch 设备。

    返回:
        (col_start, col_end) 元组；若未找到地形则返回 None。
    """
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device)
    proportions = proportions / proportions.sum()
    cumsum_props = torch.cumsum(proportions, dim=0)

    terrain_idx = sub_terrain_names.index(terrain_name)
    # 使用 round() 而非 int() 以正确分配列
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    return (col_start, col_end)


def is_env_assigned_to_terrain(env: ManagerBasedEnv, terrain_name: str) -> torch.Tensor:
    """检查哪些环境在初始化时被分配到指定地形类型。

    每个环境在初始化时被分配到特定地形单元。
    本函数返回布尔掩码，指示哪些环境被分配到给定地形类型。

    参数:
        env: 环境实例。
        terrain_name: 要检查的地形名称，例如 "pits"、"stairs"。

    返回:
        形状为 (num_envs,) 的布尔张量，True 表示该环境被分配到该地形。
    """
    # 检查地形与地形生成器是否可用
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range
    # terrain_types 直接存储列索引，因此只需判断是否在范围内
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    """检查哪些环境当前被分配到指定地形类型。

    地形导入器会跟踪每个环境当前激活的地形列。
    本辅助函数直接使用该分配信息，而非根据机器人世界坐标推断。

    参数:
        env: 环境实例。
        terrain_name: 要检查的地形名称，例如 "pits"、"stairs"。
        asset_name: 机器人资源名称，默认为 "robot"。

    返回:
        形状为 (num_envs,) 的布尔张量，True 表示机器人当前位于该地形。
    """
    # 检查地形与地形生成器是否可用
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range

    # 地形导入器已跟踪每个环境当前激活的地形列。
    # 直接使用该信息可保持与课程更新一致，并避免按世界坐标最近瓦片分类带来的误判。
    del asset_name
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


"""指令工具函数"""

def sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """从 [cfg_min, -limit_bound] U [limit_bound, cfg_max] 均匀采样"""
    width_neg = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_pos = torch.nn.functional.relu(cfg_max - limit_bound)
    
    total_width = width_neg + width_pos + 1e-6 # 加极小值防除零
    u = torch.rand(len(env_ids), device=device) * total_width
    
    samples = torch.where(
        u < width_neg, 
        cfg_min + u, 
        cfg_max - width_pos + (u - width_neg)
    )
    return samples

def sample_single_interval(env_ids, cfg_min, cfg_max, device):
    """从 [cfg_min, cfg_max] 均匀采样"""
    r = torch.rand(len(env_ids), device=device)
    samples = cfg_min + r * (cfg_max - cfg_min)
    return samples
