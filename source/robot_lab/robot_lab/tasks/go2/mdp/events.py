# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """通过加、缩放或设置随机值来随机化刚体惯性张量。

    本函数仅随机化刚体惯性张量的对角分量（xx, yy, zz）。
    从给定分布参数中采样随机值，并根据 operation 将其加到、缩放或设置到物理仿真中。

    .. tip::
        本函数使用 CPU 张量设置刚体惯性，建议仅在环境初始化时使用。
    """
    # 提取使用到的量（用于类型提示）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析环境 ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析刚体索引
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 获取当前刚体惯性张量（articulations 为 (num_assets, num_bodies, 9)）
    inertias = asset.root_physx_view.get_inertias()

    # 在默认值上应用随机化
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # 随机化每个对角元素（xx, yy, zz -> 索引 0, 4, 8）
    for idx in [0, 4, 8]:
        # 提取并随机化特定对角元素
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # 将随机化后的值写回惯性张量
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # 将惯性张量设置到物理仿真中
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """随机化刚体质心位置。

    本函数可对物理仿真中刚体的质心位置进行随机化，可通过加、缩放或设置从指定分布采样的随机值。

    .. tip::
        由于直接修改物理属性，建议在初始化或离线调整时使用。
    """
    # 提取目标资源（Articulation 或 RigidObject）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析环境 ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析刚体索引
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 获取当前质心偏移（形状：num_assets, num_bodies, 3）
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # 独立随机化 x, y, z
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # 将随机化后的质心偏移写入仿真
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""内部辅助函数。"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """根据指定操作与分布对数据进行随机化。

    参数:
        data: 待随机化的数据张量，形状为 (dim_0, dim_1)。
        distribution_parameters: 采样分布参数。
        dim_0_ids: 第一维随机化索引。
        dim_1_ids: 第二维随机化索引。
        operation: 对数据执行的操作，可选 'add'、'scale'、'abs'。
        distribution: 随机值采样分布，可选 'uniform'、'log_uniform'、'gaussian'。

    返回:
        随机化后的数据张量，形状为 (dim_0, dim_1)。

    异常:
        NotImplementedError: 不支持的操作或分布。
    """
    # 解析形状
    # -- 维度 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- 维度 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # 解析分布
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"未知分布: '{distribution}'，请使用 'uniform'、'log_uniform' 或 'gaussian'。"
        )
    # 执行操作
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"未知操作: '{operation}'，请使用 'add'、'scale' 或 'abs'。"
        )
    return data


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """在指定范围内均匀随机重置资源根状态。

    本函数对资源的根位置与速度进行随机化：

    * 从给定范围采样位置偏移，加到默认根位置上，再写入物理仿真。
    * 从给定范围采样朝向并写入物理仿真。
    * 从给定范围采样速度并写入物理仿真。

    位置与速度范围通过字典为每个轴和旋转指定，键为 ``x``、``y``、``z``、``roll``、``pitch``、``yaw``，
    值为 ``(min, max)`` 元组。若字典缺少某键，则该轴位置或速度置零。

    注意：若存在 "pits" 地形，位于该地形的环境将重置为默认状态而不添加随机扰动，以避免机器人落入坑中。
    """
    # 提取使用到的量（用于类型提示）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 分离坑地形与非坑地形环境
    # 检查哪些环境被分配到 pits 地形（不随机重置）
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]

    # 将坑地形环境重置为默认状态（无随机扰动）
    if len(pit_env_ids) > 0:
        root_states = asset.data.default_root_state[pit_env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)

    # 对非坑地形环境添加随机扰动后重置
    if len(non_pit_env_ids) > 0:
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()

        # 位姿
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # 速度
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        velocities = root_states[:, 7:13] + rand_samples

        # 写入物理仿真
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)

def randomize_action_joint_pos_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_term_name: str,
    offset_range: tuple[float, float],
):
    """对关节位置动作项的电机零偏进行随机化。"""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    action_term = env.action_manager.get_term(action_term_name)
    if not hasattr(action_term, "_offset") or not isinstance(action_term._offset, torch.Tensor):
        raise TypeError(
            f"动作项 '{action_term_name}' 未暴露张量 '_offset'，"
            "无法用于电机零偏随机化。"
        )

    cache_name = f"_default_action_offset_{action_term_name}"
    default_offset = getattr(env, cache_name, None)
    if default_offset is None or default_offset.shape != action_term._offset.shape:
        default_offset = action_term._offset.clone()
        setattr(env, cache_name, default_offset)

    offset_noise = math_utils.sample_uniform(
        offset_range[0],
        offset_range[1],
        (len(env_ids), action_term._offset.shape[1]),
        device=env.device,
    )
    action_term._offset[env_ids] = default_offset[env_ids] + offset_noise
