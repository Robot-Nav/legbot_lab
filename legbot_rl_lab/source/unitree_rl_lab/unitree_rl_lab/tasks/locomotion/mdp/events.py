# 自定义 MDP 事件项：为 sim2real 提供执行器、刚体属性、零位偏移等随机化。
from __future__ import annotations

import torch
from typing import Literal
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def randomize_actuator_torque_limit(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque_limit_scale: tuple[float, float] = (0.85, 1.15),
):
    """随机化 UnitreeActuator 的力矩限幅（Y1/Y2）。

    通过对峰值力矩参数 Y1 和 Y2 乘以随机缩放因子，模拟电机制造公差和
    磨损导致的力矩输出差异，提升 sim2real 策略鲁棒性。

    Args:
        env: 管理器式环境实例。
        env_ids: 需要随机化的环境索引。None 表示所有环境。
        asset_cfg: 场景实体配置，指定机器人。
        torque_limit_scale: 力矩缩放因子范围 (min, max)，1.0 表示标称值。
    """
    from unitree_rl_lab.assets.robots.unitree_actuators import UnitreeActuator

    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        if not isinstance(actuator, UnitreeActuator):
            continue
        # 采样缩放因子，每个环境独立采样
        scale = math_utils.sample_uniform(
            torque_limit_scale[0], torque_limit_scale[1], (len(env_ids),), device=asset.device
        )
        # _effort_y1 / _effort_y2 形状为 (num_envs, num_joints)
        actuator._effort_y1[env_ids] *= scale.unsqueeze(1)
        actuator._effort_y2[env_ids] *= scale.unsqueeze(1)


def randomize_actuator_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    min_delay_range: tuple[int, int] = (0, 2),
    max_delay_range: tuple[int, int] = (4, 8),
):
    """随机化执行器动作延迟范围，每个环境在 reset 时获得不同的延迟参数。

    模拟实物部署中策略输出到电机执行之间的通信/计算延迟不确定性。
    延迟以物理步数为单位（sim.dt=0.005s, 1步=5ms）。

    Args:
        env: 管理器式环境实例。
        env_ids: 需要随机化的环境索引。None 表示所有环境。
        asset_cfg: 场景实体配置，指定机器人。
        min_delay_range: 最小延迟的采样范围 (物理步数)。
        max_delay_range: 最大延迟的采样范围 (物理步数)。
    """
    from isaaclab.actuators.actuator_pd import DelayedPDActuator

    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        if not isinstance(actuator, DelayedPDActuator):
            continue
        # 为每个环境采样 min_delay 和 max_delay
        min_delays = torch.randint(
            min_delay_range[0], min_delay_range[1] + 1, (len(env_ids),), device=asset.device
        )
        max_delays = torch.randint(
            max_delay_range[0], max_delay_range[1] + 1, (len(env_ids),), device=asset.device
        )
        # 保证 max_delay >= min_delay
        max_delays = torch.maximum(max_delays, min_delays)
        # 更新每个环境延迟缓冲区的滞后步数
        # 采样当前回合的实际延迟
        time_lags = torch.zeros(len(env_ids), dtype=torch.int, device=asset.device)
        for i in range(len(env_ids)):
            time_lags[i] = torch.randint(min_delays[i].item(), max_delays[i].item() + 1, (1,)).item()
        actuator.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        actuator.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        actuator.efforts_delay_buffer.set_time_lag(time_lags, env_ids)


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """根据给定的操作和分布对数据张量进行随机化。"""
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(f"Unknown distribution: '{distribution}'.")

    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(f"Unknown operation: '{operation}'.")
    return data


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """随机化刚体惯性张量的对角分量 (xx, yy, zz)。

    仅在初始化时使用（CPU 张量赋值）。
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    inertias = asset.root_physx_view.get_inertias()
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    for idx in [0, 4, 8]:
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_action_joint_pos_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_term_name: str,
    offset_range: tuple[float, float],
):
    """随机化关节位置动作项的电机零位偏移，模拟编码器误差。"""
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
            f"Action term '{action_term_name}' does not expose a tensor '_offset', "
            "so it cannot be used for motor zero-offset randomization."
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
