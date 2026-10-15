"""Wheel-legged stance rewards using measured joints and wheel axle geometry.

Quaternion convention is (w, x, y, z). These functions contain no simulator
startup code and can be tested with CPU tensors.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _standing_mask(
    command: torch.Tensor,
    linear_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    # Linear and angular speeds have different units; test them separately.
    return (torch.linalg.vector_norm(command[:, :2], dim=1) < linear_threshold) & (
        torch.abs(command[:, 2]) < yaw_threshold
    )


def _flat_upright_weight(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None,
    flatness_std: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    if sensor_cfg is None:
        return upright
    # Height variation weakens the flat-ground preference on slopes and steps.
    # Invalid scans disable this auxiliary stance preference, never yield NaNs.
    heights = env.scene[sensor_cfg.name].data.ray_hits_w[..., 2]
    valid = torch.isfinite(heights).all(dim=1) & (torch.abs(heights) < 1e6).all(dim=1)
    safe_heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
    safe_heights = safe_heights.clamp(-1e6, 1e6)
    variance = torch.mean(torch.square(safe_heights - safe_heights.mean(dim=1, keepdim=True)), dim=1)
    flat = torch.exp(-variance / flatness_std**2)
    return upright * flat * valid.to(upright.dtype)


def joint_pose_l1_standing_scaled(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 2.0,
    linear_threshold: float = 0.1,
    yaw_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg | None = None,
    flatness_std: float = 0.03,
) -> torch.Tensor:
    """Increase measured joint-pose cost during quiet, nearly flat standing.

    Uses actual joint positions, so PD targets may compensate for gravity.
    Forward motion is included in standing detection, unlike the legacy hip
    reward that looked only at lateral/yaw commands.
    """
    asset = env.scene[asset_cfg.name]
    deviation = torch.sum(torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ), dim=1)
    command = env.command_manager.get_command(command_name)
    standing = _standing_mask(command, linear_threshold, yaw_threshold)
    extra = standing.to(deviation.dtype) * _flat_upright_weight(env, asset_cfg, sensor_cfg, flatness_std)
    return deviation * (1.0 + (stand_still_scale - 1.0) * extra)


def wheel_axis_alignment_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.2,
    deadzone_degrees: float = 2.0,
    straight_scale: float = 0.25,
    linear_threshold: float = 0.1,
    yaw_threshold: float = 0.1,
    flatness_std: float = 0.03,
) -> torch.Tensor:
    """Penalize wheel camber and toe from physical wheel-link orientations.

    Wheel axles are link-local +/-Y for W1W. Sign and wheel spin do not affect
    this cost. Camber uses world Z; toe uses the base forward direction.
    Primarily active at rest, weak during straight motion, off for lateral/yaw
    commands, and reduced on non-flat terrain. A small dead zone allows balance.
    """
    asset = env.scene[asset_cfg.name]
    wheel_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    w, x, y, z = wheel_quat.unbind(dim=-1)
    axle_w = torch.stack((2.0 * (x*y - w*z), 1.0 - 2.0 * (x*x + z*z),
                          2.0 * (y*z + w*x)), dim=-1)
    bw, bx, by, bz = asset.data.root_quat_w.unbind(dim=-1)
    forward_w = torch.stack((1.0 - 2.0 * (by*by + bz*bz), 2.0 * (bx*by + bw*bz),
                             2.0 * (bx*bz - bw*by)), dim=-1)
    toe = torch.sum(axle_w * forward_w[:, None, :], dim=-1)
    camber = axle_w[..., 2]
    deadzone = math.sin(math.radians(deadzone_degrees))
    error = torch.mean(
        torch.square(torch.relu(torch.abs(toe) - deadzone))
        + torch.square(torch.relu(torch.abs(camber) - deadzone)), dim=1,
    ) / std**2
    command = env.command_manager.get_command(command_name)
    standing = _standing_mask(command, linear_threshold, yaw_threshold)
    straight = (torch.abs(command[:, 1]) < linear_threshold) & (
        torch.abs(command[:, 2]) < yaw_threshold
    )
    context = torch.where(standing, torch.ones_like(error), straight_scale * straight.to(error.dtype))
    return error * context * _flat_upright_weight(env, asset_cfg, sensor_cfg, flatness_std)


def standing_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    leg_std: float = 1.0,
    wheel_std: float = 5.0,
    linear_threshold: float = 0.1,
    yaw_threshold: float = 0.1,
    flatness_std: float = 0.03,
) -> torch.Tensor:
    """Discourage unnecessary leg motion and wheel spin only at zero command."""
    asset = env.scene[asset_cfg.name]
    leg_cost = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids] / leg_std), dim=1)
    wheel_cost = torch.mean(torch.square(asset.data.joint_vel[:, wheel_asset_cfg.joint_ids] / wheel_std), dim=1)
    command = env.command_manager.get_command(command_name)
    standing = _standing_mask(command, linear_threshold, yaw_threshold)
    return (leg_cost + wheel_cost) * standing.to(leg_cost.dtype) * _flat_upright_weight(
        env, asset_cfg, sensor_cfg, flatness_std,
    )
