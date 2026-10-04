# Copyright (c) 2026 Robot-Nav
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # wheel_asset_cfg.joint_ids index the articulation, not the selected observation columns.
    all_ids = range(asset.data.joint_pos.shape[1])
    joint_ids = list(all_ids[asset_cfg.joint_ids]) if isinstance(asset_cfg.joint_ids, slice) else list(asset_cfg.joint_ids)
    wheel_ids = set(all_ids[wheel_asset_cfg.joint_ids] if isinstance(wheel_asset_cfg.joint_ids, slice) else wheel_asset_cfg.joint_ids)
    wheel_columns = [column for column, joint_id in enumerate(joint_ids) if joint_id in wheel_ids]
    joint_pos_rel[:, wheel_columns] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]

def foot_contact_force_norm(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history # [B, T_hist, num_bodies, 3]
    
    contact_force_norm = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1) # [B, T_hist, num_legs]
    max_contact_force_norm, _ = torch.max(contact_force_norm, dim=1)  # [B, num_legs]
    contact_force_norm = torch.concat([max_contact_force_norm.unsqueeze(1), contact_force_norm], dim=1)  # [B, T_hist+1, num_legs]
    
    return contact_force_norm.flatten(start_dim=-2)  # [B, (T_hist+1)*num_legs]
