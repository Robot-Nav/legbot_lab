from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            # Expand range symmetrically: increase max and decrease min by 0.1
            delta = 0.1
            new_x_min = max(ranges.lin_vel_x[0] - delta, limit_ranges.lin_vel_x[0])
            new_x_max = min(ranges.lin_vel_x[1] + delta, limit_ranges.lin_vel_x[1])
            ranges.lin_vel_x = [new_x_min, new_x_max]
            new_y_min = max(ranges.lin_vel_y[0] - delta, limit_ranges.lin_vel_y[0])
            new_y_max = min(ranges.lin_vel_y[1] + delta, limit_ranges.lin_vel_y[1])
            ranges.lin_vel_y = [new_y_min, new_y_max]

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta = 0.1
            new_z_min = max(ranges.ang_vel_z[0] - delta, limit_ranges.ang_vel_z[0])
            new_z_max = min(ranges.ang_vel_z[1] + delta, limit_ranges.ang_vel_z[1])
            ranges.ang_vel_z = [new_z_min, new_z_max]

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
