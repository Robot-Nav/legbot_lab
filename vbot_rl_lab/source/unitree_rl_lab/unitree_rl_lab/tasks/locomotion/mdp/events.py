"""Custom domain randomization event functions for vbot training."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_motor_strength(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    strength_distribution_params: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize motor strength (torque scaling) for the robot's joints.

    This simulates variations in motor torque output due to battery voltage,
    motor wear, etc. The strength multiplier is applied to the computed torques
    before they are sent to the simulation.

    Args:
        env: The environment instance.
        env_ids: The environment ids for which to randomize. If None, all envs.
        asset_cfg: The asset configuration.
        strength_distribution_params: The (min, max) range for the strength multiplier.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Initialize motor strengths and original effort limits if not exists
    if not hasattr(env, '_motor_strengths'):
        env._motor_strengths = torch.ones(
            env.num_envs, asset.num_joints, device=env.device
        )
        # Save original effort_limit_sim for each actuator to avoid cumulative scaling
        env._original_effort_limits = {}
        for actuator_name, actuator in asset.actuators.items():
            env._original_effort_limits[actuator_name] = actuator.effort_limit_sim.clone()

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Sample strength multipliers
    strengths = torch.rand(
        (len(env_ids), asset.num_joints),
        device=env.device,
    ) * (strength_distribution_params[1] - strength_distribution_params[0]) + strength_distribution_params[0]
    env._motor_strengths[env_ids] = strengths

    # Apply strength by scaling the effort limits from ORIGINAL values (not cumulative)
    for actuator_name, actuator in asset.actuators.items():
        joint_ids = actuator.joint_indices
        if isinstance(joint_ids, slice):
            actuator_strengths = env._motor_strengths[env_ids]
        else:
            actuator_strengths = env._motor_strengths[env_ids][:, joint_ids]
        # Scale from original limits, not current limits
        original_limits = env._original_effort_limits[actuator_name][env_ids]
        scaled_limits = original_limits * actuator_strengths
        actuator.effort_limit_sim[env_ids] = scaled_limits


def randomize_action_delay(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_delay_steps: int = 0,
    max_delay_steps: int = 4,
) -> None:
    """Randomize action delay by storing a history buffer and using delayed actions.

    This simulates communication/computation latency in the control loop.
    At each env reset, a random delay (in physics steps) is sampled for each env.
    During stepping, the action from `delay` steps ago is used instead of the current one.

    Uses vectorized indexing instead of per-env for loop for efficiency.

    Args:
        env: The environment instance.
        env_ids: The environment ids for which to randomize. If None, all envs.
        asset_cfg: The asset configuration.
        min_delay_steps: Minimum delay in physics steps.
        max_delay_steps: Maximum delay in physics steps.
    """
    if not hasattr(env, '_action_delay_buffer'):
        action_dim = env.action_manager.action.shape[1]
        env._action_delay_buffer = torch.zeros(
            max_delay_steps + 1, env.num_envs, action_dim, device=env.device
        )
        env._action_delay_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._max_delay_steps = max_delay_steps
        # Store original process_action method (仅保存一次，防止重复包装)
        env._original_process_action = env.action_manager.process_action

        # Monkey-patch process_action to apply delay before processing
        def _delayed_process_action(action):
            delayed_action = _apply_action_delay(env, action)
            env._original_process_action(delayed_action)

        env.action_manager.process_action = _delayed_process_action

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Sample random delay for each env
    env._action_delay_steps[env_ids] = torch.randint(
        min_delay_steps, max_delay_steps + 1, (len(env_ids),), device=env.device
    )
    # Reset delay buffer for these envs
    env._action_delay_buffer[:, env_ids, :] = 0.0


def _apply_action_delay(env, current_actions):
    """Apply action delay by using actions from the delay buffer (vectorized)."""
    if not hasattr(env, '_action_delay_buffer'):
        return current_actions

    # Roll buffer: shift all entries by one step
    env._action_delay_buffer = torch.roll(env._action_delay_buffer, -1, dims=0)
    env._action_delay_buffer[-1] = current_actions.clone()

    # Vectorized delay index: for each env, pick from buffer[max_delay - delay]
    max_delay = env._max_delay_steps
    delay_steps = env._action_delay_steps  # [num_envs]
    # Index into buffer: buffer[max_delay - delay, env_idx]
    buffer_indices = max_delay - delay_steps  # [num_envs]
    env_indices = torch.arange(env.num_envs, device=env.device)
    delayed_actions = env._action_delay_buffer[buffer_indices, env_indices]

    return delayed_actions


def reset_action_smoothness_buffer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Reset the action smoothness prev_prev buffer on env reset."""
    if hasattr(env, '_action_smoothness_prev_prev') and env._action_smoothness_prev_prev is not None:
        if env_ids is None:
            env._action_smoothness_prev_prev[:] = 0.0
        else:
            env._action_smoothness_prev_prev[env_ids] = 0.0
