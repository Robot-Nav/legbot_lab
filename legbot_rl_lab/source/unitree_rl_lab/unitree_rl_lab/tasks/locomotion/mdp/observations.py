# 自定义 MDP 观测项：提供步态相位等用于策略输入的额外观测。
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """计算基于回合时间的双足步态相位信号（sin/cos）。

    参数：
        env: 管理器式强化学习环境。
        period: 步态周期，单位：秒。

    返回：
        形状为 (num_envs, 2) 的张量，分别包含全局相位的正弦和余弦值。
    """
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
