# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""向量化环境抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class VecEnv(ABC):
    """向量化环境抽象类。

    向量化环境是一组同步运行的环境集合：对所有环境应用相同类型的动作，
    并返回相同类型的观测。
    """

    num_envs: int
    """环境数量。"""

    num_actions: int
    """动作维度。"""

    max_episode_length: int | torch.Tensor
    """最大回合长度。

    可以是标量（所有环境相同）或张量（每个环境独立），
    用于支持动态回合长度。
    """

    episode_length_buf: torch.Tensor
    """当前各环境的回合长度缓冲区。"""

    device: torch.device | str
    """计算设备。"""

    cfg: dict | object
    """配置对象。"""

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """返回当前观测。

        返回:
            环境当前观测。
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """将动作应用到环境并推进一步。

        参数:
            actions: 输入动作，形状：(num_envs, num_actions)。

        返回:
            observations: 环境观测。
            rewards: 环境奖励，形状：(num_envs,)。
            dones: 环境终止标志，形状：(num_envs,)。
            extras: 环境额外信息。

        观测说明:
            返回的 TensorDict 通常包含多个观测组。runner 配置中的 `obs_groups`
            定义了各观测集合使用哪些观测组，即将可用观测组映射到特定用途的观测集合。
            RSL-RL 当前使用的观测集合包括：

            - 'policy': 指定观测组作为演员/学生网络输入。
            - 'critic': 指定观测组作为评论家网络输入。
            - 'teacher': 指定观测组作为教师网络输入。
            - 'rnd_state': 指定观测组作为 RND 网络输入。

            不完整或错误的配置会在 `rsl_rl/utils/utils.py` 的 `resolve_obs_groups()` 中处理。

        Extras 说明:
            extras 字典包含回合奖励、回合长度等指标。RSL-RL 使用以下键：

            - 'time_outs' (torch.Tensor): 因达到时间限制而触发的超时终止标志，
              与环境到达真实终止状态不同，常用于固定长度回合。

            - 'log' (dict[str, float | torch.Tensor]): 用于日志与调试的附加信息。
              键应为字符串，建议以 '/' 开头进行命名空间划分；值为标量或张量，
              张量会自动取均值后记录。
        """
        raise NotImplementedError
