# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""标准演员-评论家网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCritic(nn.Module):
    """标准演员-评论家模型。"""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        noise_std_type: str = 'scalar',
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """初始化标准演员-评论家模型。"""
        if kwargs:
            print(
                'ActorCritic.__init__ 收到未预期参数，将被忽略：' + str([key for key in kwargs])
            )
        super().__init__()

        # 获取观测维度
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups['policy']:
            assert len(obs[obs_group].shape) == 2, 'ActorCritic 模块仅支持 1D 观测。'
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups['critic']:
            assert len(obs[obs_group].shape) == 2, 'ActorCritic 模块仅支持 1D 观测。'
            num_critic_obs += obs[obs_group].shape[-1]

        # 演员
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f'演员 MLP：{self.actor}')

        # 演员观测归一化
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # 评论家
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f'评论家 MLP：{self.critic}')

        # 评论家观测归一化
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # 动作噪声
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == 'scalar':
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == 'log':
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')
        else:
            if self.noise_std_type == 'scalar':
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == 'log':
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')

        # 动作分布
        # 注意：在 update_distribution 中创建
        self.distribution = None

        # 禁用参数校验以加速
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """重置状态（标准模型无循环状态，为空实现）。"""
        pass

    def forward(self) -> NoReturn:
        """未实现：演员-评论家模型通过专用方法前向传播。"""
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        """动作分布均值。"""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """动作分布标准差。"""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """动作分布熵。"""
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """根据观测更新动作分布。"""
        if self.state_dependent_std:
            # 计算均值与标准差
            mean_and_std = self.actor(obs)
            if self.noise_std_type == 'scalar':
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == 'log':
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')
        else:
            # 计算均值
            mean = self.actor(obs)
            # 计算标准差
            if self.noise_std_type == 'scalar':
                std = self.std.expand_as(mean)
            elif self.noise_std_type == 'log':
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')
        # 创建分布
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """根据观测采样动作。"""
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """推理阶段获取确定性动作。"""
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """估计状态价值。"""
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """拼接演员观测组。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups['policy']]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """拼接评论家观测组。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups['critic']]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算动作对数概率。"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """更新观测归一化统计量。"""
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载演员-评论家模型参数。

        参数:
            state_dict: 模型状态字典。
            strict: 是否严格匹配状态字典键。

        返回:
            是否恢复之前的训练。该标志供 OnPolicyRunner 的 load 函数使用，
            以决定如何加载其他参数（例如蒸馏相关参数）。
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
