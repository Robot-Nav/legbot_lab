# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MoE-CTS 演员-评论家网络：教师编码器、学生 MoE 编码器、演员与评论家。"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn
from rsl_rl.networks.moe import MLP

from rsl_rl.networks import EmpiricalNormalization, L2Norm, SimNorm, MoE


class StudentMoEEncoder(nn.Module):
    """学生 MoE 编码器：通过混合专家网络将观测压缩为潜在向量。"""

    def __init__(
        self,
        expert_num,
        input_dim,
        hidden_dims,
        output_dim,
        activation='elu',
        norm_type='l2norm',
    ):
        """初始化学生 MoE 编码器。"""
        super().__init__()
        self.norm_layer = L2Norm() if norm_type == 'l2norm' else SimNorm()
        self.moe = MoE(
            expert_num=expert_num,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )
    
    def forward(self, obs):
        """前向传播，返回潜在向量与门控权重。"""
        latent, weights = self.moe(obs)
        latent = self.norm_layer(latent)
        return latent, weights


class ActorCriticMoECTS(nn.Module):
    """MoE-CTS 演员-评论家模型。"""

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
        teacher_encoder_hidden_dims: tuple[int] | list[int] = [512, 256],
        student_encoder_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        expert_num: int = 8,
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        noise_std_type: str = 'scalar',
        state_dependent_std: bool = False,
        latent_dim: int = 32,
        norm_type: str = 'l2norm',
        **kwargs: dict[str, Any],
    ) -> None:
        """初始化 MoE-CTS 演员-评论家模型。"""
        if kwargs:
            print(
                'ActorCriticMoECTS.__init__ 收到未预期参数，将被忽略：' + str([key for key in kwargs])
            )
        assert norm_type in ['l2norm', 'simnorm'], f'不支持的归一化类型：{norm_type}'
        assert 'policy' in obs.keys() and 'critic' in obs.keys() and 'single_obs' in obs.keys(), \
            "ActorCriticMoECTS 的 obs 必须包含 'policy'、'critic' 和 'single_obs' 键。"
        super().__init__()
        
        self.num_actions = num_actions

        # 获取观测维度
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups['policy']:
            assert len(obs[obs_group].shape) == 2, 'ActorCriticMoECTS 模块仅支持 1D 观测。'
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups['critic']:
            assert len(obs[obs_group].shape) == 2, 'ActorCriticMoECTS 模块仅支持 1D 观测。'
            num_critic_obs += obs[obs_group].shape[-1]
        
        # MLP 输入维度（教师、学生、演员、评论家）
        self.num_actor_obs = num_actor_obs
        self.num_single_obs = obs['single_obs'].shape[-1]
        mlp_input_dim_t = num_critic_obs
        mlp_input_dim_s = num_actor_obs
        mlp_input_dim_a = latent_dim + self.num_single_obs
        mlp_input_dim_c = latent_dim + num_critic_obs

        # 教师编码器
        self.teacher_encoder = nn.Sequential(
            MLP(mlp_input_dim_t, latent_dim, teacher_encoder_hidden_dims, activation=activation),
            L2Norm() if norm_type == 'l2norm' else SimNorm()
        )
        print(f'教师编码器：{self.teacher_encoder}')
        
        # 学生 MoE 编码器
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=mlp_input_dim_s,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )
        print(f'学生 MoE 编码器：{self.student_moe_encoder}')
        
        # 演员
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(mlp_input_dim_a, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)
        print(f'演员 MLP：{self.actor}')

        # 演员观测归一化
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
            self.single_obs_normalizer = EmpiricalNormalization(self.num_single_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            self.single_obs_normalizer = torch.nn.Identity()

        # 评论家
        self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)
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
        """重置状态（MoE-CTS 无循环状态，为空实现）。"""
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

    def _update_distribution(self, latent_and_obs: torch.Tensor) -> None:
        """根据潜在向量与观测更新动作分布。"""
        if self.state_dependent_std:
            # 计算均值与标准差
            mean_and_std = self.actor(latent_and_obs)
            if self.noise_std_type == 'scalar':
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == 'log':
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')
        else:
            # 计算均值
            mean = self.actor(latent_and_obs)
            # 计算标准差
            if self.noise_std_type == 'scalar':
                std = self.std.expand_as(mean)
            elif self.noise_std_type == 'log':
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')
        # 创建分布
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, is_teacher: bool, **kwargs: dict[str, Any]) -> torch.Tensor:
        """根据观测采样动作。"""
        single_obs = self.single_obs_normalizer(obs['single_obs'])
        if is_teacher:
            obs_c = self.get_critic_obs(obs)
            obs_c = self.critic_obs_normalizer(obs_c)
            latent = self.teacher_encoder(obs_c)
        else:
            with torch.no_grad():
                obs_a = self.get_actor_obs(obs)
                obs_a = self.actor_obs_normalizer(obs_a)
                latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        self._update_distribution(latent_and_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """推理阶段使用学生编码器获取确定性动作。"""
        single_obs = self.single_obs_normalizer(obs['single_obs'])
        obs_a = self.get_actor_obs(obs)
        obs_a = self.actor_obs_normalizer(obs_a)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        else:
            return self.actor(latent_and_obs)

    def evaluate(self, obs: TensorDict, is_teacher: bool, **kwargs: dict[str, Any]) -> torch.Tensor:
        """估计状态价值。"""
        obs_c = self.get_critic_obs(obs)
        obs_c = self.critic_obs_normalizer(obs_c)
        if is_teacher:
            latent = self.teacher_encoder(obs_c)
        else:
            obs_a = self.get_actor_obs(obs)
            obs_a = self.actor_obs_normalizer(obs_a)
            latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent.detach(), obs_c], dim=-1)
        return self.critic(latent_and_obs)

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
            self.single_obs_normalizer.update(obs['single_obs'])
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
