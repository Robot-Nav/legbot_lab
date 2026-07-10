# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""带 CNN 编码器的标准演员-评论家网络，用于处理 2D 图像观测。"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization

from .actor_critic import ActorCritic


class ActorCriticCNN(ActorCritic):
    """带 CNN 编码的演员-评论家模型。

    支持 1D 向量观测与 2D 图像观测混合输入，
    图像观测经 CNN 编码后与向量观测拼接，再传入 MLP。
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_cnn_cfg: dict[str, dict] | dict | None = None,
        critic_cnn_cfg: dict[str, dict] | dict | None = None,
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        noise_std_type: str = 'scalar',
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """初始化 CNN 演员-评论家模型。"""
        if kwargs:
            print(
                'ActorCriticCNN.__init__ 收到未预期参数，将被忽略：'
                + str([key for key in kwargs])
            )
        super(ActorCritic, self).__init__()

        # 获取观测维度，区分 1D 向量与 2D 图像
        self.obs_groups = obs_groups
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        actor_in_dims_2d = []
        actor_in_channels_2d = []
        self.actor_obs_groups_2d = []
        for obs_group in obs_groups['policy']:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.actor_obs_groups_2d.append(obs_group)
                actor_in_dims_2d.append(obs[obs_group].shape[2:4])
                actor_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f'观测组 {obs_group} 的形状无效：{obs[obs_group].shape}')

        num_critic_obs_1d = 0
        self.critic_obs_groups_1d = []
        critic_in_dims_2d = []
        critic_in_channels_2d = []
        self.critic_obs_groups_2d = []
        for obs_group in obs_groups['critic']:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.critic_obs_groups_2d.append(obs_group)
                critic_in_dims_2d.append(obs[obs_group].shape[2:4])
                critic_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f'观测组 {obs_group} 的形状无效：{obs[obs_group].shape}')

        # 确认至少存在一组 2D 观测，否则应使用 ActorCritic
        assert self.actor_obs_groups_2d or self.critic_obs_groups_2d, (
            '未提供 2D 观测，若仅使用 1D 观测，请改用 ActorCritic 模块。'
        )

        # 演员 CNN
        if self.actor_obs_groups_2d:
            # 解析演员 CNN 配置
            assert actor_cnn_cfg is not None, '演员存在 2D 观测时需提供 CNN 配置。'
            # 若传入单一配置字典，则为每个 2D 观测组复制一份
            if not all(isinstance(v, dict) for v in actor_cnn_cfg.values()):
                actor_cnn_cfg = {group: actor_cnn_cfg for group in self.actor_obs_groups_2d}
            # 确认配置数量与 2D 观测组数量一致
            assert len(actor_cnn_cfg) == len(self.actor_obs_groups_2d), (
                'CNN 配置数量必须与演员 2D 观测组数量一致。'
            )

            # 为每个演员 2D 观测组创建 CNN
            self.actor_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.actor_obs_groups_2d):
                self.actor_cnns[obs_group] = CNN(
                    input_dim=actor_in_dims_2d[idx],
                    input_channels=actor_in_channels_2d[idx],
                    **actor_cnn_cfg[obs_group],
                )
                print(f'演员 CNN（{obs_group}）：{self.actor_cnns[obs_group]}')
                # 累计 CNN 输出维度
                if self.actor_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.actor_cnns[obs_group].output_dim)
                else:
                    raise ValueError('演员 CNN 输出必须在传入 MLP 前展平。')
        else:
            self.actor_cnns = None
            encoding_dim = 0

        # 演员 MLP
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs_1d + encoding_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs_1d + encoding_dim, num_actions, actor_hidden_dims, activation)
        print(f'演员 MLP：{self.actor}')

        # 演员观测归一化（仅针对 1D 观测）
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # 评论家 CNN
        if self.critic_obs_groups_2d:
            # 解析评论家 CNN 配置
            assert critic_cnn_cfg is not None, '评论家存在 2D 观测时需提供 CNN 配置。'
            # 若传入单一配置字典，则为每个 2D 观测组复制一份
            if not all(isinstance(v, dict) for v in critic_cnn_cfg.values()):
                critic_cnn_cfg = {group: critic_cnn_cfg for group in self.critic_obs_groups_2d}
            # 确认配置数量与 2D 观测组数量一致
            assert len(critic_cnn_cfg) == len(self.critic_obs_groups_2d), (
                'CNN 配置数量必须与评论家 2D 观测组数量一致。'
            )

            # 为每个评论家 2D 观测组创建 CNN
            self.critic_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.critic_obs_groups_2d):
                self.critic_cnns[obs_group] = CNN(
                    input_dim=critic_in_dims_2d[idx],
                    input_channels=critic_in_channels_2d[idx],
                    **critic_cnn_cfg[obs_group],
                )
                print(f'评论家 CNN（{obs_group}）：{self.critic_cnns[obs_group]}')
                # 累计 CNN 输出维度
                if self.critic_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.critic_cnns[obs_group].output_dim)
                else:
                    raise ValueError('评论家 CNN 输出必须在传入 MLP 前展平。')
        else:
            self.critic_cnns = None
            encoding_dim = 0

        # 评论家 MLP
        self.critic = MLP(num_critic_obs_1d + encoding_dim, 1, critic_hidden_dims, activation)
        print(f'评论家 MLP：{self.critic}')

        # 评论家观测归一化（仅针对 1D 观测）
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # 动作噪声参数
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

        # 动作分布，在 _update_distribution 中创建
        self.distribution = None

        # 禁用分布参数校验以加速
        Normal.set_default_validate_args(False)

    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        """根据 MLP 观测与 CNN 编码更新动作分布。"""
        if self.actor_cnns is not None:
            # 编码 2D 演员观测
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # 与 1D 观测拼接
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        super()._update_distribution(mlp_obs)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """根据观测采样动作。"""
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, cnn_obs)
        return self.distribution.sample()  # type: ignore

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """推理阶段获取确定性动作。"""
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_cnns is not None:
            # 编码 2D 演员观测
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # 与 1D 观测拼接
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        if self.state_dependent_std:
            return self.actor(mlp_obs)[..., 0, :]
        else:
            return self.actor(mlp_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """估计状态价值。"""
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_cnns is not None:
            # 编码 2D 评论家观测
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # 与 1D 观测拼接
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        return self.critic(mlp_obs)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """拆分演员 1D 观测与 2D 观测。"""
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.actor_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """拆分评论家 1D 观测与 2D 观测。"""
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.critic_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def update_normalization(self, obs: TensorDict) -> None:
        """更新 1D 观测归一化统计量。"""
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
