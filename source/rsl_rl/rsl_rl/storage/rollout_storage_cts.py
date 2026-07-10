# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CTS rollout 存储，用于收集教师-学生分割的训练数据。"""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories
from functools import partial


class RolloutStorageCTS:
    """Rollout 阶段采集数据的存储容器。

    在 rollout 阶段通过 add_transition 填充，随后根据算法与策略结构返回学习用的生成器。
    """

    class Transition:
        """单个状态转移的数据容器。"""

        def __init__(self) -> None:
            """初始化转移容器。"""
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.privileged_actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)

        def clear(self) -> None:
            """清空转移数据。"""
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        teacher_num_envs: int, 
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = 'cpu',
    ) -> None:
        """初始化 CTS rollout 存储。"""
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs

        # 核心数据
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        
        # 蒸馏相关
        if training_type == 'distillation':
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # 强化学习相关
        if training_type == 'rl':
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # 循环网络相关（暂未实现）
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        # 已存储转移计数
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """将一条转移存入当前 step，并递增计数器。"""
        # 检查存储是否已满
        if self.step >= self.num_transitions_per_env:
            raise OverflowError('Rollout 缓冲区溢出，请在添加新转移前调用 clear()。')

        # 核心数据
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # 蒸馏相关
        if self.training_type == 'distillation':
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # 强化学习相关
        if self.training_type == 'rl':
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # 循环网络相关（暂未实现）
        self._save_hidden_states(transition.hidden_states)

        # 递增计数器
        self.step += 1

    def clear(self) -> None:
        """清空计数器，复用底层张量。"""
        self.step = 0

    def generator(self) -> Generator:
        """蒸馏训练的生成器，返回观测、动作、特权动作与终止标志。"""
        if self.training_type != 'distillation':
            raise ValueError('该生成器仅适用于蒸馏训练。')

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        """前馈网络的强化学习 mini-batch 生成器，按教师-学生比例切片。"""
        if self.training_type != 'rl':
            raise ValueError('该生成器仅适用于强化学习训练。')
        
        # 准备索引
        teacher_samples_num = self.teacher_num_envs * self.num_transitions_per_env
        student_samples_num = self.student_num_envs * self.num_transitions_per_env
        teacher_mini_batch_size = teacher_samples_num // num_mini_batches
        student_mini_batch_size = student_samples_num // num_mini_batches
        teacher_indices = torch.randperm(teacher_samples_num, requires_grad=False, device=self.device)
        student_indices = teacher_samples_num + torch.randperm(student_samples_num, requires_grad=False, device=self.device)
        
        # 核心数据展平为 (num_envs * num_transitions, ...)
        observations = self.observations.transpose(0, 1).flatten(0, 1)
        actions = self.actions.transpose(0, 1).flatten(0, 1)
        values = self.values.transpose(0, 1).flatten(0, 1)
        returns = self.returns.transpose(0, 1).flatten(0, 1)

        # PPO 相关历史数据
        old_actions_log_prob = self.actions_log_prob.transpose(0, 1).flatten(0, 1)
        advantages = self.advantages.transpose(0, 1).flatten(0, 1)
        old_mu = self.mu.transpose(0, 1).flatten(0, 1)
        old_sigma = self.sigma.transpose(0, 1).flatten(0, 1)
        
        def _get_teacher_student_samples(data, slice):
            (i1, i2), (j1, j2) = slice
            return torch.cat([data[teacher_indices[i1:i2]], data[student_indices[j1:j2]]], 0).detach()

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # 选取当前 mini-batch 的索引范围
                slice = (
                    (i * teacher_mini_batch_size, (i+1) * teacher_mini_batch_size),
                    (i * student_mini_batch_size, (i+1) * student_mini_batch_size),
                )
                
                # 构造 mini-batch
                get_batch = partial(_get_teacher_student_samples, slice=slice)
                obs_batch, actions_batch, target_values_batch, returns_batch, \
                old_actions_log_prob_batch, advantages_batch, old_mu_batch, \
                old_sigma_batch = map(get_batch, [
                    observations,
                    actions,
                    values,
                    returns,
                    old_actions_log_prob,
                    advantages,
                    old_mu,
                    old_sigma
                ])

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                # 产出 mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                )

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        """循环网络的 mini-batch 生成器（CTS 暂不支持）。"""
        return NotImplementedError('CTS rollout 存储暂不支持循环网络。')

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        """保存隐藏状态（CTS 暂不支持）。"""
        return NotImplementedError('CTS rollout 存储暂不支持循环网络。')
