# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""随机网络蒸馏（RND）模块，用于提供内在探索奖励。"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, NoReturn

from rsl_rl.env import VecEnv
from rsl_rl.networks import MLP, EmpiricalDiscountedVariationNormalization, EmpiricalNormalization


class RandomNetworkDistillation(nn.Module):
    """随机网络蒸馏（RND）实现。

    通过固定目标网络与可训练预测网络之间的 embedding 差异计算内在奖励，
    鼓励智能体探索未充分访问的状态。

    参考文献：Burda, Yuri, et al. "Exploration by Random Network Distillation." arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        num_states: int,
        obs_groups: dict,
        num_outputs: int,
        predictor_hidden_dims: tuple[int] | list[int],
        target_hidden_dims: tuple[int] | list[int],
        activation: str = 'elu',
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = 'cpu',
        weight_schedule: dict | None = None,
    ) -> None:
        """初始化 RND 模块。

        - 启用 state_normalization 时，使用经验归一化层处理输入状态。
        - 启用 reward_normalization 时，使用经验折扣变化归一化层处理内在奖励。
        - predictor 与 target 隐藏层维度为 -1 时，自动使用 num_states 作为隐藏层维度。

        参数:
            num_states: 预测网络与目标网络的输入状态维度。
            obs_groups: 观测分组字典。
            num_outputs: 预测网络与目标网络的输出 embedding 维度。
            predictor_hidden_dims: 预测网络隐藏层维度列表。
            target_hidden_dims: 目标网络隐藏层维度列表。
            activation: 激活函数。
            weight: 内在奖励缩放系数。
            state_normalization: 是否对输入状态进行归一化。
            reward_normalization: 是否对内在奖励进行归一化。
            device: 计算设备。
            weight_schedule: RND 权重调度配置字典，支持以下模式：
                - 'constant'：恒定权重。
                - 'step'：在 final_step 时跳变为 final_value。
                - 'linear'：在 initial_step 到 final_step 之间线性变化到 final_value。
        """
        # 初始化父类
        super().__init__()

        # 保存参数
        self.num_states = num_states
        self.obs_groups = obs_groups
        self.num_outputs = num_outputs
        self.initial_weight = weight
        self.device = device
        self.state_normalization = state_normalization
        self.reward_normalization = reward_normalization

        # 输入状态归一化
        if state_normalization:
            self.state_normalizer = EmpiricalNormalization(shape=[self.num_states], until=1.0e8).to(self.device)
        else:
            self.state_normalizer = torch.nn.Identity()

        # 内在奖励归一化
        if reward_normalization:
            self.reward_normalizer = EmpiricalDiscountedVariationNormalization(shape=[], until=1.0e8).to(self.device)
        else:
            self.reward_normalizer = torch.nn.Identity()

        # 更新计数器
        self.update_counter = 0

        # 解析权重调度策略
        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            self.weight_scheduler = getattr(self, f'_{weight_schedule["mode"]}_weight_schedule')
        else:
            self.weight_scheduler = None

        # 构建预测网络与目标网络
        self.predictor = MLP(num_states, num_outputs, predictor_hidden_dims, activation).to(self.device)
        self.target = MLP(num_states, num_outputs, target_hidden_dims, activation).to(self.device)

        # 目标网络不参与训练
        self.target.eval()

    def get_intrinsic_reward(self, obs: TensorDict) -> torch.Tensor:
        """根据观测计算内在奖励。"""
        # 计数器按每次学习迭代中的环境步数递增
        self.update_counter += 1
        # 从观测中提取 RND 状态
        rnd_state = self.get_rnd_state(obs)
        rnd_state = self.state_normalizer(rnd_state)
        # 分别通过目标网络与预测网络获取 embedding
        target_embedding = self.target(rnd_state).detach()
        predictor_embedding = self.predictor(rnd_state).detach()
        # 内在奖励为两个 embedding 之间的距离
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # 归一化内在奖励
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)
        # 根据调度策略计算当前权重
        if self.weight_scheduler is not None:
            self.weight = self.weight_scheduler(step=self.update_counter, **self.weight_scheduler_params)
        else:
            self.weight = self.initial_weight
        # 缩放内在奖励
        intrinsic_reward *= self.weight

        return intrinsic_reward

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> NoReturn:
        """未实现：RND 模块通过 get_intrinsic_reward 计算奖励。"""
        raise RuntimeError('RND 模块未实现 forward 方法，请使用 get_intrinsic_reward。')

    def train(self, mode: bool = True) -> RandomNetworkDistillation:
        """设置训练模式（目标网络始终保持评估模式）。"""
        # 仅预测网络参与训练
        self.predictor.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        if self.reward_normalization:
            self.reward_normalizer.train(mode)
        return self

    def eval(self) -> RandomNetworkDistillation:
        """设置评估模式。"""
        return self.train(False)

    def get_rnd_state(self, obs: TensorDict) -> torch.Tensor:
        """拼接 RND 状态观测组。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups['rnd_state']]
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """更新 RND 状态归一化统计量。"""
        # 归一化输入状态
        if self.state_normalization:
            rnd_state = self.get_rnd_state(obs)
            self.state_normalizer.update(rnd_state)

    def _constant_weight_schedule(self, step: int, **kwargs: dict[str, Any]) -> float:
        """恒定权重调度。"""
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs: dict[str, Any]) -> float:
        """阶跃权重调度。"""
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(
        self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs: dict[str, Any]
    ) -> float:
        """线性权重调度。"""
        if step < initial_step:
            return self.initial_weight
        elif step > final_step:
            return final_value
        else:
            return self.initial_weight + (final_value - self.initial_weight) * (step - initial_step) / (
                final_step - initial_step
            )


def resolve_rnd_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]], env: VecEnv) -> dict:
    """解析 RND 配置。

    根据观测分组计算 RND 状态维度，并注入配置；同时按环境时间步长缩放权重。

    参数:
        alg_cfg: 算法配置字典。
        obs: 观测字典。
        obs_groups: 观测分组字典。
        env: 环境对象。

    返回:
        解析后的算法配置字典。
    """
    # 计算 RND 状态维度
    if 'rnd_cfg' in alg_cfg and alg_cfg['rnd_cfg'] is not None:
        num_rnd_state = 0
        for obs_group in obs_groups['rnd_state']:
            assert len(obs[obs_group].shape) == 2, 'RND 模块仅支持 1D 观测。'
            num_rnd_state += obs[obs_group].shape[-1]
        # 将 RND 状态维度与观测分组写入配置
        alg_cfg['rnd_cfg']['num_states'] = num_rnd_state
        alg_cfg['rnd_cfg']['obs_groups'] = obs_groups
        # 按环境时间步长缩放权重
        alg_cfg['rnd_cfg']['weight'] *= env.unwrapped.step_dt
    else:
        alg_cfg['rnd_cfg'] = None
    return alg_cfg
