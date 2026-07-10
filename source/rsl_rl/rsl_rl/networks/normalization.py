# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.

"""经验归一化层，用于在线估计并归一化输入分布。"""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """基于经验均值与方差的归一化层。"""

    def __init__(self, shape: int | tuple[int] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """初始化经验归一化模块。

        注意：归一化参数在整个 batch 上计算，而非按环境单独计算。

        参数:
            shape: 输入形状（不包含 batch 维度）。
            eps: 防止除零的小常数。
            until: 累计 batch 大小超过该值后停止更新统计量。
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer('_mean', torch.zeros(shape).unsqueeze(0))
        self.register_buffer('_var', torch.ones(shape).unsqueeze(0))
        self.register_buffer('_std', torch.ones(shape).unsqueeze(0))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        """返回当前经验均值。"""
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        """返回当前经验标准差。"""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用经验均值与标准差对输入进行归一化。"""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """根据新样本更新经验统计量（不输出归一化结果）。"""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """根据经验统计量对归一化后的值进行反归一化。"""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Pathak 等人大规模 PPO 研究中使用的奖励归一化方法。

    由于奖励函数通常是非平稳的，对其量级进行归一化有助于价值函数快速学习。
    该方法通过折扣奖励累积值的运行标准差来缩放奖励。
    """

    def __init__(
        self, shape: int | tuple[int] | list[int], eps: float = 1e-2, gamma: float = 0.99, until: int | None = None
    ) -> None:
        """初始化经验折扣变化归一化模块。"""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """归一化奖励。"""
        if self.training:
            # 更新折扣奖励平均
            avg = self.disc_avg.update(rew)
            # 使用折扣奖励更新经验矩
            self.emp_norm.update(avg)

        # 使用经验标准差归一化奖励
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class _DiscountedAverage:
    r"""奖励的折扣平均。

    折扣平均定义为：

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """初始化折扣平均器。"""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """更新折扣平均并返回当前值。"""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
