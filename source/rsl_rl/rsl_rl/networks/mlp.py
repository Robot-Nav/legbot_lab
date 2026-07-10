# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""多层感知机（MLP）网络实现。"""

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """多层感知机。

    由线性层与激活函数交替堆叠而成；除非指定最后一层激活，否则最后一层为线性层。

    额外便利：
    - 隐藏层维度为 -1 时，自动推断为输入维度。
    - 输出维度为 tuple 时，输出会被 reshape 为对应形状。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = 'elu',
        last_activation: str | None = None,
    ) -> None:
        """初始化 MLP。

        参数:
            input_dim: 输入维度。
            output_dim: 输出维度。
            hidden_dims: 隐藏层维度列表；-1 表示自动推断为输入维度。
            activation: 中间层激活函数。
            last_activation: 最后一层激活函数；None 表示最后一层为线性层。
        """
        super().__init__()

        # 解析激活函数
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # 将 -1 替换为输入维度
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # 顺序构建网络层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # 添加输出层
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # 计算总输出维度，并通过 Unflatten  reshape 为指定形状
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # 若指定最后一层激活，则追加
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # 注册各层
        for idx, layer in enumerate(layers):
            self.add_module(f'{idx}', layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """使用正交初始化初始化 MLP 权重。

        参数:
            scales: 权重缩放因子。
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP 前向传播。"""
        for layer in self:
            x = layer(x)
        return x
