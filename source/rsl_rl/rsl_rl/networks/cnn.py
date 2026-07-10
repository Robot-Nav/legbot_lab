# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""卷积神经网络（CNN）实现，用于编码 2D 图像观测。"""

from __future__ import annotations

import math
import torch
from torch import nn as nn

from rsl_rl.utils import get_param, resolve_nn_activation


class CNN(nn.Sequential):
    """卷积神经网络。

    由卷积层、可选归一化层、可选激活函数与可选池化层顺序组成，
    最终输出可展平为一维向量。
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int] | list[int],
        kernel_size: int | tuple[int] | list[int],
        stride: int | tuple[int] | list[int] = 1,
        dilation: int | tuple[int] | list[int] = 1,
        padding: str = 'none',
        norm: str | tuple[str] | list[str] = 'none',
        activation: str = 'elu',
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = 'none',
        flatten: bool = True,
    ) -> None:
        """初始化 CNN。

        参数:
            input_dim: 输入图像高与宽。
            input_channels: 输入通道数。
            output_channels: 每层卷积的输出通道数列表。
            kernel_size: 每层卷积核大小；可传单个值表示所有层相同。
            stride: 每层卷积步长；可传单个值表示所有层相同。
            dilation: 每层卷积空洞率；可传单个值表示所有层相同。
            padding: 填充类型，可选 'none'、'zeros'、'reflect'、'replicate'、'circular'。
            norm: 每层归一化类型，可选 'none'、'batch'、'layer'；可传单个值表示所有层相同。
            activation: 激活函数。
            max_pool: 每层是否在后接最大池化；可传单个值表示所有层相同。
            global_pool: 最终全局池化类型，可选 'none'、'max'、'avg'。
            flatten: 是否将最终输出展平。
        """
        super().__init__()

        # 解析激活函数
        activation_function = resolve_nn_activation(activation)

        # 顺序构建网络层
        layers = []
        last_channels = input_channels
        last_dim = input_dim
        for idx in range(len(output_channels)):
            # 获取当前层参数
            k = get_param(kernel_size, idx)
            s = get_param(stride, idx)
            d = get_param(dilation, idx)
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ['zeros', 'reflect', 'replicate', 'circular']
                else (0, 0)
            )

            # 添加卷积层
            layers.append(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ['zeros', 'reflect', 'replicate', 'circular'] else 'zeros',
                )
            )

            # 按需添加归一化层
            n = get_param(norm, idx)
            if n == 'none':
                pass
            elif n == 'batch':
                layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == 'layer':
                norm_input_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0], norm_input_dim[1]]))
            else:
                raise ValueError(
                    f'不支持的归一化类型：{n}，仅支持 none、batch、layer。'
                )

            # 添加激活函数
            layers.append(activation_function)

            # 按需添加最大池化
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            # 更新通道数与空间尺寸
            last_channels = output_channels[idx]
            last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))

        # 按需添加全局池化
        if global_pool == 'none':
            pass
        elif global_pool == 'max':
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            last_dim = (1, 1)
        elif global_pool == 'avg':
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            last_dim = (1, 1)
        else:
            raise ValueError(
                f'不支持的全局池化类型：{global_pool}，仅支持 none、max、avg。'
            )

        # 按需展平输出
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        # 记录最终输出维度
        self._output_channels = last_channels if not flatten else None
        self._output_dim = last_dim if not flatten else last_channels * last_dim[0] * last_dim[1]

        # 注册各层
        for idx, layer in enumerate(layers):
            self.add_module(f'{idx}', layer)

    @property
    def output_channels(self) -> int | None:
        """返回输出通道数；若输出已展平则返回 None。"""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """返回输出高宽；若输出已展平则返回总维度。"""
        return self._output_dim

    def init_weights(self) -> None:
        """使用 Kaiming 初始化初始化 CNN 权重。"""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CNN 前向传播。"""
        for layer in self:
            x = layer(x)
        return x


def _compute_padding(input_hw: tuple[int, int], kernel: int, stride: int, dilation: int) -> tuple[int, int]:
    """计算当前层的最优填充。

    参考：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    h = math.ceil((stride * math.floor(input_hw[0] / stride) - input_hw[0] - stride + dilation * (kernel - 1) + 1) / 2)
    w = math.ceil((stride * math.floor(input_hw[1] / stride) - input_hw[1] - stride + dilation * (kernel - 1) + 1) / 2)
    return (h, w)


def _compute_output_dim(
    input_hw: tuple[int, int],
    kernel: int,
    stride: int,
    dilation: int,
    padding: tuple[int, int],
    is_max_pool: bool = False,
) -> tuple[int, int]:
    """计算当前层的输出高宽。

    参考：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    h = math.floor((input_hw[0] + 2 * padding[0] - dilation * (kernel - 1) - 1) / stride + 1)
    w = math.floor((input_hw[1] + 2 * padding[1] - dilation * (kernel - 1) - 1) / stride + 1)

    if is_max_pool:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)

    return (h, w)
