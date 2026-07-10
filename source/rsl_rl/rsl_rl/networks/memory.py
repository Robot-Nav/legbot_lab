# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""循环记忆网络（Memory），用于处理时序观测。"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Union

from rsl_rl.utils import unpad_trajectories

# 由于 Python 版本限制，使用 Union 定义隐藏状态类型别名
HiddenState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]
"""RNN（GRU/LSTM）隐藏状态类型别名。

GRU 为单个张量，LSTM 为两个张量组成的 tuple（隐藏状态与细胞状态）。
"""


class Memory(nn.Module):
    """循环记忆网络。

    用于存储策略的隐藏状态，目前支持 GRU 与 LSTM。
    """

    def __init__(self, input_size: int, hidden_dim: int = 256, num_layers: int = 1, type: str = 'lstm') -> None:
        """初始化循环记忆网络。

        参数:
            input_size: 输入维度。
            hidden_dim: 隐藏层维度。
            num_layers: 循环网络层数。
            type: 循环网络类型，'lstm' 或 'gru'。
        """
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden_state = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """循环记忆前向传播。

        参数:
            input: 输入张量。
            masks: 序列有效掩码；传入时进入 batch 训练模式，否则为推理/蒸馏模式。
            hidden_state: 外部传入的隐藏状态。

        返回:
            循环网络输出。
        """
        batch_mode = masks is not None
        if batch_mode:
            # batch 训练模式需要外部传入隐藏状态
            if hidden_state is None:
                raise ValueError('batch 模式下必须向记忆模块传入隐藏状态')
            out, _ = self.rnn(input, hidden_state)
            out = unpad_trajectories(out, masks)
        else:
            # 推理/蒸馏模式使用上一步保存的隐藏状态
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """重置隐藏状态。

        参数:
            dones: 终止标志；None 时重置全部隐藏状态，否则仅重置已终止环境的隐藏状态。
            hidden_state: 用于替换的隐藏状态。
        """
        if dones is None:  # 重置全部隐藏状态
            if hidden_state is None:
                self.hidden_state = None
            else:
                self.hidden_state = hidden_state
        elif self.hidden_state is not None:  # 仅重置已终止环境的隐藏状态
            if hidden_state is None:
                if isinstance(self.hidden_state, tuple):  # LSTM 为 tuple
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_state[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    '使用自定义隐藏状态重置已终止环境的隐藏状态尚未实现。'
                )

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """分离隐藏状态以截断反向传播梯度。

        参数:
            dones: 终止标志；None 时分离全部隐藏状态，否则仅分离已终止环境的隐藏状态。
        """
        if self.hidden_state is not None:
            if dones is None:  # 分离全部隐藏状态
                if isinstance(self.hidden_state, tuple):  # LSTM 为 tuple
                    self.hidden_state = tuple(hidden_state.detach() for hidden_state in self.hidden_state)
                else:
                    self.hidden_state = self.hidden_state.detach()
            else:  # 仅分离已终止环境的隐藏状态
                if isinstance(self.hidden_state, tuple):  # LSTM 为 tuple
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()
