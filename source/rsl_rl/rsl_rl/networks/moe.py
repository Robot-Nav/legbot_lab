"""MoE 相关网络组件：CatELU、归一化层、MLP、Experts 与 MoE。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CatELU(nn.Module):
    """CatELU 激活函数（特征翻倍版本）。

    对输入及其取反分别应用 ELU，并将结果拼接，使特征维度翻倍。

    输出：输入形状为 [..., D] 时，输出为 [..., 2 * D]。

    注意：
        这是结构性激活函数，非逐元素操作；默认最后一维为特征维度。
    """
    def __init__(self, inplace: bool = False):
        """初始化 CatELU。"""
        super().__init__()
        self.elu = nn.ELU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        assert x.dim() >= 2, \
            f'CatELU 要求最后一维为特征维度，当前形状：{x.shape}'

        y1 = self.elu(x)
        y2 = self.elu(-x)
        return torch.cat((y1, y2), dim=-1)


def get_activation(act_name):
    """根据名称获取激活函数实例。"""
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'selu':
        return nn.SELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'crelu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    elif act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'sigmoid':
        return nn.Sigmoid()
    elif act_name == 'cat_elu':
        return CatELU()
    else:
        print('无效的激活函数！')
        return None


class L2Norm(nn.Module):
    """L2 归一化层。"""
    
    def __init__(self):
        """初始化 L2 归一化层。"""
        super().__init__()

    def forward(self, x):
        """沿最后一维执行 L2 归一化。"""
        return F.normalize(x, p=2.0, dim=-1)


class SimNorm(nn.Module):
    """单纯形归一化（Simplicial normalization）。

    改编自 https://arxiv.org/abs/2204.00616。
    """

    def __init__(self):
        """初始化 SimNorm，默认分组维度为 8（适用于潜在维度 512）。"""
        super().__init__()
        self.dim = 8  # 针对潜在维度 512 设计

    def forward(self, x):
        """前向传播：将最后一维分组后做 softmax，再展平。"""
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f'SimNorm(dim={self.dim})'


class MLP(nn.Module):
    """MoE 使用的多层感知机。"""
    def __init__(self, input_dim, output_dim, hidden_dims, activation='elu', last_activation: str | None = None):
        """初始化 MLP。"""
        super().__init__()

        dims = [input_dim] + hidden_dims
        act_func = get_activation(activation)
        layers = []
        last_dim = dims[0]
        for h_dim in dims[1:]:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(act_func)
            # cat_elu 会使特征维度翻倍
            if activation == 'cat_elu':
                last_dim = h_dim * 2
            else:
                last_dim = h_dim
            
        if isinstance(output_dim, int):
            layers.append(nn.Linear(last_dim, output_dim))
        elif isinstance(output_dim, tuple) or isinstance(output_dim, list):
            layers.append(nn.Linear(last_dim, np.prod(output_dim)))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))
        else:
            raise ValueError('output_dim 必须是 int、tuple 或 list。')        
        
        if last_activation is not None:
            last_act_func = get_activation(last_activation)
            layers.append(last_act_func)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。"""
        return self.network(x)


class Experts(nn.Module):
    """专家网络组：共享主干 + 分组 1D 卷积实现多专家并行。"""
    def __init__(self,
                 expert_num,
                 input_dim,
                 backbone_hidden_dims,
                 expert_hidden_dim,
                 output_dim,
                 activation='elu',
    ):
        """初始化专家网络组。"""
        super().__init__()
        self.expert_num = expert_num
        self.output_dim = output_dim

        self.backbone = MLP(input_dim, expert_num * expert_hidden_dim, backbone_hidden_dims, activation, last_activation=activation)
        self.experts = nn.Conv1d(
            in_channels=expert_num*expert_hidden_dim if activation != 'cat_elu' else expert_num*expert_hidden_dim*2,
            out_channels=expert_num*output_dim,
            kernel_size=1,
            groups=expert_num,
        )
    
    def forward(self, x):
        """前向传播，返回各专家输出 (B, expert_num, output_dim)。"""
        shared_features = self.backbone(x).unsqueeze(-1)  # (B, expert_num * expert_hidden_dim, 1)
        expert_outs = self.experts(shared_features).squeeze(-1)  # (B, expert_num * output_dim)
        expert_outs = expert_outs.reshape(-1, self.expert_num, self.output_dim)
        return expert_outs


class MoE(nn.Module):
    """混合专家网络：门控网络加权融合多个专家输出。"""
    def __init__(self,
                 expert_num,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation='elu',
    ):
        """初始化 MoE。"""
        super().__init__()

        # 专家网络
        self.experts = Experts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        
        # 门控网络
        self.gating_network = nn.Sequential(
            MLP(input_dim, expert_num, hidden_dims, activation),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """前向传播，返回加权输出与门控权重。"""
        weights = self.gating_network(x)  # (B, expert_num)
        expert_outs = self.experts(x)  # (B, expert_num, output_dim)
        output = torch.sum(weights.unsqueeze(-1) * expert_outs, dim=1)  # (B, output_dim)
        return output, weights
