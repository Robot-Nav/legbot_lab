# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""将 CTS 策略导出为 Torch JIT 脚本模型，用于部署。"""

import copy
import os
import torch


def export_cts_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    single_obs_normalizer: object | None,
    path: str,
    filename: str = 'policy.pt',
) -> None:
    """导出 CTS 策略为 Torch JIT 文件，输入为 single_obs。"""
    policy_exporter = _TorchPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer)
    policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    """CTS 演员-评论家策略导出器。"""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None):
        assert not policy.is_recurrent, 'CTS 策略不支持循环网络。'
        super().__init__()

        # 提取演员或学生网络作为策略
        if hasattr(policy, 'actor'):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, 'student'):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError('策略中未找到 actor 或 student 模块。')
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.state_dependent_std = policy.state_dependent_std
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f'num_actor_obs ({self.num_actor_obs}) 必须能被 num_single_obs ({self.num_single_obs}) 整除。'
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        # single_obs 的各特征维度：投影重力、角速度、命令、上一动作、目标关节位置、目标关节速度
        self.feature_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                '不支持的 single_obs 布局：期望 3+3+3+3*num_actions 等于 num_single_obs。'
            )
        self.register_buffer('obs_history', torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

        # 归一化层
        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def forward(self, single_obs: torch.Tensor):
        """前向推理，维护历史观测并输出动作。"""
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(
                f'期望 single_obs 最后一维为 {self.num_single_obs}，实际为 {single_obs.shape[-1]}。'
            )
        if single_obs.shape[0] != 1:
            raise ValueError('TorchScript CTS 部署当前仅支持 batch size 为 1。')

        # 更新历史观测：每个特征块按 history_len 长度移位
        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        # 归一化并前向传播
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(self.obs_history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        return self.actor(latent_and_obs)

    @torch.jit.export
    def reset(self):
        """重置历史观测缓冲区。"""
        self.obs_history.zero_()

    def export(self, path, filename):
        """导出为 Torch JIT 脚本并保存。"""
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
