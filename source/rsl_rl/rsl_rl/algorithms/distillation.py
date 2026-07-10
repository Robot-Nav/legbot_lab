# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""蒸馏算法，用于训练学生网络模仿教师网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class Distillation:
    """蒸馏算法。

    通过行为克隆让学生网络学习教师网络的输出分布，
    常用于将特权教师策略蒸馏到非特权学生策略。
    """

    policy: StudentTeacher | StudentTeacherRecurrent
    """学生-教师模型。"""

    def __init__(
        self,
        policy: StudentTeacher | StudentTeacherRecurrent,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = 'mse',
        optimizer: str = 'adam',
        device: str = 'cpu',
        # 分布式训练参数
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """初始化蒸馏算法。"""
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # 多 GPU 参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg['global_rank']
            self.gpu_world_size = multi_gpu_cfg['world_size']
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # 蒸馏组件
        self.policy = policy
        self.policy.to(self.device)

        # 创建优化器
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # 存储相关
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # 蒸馏参数
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # 初始化损失函数
        loss_fn_dict = {
            'mse': nn.functional.mse_loss,
            'huber': nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f'未知损失类型：{loss_type}，支持：{list(loss_fn_dict.keys())}')

        self.num_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """根据观测采样学生动作，并记录教师动作作为蒸馏目标。"""
        # 计算动作与特权动作
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        # 记录观测
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """处理环境返回的转移数据。"""
        # 更新归一化统计量
        self.policy.update_normalization(obs)
        # 记录奖励与终止标志
        self.transition.rewards = rewards
        self.transition.dones = dones
        # 存入 rollout 存储
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """计算回报（蒸馏无需此步骤）。"""
        pass

    def update(self) -> dict[str, float]:
        """执行一次蒸馏更新，返回行为克隆损失。"""
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                # 学生网络推理（保留梯度）
                actions = self.policy.act_inference(obs)

                # 行为克隆损失
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # 累计损失
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # 每累计 gradient_length 个样本执行一次梯度更新
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # 根据终止标志重置循环状态
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # 构造损失字典
        loss_dict = {'behavior': mean_behavior_loss}

        return loss_dict

    def broadcast_parameters(self) -> None:
        """将模型参数从主 GPU 广播到所有 GPU。"""
        # 获取当前 GPU 上的模型参数
        model_params = [self.policy.state_dict()]
        # 广播参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 所有 GPU 加载主 GPU 的参数
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """收集并平均所有 GPU 的梯度。

        在反向传播后调用，以同步各 GPU 之间的梯度。
        """
        # 收集所有梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # 跨 GPU 求和并平均
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # 将平均后的梯度写回各参数
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
