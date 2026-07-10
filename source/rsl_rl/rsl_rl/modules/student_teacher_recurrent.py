# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""带循环记忆的学生-教师蒸馏网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, Memory


class StudentTeacherRecurrent(nn.Module):
    """带循环记忆的学生-教师模型。

    学生网络使用循环记忆处理普通观测；教师网络可选择是否使用循环记忆处理特权观测。
    训练时教师固定为评估模式，仅学生网络参数参与更新。
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = 'elu',
        init_noise_std: float = 0.1,
        noise_std_type: str = 'scalar',
        rnn_type: str = 'lstm',
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        teacher_recurrent: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """初始化循环学生-教师模型。

        参数:
            obs: 观测样例，用于推断各观测组维度。
            obs_groups: 观测分组字典，需包含 'policy' 与 'teacher'。
            num_actions: 动作维度。
            student_obs_normalization: 是否对学生观测进行经验归一化。
            teacher_obs_normalization: 是否对教师观测进行经验归一化。
            student_hidden_dims: 学生 MLP 隐藏层维度。
            teacher_hidden_dims: 教师 MLP 隐藏层维度。
            activation: 激活函数类型。
            init_noise_std: 初始动作标准差。
            noise_std_type: 标准差参数化方式，'scalar' 或 'log'。
            rnn_type: 循环网络类型，'lstm' 或 'gru'。
            rnn_hidden_dim: 循环网络隐藏层维度。
            rnn_num_layers: 循环网络层数。
            teacher_recurrent: 教师是否也使用循环网络。
            kwargs: 额外参数，仅打印警告后忽略。
        """
        if 'rnn_hidden_size' in kwargs:
            warnings.warn(
                '参数 `rnn_hidden_size` 已弃用，将在未来版本移除，请使用 `rnn_hidden_dim`。',
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # 仅在新参数为默认值时覆盖
                rnn_hidden_dim = kwargs.pop('rnn_hidden_size')
        if kwargs:
            print(
                'StudentTeacherRecurrent.__init__ 收到未预期参数，将被忽略：'
                + str(kwargs.keys()),
            )
        super().__init__()

        self.loaded_teacher = False  # 标记教师参数是否已加载
        self.teacher_recurrent = teacher_recurrent  # 标记教师是否使用循环网络

        # 获取学生与教师观测维度
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups['policy']:
            assert len(obs[obs_group].shape) == 2, 'StudentTeacherRecurrent 模块仅支持 1D 观测。'
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups['teacher']:
            assert len(obs[obs_group].shape) == 2, 'StudentTeacherRecurrent 模块仅支持 1D 观测。'
            num_teacher_obs += obs[obs_group].shape[-1]

        # 学生网络：循环记忆 + MLP
        self.memory_s = Memory(num_student_obs, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.student = MLP(rnn_hidden_dim, num_actions, student_hidden_dims, activation)
        print(f'学生 RNN：{self.memory_s}')
        print(f'学生 MLP：{self.student}')

        # 学生观测归一化
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # 教师网络
        if self.teacher_recurrent:
            self.memory_t = Memory(num_teacher_obs, rnn_hidden_dim, rnn_num_layers, rnn_type)
        teacher_input_dim = rnn_hidden_dim if self.teacher_recurrent else num_teacher_obs
        self.teacher = MLP(teacher_input_dim, num_actions, teacher_hidden_dims, activation)
        if self.teacher_recurrent:
            print(f'教师 RNN：{self.memory_t}')
        print(f'教师 MLP：{self.teacher}')

        # 教师观测归一化
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # 动作噪声参数
        self.noise_std_type = noise_std_type
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

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        """根据终止标志重置学生与教师的循环状态。"""
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])

    def forward(self) -> NoReturn:
        """未实现：循环学生-教师模型通过专用方法前向传播。"""
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        """动作分布均值。"""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """动作分布标准差。"""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """动作分布熵。"""
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """根据学生循环记忆输出更新动作分布。"""
        # 计算均值
        mean = self.student(obs)
        # 计算标准差
        if self.noise_std_type == 'scalar':
            std = self.std.expand_as(mean)
        elif self.noise_std_type == 'log':
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f'未知标准差类型：{self.noise_std_type}，应为 scalar 或 log。')
        # 创建分布
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """根据学生观测采样动作。"""
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        out_mem = self.memory_s(obs).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """推理阶段获取学生网络的确定性动作。"""
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        out_mem = self.memory_s(obs).squeeze(0)
        return self.student(out_mem)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """使用教师网络估计特权动作（蒸馏目标）。"""
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            if self.teacher_recurrent:
                self.memory_t.eval()
                obs = self.memory_t(obs).squeeze(0)
            return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """拼接学生观测组。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups['policy']]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        """拼接教师观测组。"""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups['teacher']]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        """返回学生与教师的当前隐藏状态。"""
        if self.teacher_recurrent:
            return self.memory_s.hidden_state, self.memory_t.hidden_state
        else:
            return self.memory_s.hidden_state, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        """分离循环隐藏状态以截断梯度。"""
        self.memory_s.detach_hidden_state(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_state(dones)

    def train(self, mode: bool = True) -> None:
        """设置训练模式，并强制教师网络保持评估模式。"""
        super().train(mode)
        # 教师网络不参与训练
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        """更新学生观测归一化统计量。"""
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载学生与教师网络参数。

        参数:
            state_dict: 模型状态字典。
            strict: 是否严格匹配状态字典键。

        返回:
            是否恢复之前的训练。该标志供 OnPolicyRunner 的 load 函数使用，
            以决定如何加载其他参数（例如蒸馏相关参数）。
        """
        # 判断状态字典来源：强化学习训练或蒸馏训练
        if any('actor' in key for key in state_dict):  # 来自强化学习训练的教师参数
            # 将 actor 键重命名为教师键，并剔除评论家参数
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if 'actor.' in key:
                    teacher_state_dict[key.replace('actor.', '')] = value
                if 'actor_obs_normalizer.' in key:
                    teacher_obs_normalizer_state_dict[key.replace('actor_obs_normalizer.', '')] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            # 若教师为循环网络，同时加载其循环记忆参数
            if self.teacher_recurrent:
                memory_t_state_dict = {}
                for key, value in state_dict.items():
                    if 'memory_a.' in key:
                        memory_t_state_dict[key.replace('memory_a.', '')] = value
                self.memory_t.load_state_dict(memory_t_state_dict, strict=strict)
            # 设置教师加载成功标志
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # 并非恢复蒸馏训练
        elif any('student' in key for key in state_dict):  # 来自蒸馏训练
            super().load_state_dict(state_dict, strict=strict)
            # 设置教师加载成功标志
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # 恢复蒸馏训练
        else:
            raise ValueError('state_dict 中未包含学生或教师参数。')
