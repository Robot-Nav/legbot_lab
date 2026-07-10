# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""蒸馏 runner：用于教师-学生方法的训练与评估。"""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable


class DistillationRunner(OnPolicyRunner):
    """教师-学生蒸馏训练的 runner。"""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """执行蒸馏训练。

        参数:
            num_learning_iterations: 学习迭代次数。
            init_at_random_ep_len: 是否以随机回合长度初始化环境。
        """
        # 检查教师模型是否已加载
        if not self.alg.policy.loaded_teacher:
            raise ValueError('未加载教师模型参数，请先加载教师模型再进行蒸馏。')

        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _get_default_obs_sets(self) -> list[str]:
        """获取算法默认需要的观测集合。

        .. note::
            观测集合的处理方式详见 :func:`resolve_obs_groups`。
        """
        return ['teacher']

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """构建蒸馏算法。"""
        # 初始化策略
        student_teacher_class = resolve_callable(self.policy_cfg.pop('class_name'))
        student_teacher: StudentTeacher | StudentTeacherRecurrent = student_teacher_class(
            obs, self.cfg['obs_groups'], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 初始化数据存储
        storage = RolloutStorage(
            'distillation', self.env.num_envs, self.cfg['num_steps_per_env'], obs, [self.env.num_actions], self.device
        )

        # 初始化算法
        alg_class = resolve_callable(self.alg_cfg.pop('class_name'))
        alg: Distillation = alg_class(
            student_teacher, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 蒸馏不使用 RND，将配置置空
        self.cfg['algorithm']['rnd_cfg'] = None

        return alg
