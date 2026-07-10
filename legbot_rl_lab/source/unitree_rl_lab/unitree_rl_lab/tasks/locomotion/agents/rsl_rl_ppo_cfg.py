# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO 训练器配置。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """足式运动任务的基础 PPO Runner 配置。"""

    num_steps_per_env = 24              # 每个环境每次迭代的步数
    max_iterations = 100000             # 最大训练迭代次数
    save_interval = 100                 # 模型保存间隔（迭代数）
    experiment_name = ""                # 实验名，默认使用任务名
    empirical_normalization = False     # 是否使用经验归一化
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}  # 观测组映射
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,             # 策略初始噪声标准差
        actor_hidden_dims=[512, 256, 128],   # Actor 隐藏层维度
        critic_hidden_dims=[512, 256, 128],  # Critic 隐藏层维度
        activation="elu",               # 激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,            # 价值损失系数
        use_clipped_value_loss=True,    # 是否裁剪价值损失
        clip_param=0.2,                 # PPO 裁剪参数
        entropy_coef=0.01,              # 熵奖励系数
        num_learning_epochs=5,          # 每次采样的学习轮数
        num_mini_batches=4,             # 每次学习迭代的 mini-batch 数
        learning_rate=1.0e-3,           # 学习率
        schedule="adaptive",            # 学习率调度策略
        gamma=0.99,                     # 折扣因子
        lam=0.95,                       # GAE lambda
        desired_kl=0.01,                # 目标 KL 散度，用于自适应学习率
        max_grad_norm=1.0,              # 梯度裁剪范数
    )
