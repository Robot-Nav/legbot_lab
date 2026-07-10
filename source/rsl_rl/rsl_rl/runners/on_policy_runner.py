# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于 actor-critic 方法的 on-policy 训练与评估 runner。"""

from __future__ import annotations

import os
import time
import torch
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger


class OnPolicyRunner:
    """用于 actor-critic 方法的 on-policy 训练与评估 runner。"""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = 'cpu') -> None:
        """初始化 runner。

        Args:
            env: 向量化环境。
            train_cfg: 训练配置字典。
            log_dir: 日志目录。
            device: 训练设备。
        """
        self.cfg = train_cfg
        self.policy_cfg = train_cfg['policy']
        self.alg_cfg = train_cfg['algorithm']
        self.device = device
        self.env = env

        # 若启用则配置多 GPU 训练
        self._configure_multi_gpu()

        # 从环境获取观测以构建算法
        obs = self.env.get_observations()
        self.cfg['obs_groups'] = resolve_obs_groups(obs, self.cfg['obs_groups'], self._get_default_obs_sets())

        # 创建算法
        self.alg = self._construct_algorithm(obs)

        # 创建日志记录器
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """执行 PPO 训练。

        Args:
            num_learning_iterations: 需要训练的总迭代次数。
            init_at_random_ep_len: 是否随机初始化 episode 长度以增加探索。
        """
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        # 切换到训练模式（例如启用 dropout）
        self.train_mode()

        # 分布式训练下同步各进程参数
        if self.is_distributed:
            print(f'Synchronizing parameters for rank {self.gpu_global_rank}...')
            self.alg.broadcast_parameters()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # 采集 rollout
            with torch.inference_mode():
                for _ in range(self.cfg['num_steps_per_env']):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # 内禀奖励仅用于日志记录
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg['rnd_cfg'] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg['rnd_cfg'] else None,
            )

            if it % self.cfg['save_interval'] == 0:
                self.save(os.path.join(self.logger.log_dir, f'model_{it}.pt'))  # type: ignore

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def save(self, path: str, infos: dict | None = None) -> None:
        """保存模型与优化器状态。

        Args:
            path: 保存路径。
            infos: 附加信息。
        """
        saved_dict = {
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.alg_cfg['rnd_cfg']:
            saved_dict['rnd_state_dict'] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict['rnd_optimizer_state_dict'] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """加载模型与优化器状态。

        Args:
            path: 模型文件路径。
            load_optimizer: 是否加载优化器状态。
            map_location: 加载时的设备映射。

        Returns:
            加载的附加信息。
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict['model_state_dict'])
        if self.alg_cfg['rnd_cfg']:
            self.alg.rnd.load_state_dict(loaded_dict['rnd_state_dict'])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if self.alg_cfg['rnd_cfg']:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict['rnd_optimizer_state_dict'])
        if resumed_training:
            self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device: str | None = None) -> callable:
        """获取用于推理的策略。

        Args:
            device: 推理设备。

        Returns:
            策略推理函数。
        """
        # 切换到评估模式（例如禁用 dropout）
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        """将策略与 RND 切换到训练模式。"""
        self.alg.policy.train()
        if self.alg_cfg['rnd_cfg']:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        """将策略与 RND 切换到评估模式。"""
        self.alg.policy.eval()
        if self.alg_cfg['rnd_cfg']:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """将 git 仓库路径加入日志记录。

        Args:
            repo_file_path: 仓库文件路径。
        """
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        """获取算法默认需要的观测集合。

        .. note::
            关于观测集合的处理细节，参见 :func:`resolve_obs_groups`。
        """
        default_sets = ['critic']
        if 'rnd_cfg' in self.alg_cfg and self.alg_cfg['rnd_cfg'] is not None:
            default_sets.append('rnd_state')
        return default_sets

    def _configure_multi_gpu(self) -> None:
        """配置多 GPU 分布式训练。"""
        self.gpu_world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv('LOCAL_RANK', '0'))
        self.gpu_global_rank = int(os.getenv('RANK', '0'))

        self.multi_gpu_cfg = {
            'global_rank': self.gpu_global_rank,  # 主进程 rank
            'local_rank': self.gpu_local_rank,  # 当前进程 rank
            'world_size': self.gpu_world_size,  # 进程总数
        }

        if self.device != f'cuda:{self.gpu_local_rank}':
            raise ValueError(
                f'Device \'{self.device}\' does not match expected device for local rank \'{self.gpu_local_rank}\'.'
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f'Local rank \'{self.gpu_local_rank}\' is greater than or equal to world size \'{self.gpu_world_size}\'.'
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f'Global rank \'{self.gpu_global_rank}\' is greater than or equal to world size \'{self.gpu_world_size}\'.'
            )

        torch.distributed.init_process_group(backend='nccl', rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """构建 actor-critic 算法。

        Args:
            obs: 环境观测。

        Returns:
            配置好的 PPO 算法实例。
        """
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg['obs_groups'], self.env)

        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get('empirical_normalization') is not None:
            warnings.warn(
                'The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and '
                '`critic_obs_normalization` as part of the `policy` configuration instead.',
                DeprecationWarning,
            )
            if self.policy_cfg.get('actor_obs_normalization') is None:
                self.policy_cfg['actor_obs_normalization'] = self.cfg['empirical_normalization']
            if self.policy_cfg.get('critic_obs_normalization') is None:
                self.policy_cfg['critic_obs_normalization'] = self.cfg['empirical_normalization']

        actor_critic_class = resolve_callable(self.policy_cfg.pop('class_name'))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN = actor_critic_class(
            obs, self.cfg['obs_groups'], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        storage = RolloutStorage(
            'rl', self.env.num_envs, self.cfg['num_steps_per_env'], obs, [self.env.num_actions], self.device
        )

        alg_class = resolve_callable(self.alg_cfg.pop('class_name'))
        alg: PPO = alg_class(
            actor_critic, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
