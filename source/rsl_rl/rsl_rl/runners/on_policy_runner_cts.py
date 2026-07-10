# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于 actor-critic 方法（含 CTS 变体）的 on-policy 训练与评估 runner。"""

from __future__ import annotations

import os
import time
import torch
import warnings
import yaml
import numpy as np
from tensordict import TensorDict

from rsl_rl.algorithms import MoECTS
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCriticMoECTS,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.storage import RolloutStorageCTS
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger_cts import LoggerCTS
from rsl_rl.utils.exporter_cts import export_cts_policy_as_jit


def numpy_representer(dumper: yaml.SafeDumper, data: np.floating) -> yaml.Node:
    """将 numpy 浮点数序列化为 YAML 浮点数。"""
    return dumper.represent_float(float(data))


def numpy_int_representer(dumper: yaml.SafeDumper, data: np.integer) -> yaml.Node:
    """将 numpy 整数序列化为 YAML 整数。"""
    return dumper.represent_int(int(data))


yaml.add_representer(np.float32, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int32, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int64, numpy_int_representer, Dumper=yaml.SafeDumper)


class OnPolicyRunnerCTS:
    """用于 actor-critic 方法（含 CTS 变体）的 on-policy 训练与评估 runner。"""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = 'cpu') -> None:
        """初始化 CTS runner。

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
        obs = self._get_obs_dict()
        self.cfg['obs_groups'] = resolve_obs_groups(obs, self.cfg['obs_groups'], self._get_default_obs_sets())

        # 创建算法
        self.alg = self._construct_algorithm(obs)

        # 创建日志记录器
        self.logger = LoggerCTS(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            teacher_env_idxs=self.alg.teacher_env_idxs,
            device=self.device,
        )

        self.current_learning_iteration = 0

        # RoboGauge 客户端
        try:
            robogauge_cfg = train_cfg.get('robogauge', {})
            if not robogauge_cfg.get('enabled', False):
                raise ImportError('config disabled')
            from robogauge.scripts.client import RoboGaugeClient

            robogauge_port = robogauge_cfg.get('port', 9973)
            self.robogauge_client = RoboGaugeClient(f'http://127.0.0.1:{robogauge_port}')
            self.robogauge_client.wait_until_available()
        except Exception as e:
            print(f'[INFO] RoboGauge client could not be initialized: {e}, disabling RoboGauge interface.')
            self.robogauge_client = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """执行 CTS 训练。

        Args:
            num_learning_iterations: 需要训练的总迭代次数。
            init_at_random_ep_len: 是否随机初始化 episode 长度以增加探索。
        """
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self._get_obs_dict().to(self.device)
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
                    _, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # 新版 IsaacLab API 的完整观测存放在 extras['observations']
                    obs = extras['observations']
                    if not isinstance(obs, TensorDict):
                        batch_size = [next(iter(obs.values())).shape[0]]
                        obs = TensorDict(obs, batch_size=batch_size, device=self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
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
                self.save(os.path.join(self.logger.log_dir, f'model_{it}.pt'), it=it, last_model=False)  # type: ignore

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(
                os.path.join(self.logger.log_dir, f'model_{self.current_learning_iteration}.pt'),
                it=self.current_learning_iteration,
                last_model=True,
            )

    def save(self, path: str, it: int, last_model: bool, infos: dict | None = None) -> None:
        """保存模型与优化器状态。

        Args:
            path: 保存路径。
            it: 当前迭代步数。
            last_model: 是否为最终模型。
            infos: 附加信息。
        """
        saved_dict = {
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'optimizer_stu_enc_state_dict': self.alg.optimizer_stu_enc.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.alg_cfg['rnd_cfg']:
            saved_dict['rnd_state_dict'] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict['rnd_optimizer_state_dict'] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        self.logger.save_model(path, self.current_learning_iteration)
        self.update_robogauge(it, last_model)

    def update_robogauge(self, it: int, last_model: bool) -> None:
        """提交当前模型到 RoboGauge 并拉取评估结果。

        Args:
            it: 当前迭代步数。
            last_model: 是否为最终模型，最终模型会持续等待结果。
        """
        if self.robogauge_client is None or self.logger.log_dir is None or self.logger.disable_logs:
            return

        try:
            if it % 500 == 0 or last_model:
                # 导出 JIT 模型
                jit_dir = os.path.join(self.logger.log_dir, 'jit_models')
                jit_path = os.path.join(jit_dir, f'policy_jit_{it}.pt')
                export_cts_policy_as_jit(
                    self.alg.policy,
                    actor_obs_normalizer=self.alg.policy.actor_obs_normalizer,
                    single_obs_normalizer=self.alg.policy.single_obs_normalizer,
                    path=jit_dir,
                    filename=f'policy_jit_{it}.pt',
                )
                # 提交到 RoboGauge
                self.robogauge_client.submit_task(
                    model_path=jit_path,
                    step=it,
                    task_name='go2_lab',
                    experiment_name=self.cfg['experiment_name'],
                )
        except Exception as e:
            print(f'[WARN] RoboGauge submit failed at step {it}: {e}')
            return

        check_times = 1
        if last_model:
            check_times = int(1e9)  # 持续检查直到手动停止

        while check_times > 0:
            check_times -= 1
            try:
                self.robogauge_client.monitor_tasks()
            except Exception as e:
                print(f'[WARN] RoboGauge monitor failed at step {it}: {e}')
                break

            results_dir = os.path.join(self.logger.log_dir, 'robogauge_results')
            os.makedirs(results_dir, exist_ok=True)
            result_received = False

            for task_id, resp in self.robogauge_client.response_data.items():
                if not isinstance(resp, dict):
                    print(f'[WARN] RoboGauge returned an invalid response for task {task_id}: {resp}')
                    continue
                results = resp.get('results')
                step = resp.get('step', it)
                if results is None:
                    print(f'[WARN] RoboGauge returned empty results for task {task_id} at step {step}.')
                    continue
                scores = results.get('scores')
                if scores is None:
                    print(f'[WARN] RoboGauge results for task {task_id} at step {step} do not contain \'scores\'.')
                    continue
                if step == it:
                    result_received = True
                if self.logger.writer is not None:
                    for key, val in scores.items():
                        self.logger.writer.add_scalar(f'RoboGauge/{key}', val, step)
                results_path = os.path.join(results_dir, f'results_{step}.yaml')
                with open(results_path, 'w', encoding='utf-8') as f:
                    yaml.dump(results, f, allow_unicode=True, sort_keys=False)

            if last_model and result_received:
                print(f'RoboGauge result for step {it} received. Exiting wait loop.')
                break

            if check_times > 0:
                print('Sleeping for 1 minute before checking RoboGauge results again...')
                time.sleep(60)  # 等待 1 分钟后再次检查

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
            self.alg.optimizer_stu_enc.load_state_dict(loaded_dict['optimizer_stu_enc_state_dict'])
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

    def _get_obs_dict(self):
        """从环境获取完整观测字典。

        兼容旧版 IsaacLab API（get_observations() 返回 TensorDict）
        与新版 API（返回元组 (policy_obs, extras)）。
        若观测为普通字典则转换为 TensorDict。

        Returns:
            环境观测的 TensorDict。
        """
        obs_result = self.env.get_observations()
        if isinstance(obs_result, tuple):
            # 新版 IsaacLab API: (policy_obs, {'observations': full_obs_dict})
            obs = obs_result[1]['observations']
        else:
            obs = obs_result
        if not isinstance(obs, TensorDict):
            # 从第一个观测张量推断 batch size
            batch_size = [next(iter(obs.values())).shape[0]]
            obs = TensorDict(obs, batch_size=batch_size, device=self.device)
        return obs

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

    def _construct_algorithm(self, obs: TensorDict) -> MoECTS:
        """构建 actor-critic 算法。

        Args:
            obs: 环境观测。

        Returns:
            配置好的 MoECTS 算法实例。
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

        # 临时使用 eval 以规避导入问题
        actor_critic_class = eval(self.policy_cfg.pop('class_name'))
        actor_critic: ActorCriticMoECTS = actor_critic_class(
            obs, self.cfg['obs_groups'], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        storage = RolloutStorageCTS(
            'rl', self.env.num_envs, max(int(self.env.num_envs * self.alg_cfg['teacher_env_ratio']), 1), self.cfg['num_steps_per_env'], obs, [self.env.num_actions], self.device
        )

        # 临时使用 eval 以规避导入问题
        alg_class = eval(self.alg_cfg.pop('class_name'))
        alg: MoECTS = alg_class(
            actor_critic, storage, self.env.num_envs, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
