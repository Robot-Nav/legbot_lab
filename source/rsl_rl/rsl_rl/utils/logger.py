# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""训练日志记录器，支持 TensorBoard、W&B 与 Neptune。"""

from __future__ import annotations

import git
import os
import pathlib
import statistics
import time
import torch
from collections import deque

import rsl_rl


class Logger:
    """训练指标日志记录器。"""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
    ) -> None:
        """初始化日志记录器。"""
        self.log_dir = log_dir
        self.cfg = cfg
        self.num_envs = num_envs
        self.gpu_world_size = gpu_world_size
        self.device = device
        self.git_status_repos = [rsl_rl.__file__]
        self.tot_timesteps = 0
        self.tot_time = 0

        # 创建缓冲区
        self.ep_extras = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # RND 相关缓冲区
        if self.cfg['algorithm']['rnd_cfg']:
            self.erewbuffer = deque(maxlen=100)
            self.irewbuffer = deque(maxlen=100)
            self.cur_ereward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.cur_ireward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 仅主进程记录日志
        self.disable_logs = is_distributed and gpu_global_rank != 0

        # 初始化日志写入器
        self._prepare_logging_writer()

        # 记录代码状态
        self._store_code_state()

        # 记录配置
        if self.writer and not self.disable_logs and self.logger_type in ['wandb', 'neptune']:
            self.writer.store_config(env_cfg, self.cfg)

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        """处理环境步返回的指标并加入缓冲区。"""
        if self.log_dir is not None:
            if 'episode' in extras:
                self.ep_extras.append(extras['episode'])
            elif 'log' in extras:
                self.ep_extras.append(extras['log'])

            # 更新奖励与回合长度
            if intrinsic_rewards is not None:
                self.cur_ereward_sum += rewards
                self.cur_ireward_sum += intrinsic_rewards
                self.cur_reward_sum += rewards + intrinsic_rewards
            else:
                self.cur_reward_sum += rewards
            self.cur_episode_length += 1

            # 清空已完成回合的数据
            new_ids = (dones > 0).nonzero(as_tuple=False)
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
            if intrinsic_rewards is not None:
                self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_ereward_sum[new_ids] = 0
                self.cur_ireward_sum[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """将训练指标写入日志并打印到控制台。"""
        if self.log_dir is not None and not self.disable_logs:
            collection_size = self.cfg['num_steps_per_env'] * self.num_envs * self.gpu_world_size
            iteration_time = collect_time + learn_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            # 记录回合额外信息
            extras_string = ''
            if self.ep_extras:
                for key in self.ep_extras[0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in self.ep_extras:
                        # 处理缺失、标量与零维张量
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    if '/' in key:
                        self.writer.add_scalar(key, value, it)
                        extras_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar('Episode/' + key, value, it)
                        extras_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

            # 记录损失
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'Loss/{key}', value, it)
            self.writer.add_scalar('Loss/learning_rate', learning_rate, it)

            # 记录动作噪声标准差
            self.writer.add_scalar('Policy/mean_noise_std', action_std.mean().item(), it)

            # 记录性能
            fps = int(collection_size / (collect_time + learn_time))
            self.writer.add_scalar('Perf/total_fps', fps, it)
            self.writer.add_scalar('Perf/collection_time', collect_time, it)
            self.writer.add_scalar('Perf/learning_time', learn_time, it)

            # 记录奖励与回合长度
            if len(self.rewbuffer) > 0:
                if self.cfg['algorithm']['rnd_cfg']:
                    self.writer.add_scalar('Rnd/mean_extrinsic_reward', statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar('Rnd/mean_intrinsic_reward', statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar('Rnd/weight', rnd_weight, it)
                self.writer.add_scalar('Train/mean_reward', statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(self.lenbuffer), it)
                if self.logger_type != 'wandb':
                    self.writer.add_scalar(
                        'Train/mean_reward/time', statistics.mean(self.rewbuffer), int(self.tot_time)
                    )
                    self.writer.add_scalar(
                        'Train/mean_episode_length/time', statistics.mean(self.lenbuffer), int(self.tot_time)
                    )

            # 打印到控制台
            log_string = f"""{'#' * width}\n"""
            log_string += f"""\033[1m{f' Learning iteration {it}/{total_it} '.center(width)}\033[0m \n\n"""

            # 打印运行名
            run_name = self.cfg.get('run_name')
            log_string += f"""{'Run name:':>{pad}} {run_name}\n""" if run_name else ''

            # 打印性能
            log_string += (
                f"""{'Total steps:':>{pad}} {self.tot_timesteps} \n"""
                f"""{'Steps per second:':>{pad}} {fps:.0f} \n"""
                f"""{'Collection time:':>{pad}} {collect_time:.3f}s \n"""
                f"""{'Learning time:':>{pad}} {learn_time:.3f}s \n"""
            )

            # 打印损失
            for key, value in loss_dict.items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""

            # 打印奖励与回合长度
            if len(self.rewbuffer) > 0:
                if self.cfg['algorithm']['rnd_cfg']:
                    log_string += f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                    log_string += f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
                log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            # 打印动作噪声标准差
            log_string += f"""{'Mean action noise std:':>{pad}} {action_std.mean().item():.2f}\n"""

            # 打印回合额外信息
            if not print_minimal:
                log_string += extras_string

            # 打印页脚
            done_it = it + 1 - start_it
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                f"""{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"""
                f"""{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"""
            )
            print(log_string)

            # 清空额外信息缓冲区
            self.ep_extras.clear()

    def save_model(self, path: str, it: int) -> None:
        """保存模型到外部日志服务。"""
        if self.writer and not self.disable_logs and self.logger_type in ['neptune', 'wandb']:
            self.writer.save_model(path, it)

    def _prepare_logging_writer(self) -> None:
        """准备日志写入器，支持 TensorBoard、W&B 与 Neptune。"""
        if self.log_dir is not None and not self.disable_logs:
            self.logger_type = self.cfg.get('logger', 'tensorboard')
            self.logger_type = self.logger_type.lower()

            if self.logger_type == 'neptune':
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == 'wandb':
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == 'tensorboard':
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("日志类型不存在，请选择 'wandb'、'neptune' 或 'tensorboard'。")
        else:
            self.writer = None

    def _store_code_state(self) -> None:
        """保存实验相关代码仓库的 git diff。"""
        if self.log_dir is not None and not self.disable_logs:
            git_log_dir = os.path.join(self.log_dir, 'git')
            os.makedirs(git_log_dir, exist_ok=True)
            file_paths = []
            for repository_file_path in self.git_status_repos:
                try:
                    repo = git.Repo(repository_file_path, search_parent_directories=True)
                    commit = repo.head.commit
                    t = commit.tree
                except Exception:
                    print(f'未在 {repository_file_path} 找到 git 仓库，跳过。')
                    continue
                # 获取仓库名
                repo_name = pathlib.Path(repo.working_dir).name
                diff_file_name = os.path.join(git_log_dir, f'{repo_name}.diff')
                # 写入 diff 文件
                print(f"保存 '{repo_name}' 的 git diff 到：{diff_file_name}")
                with open(diff_file_name, 'w', encoding='utf-8') as f:
                    content = (
                        '--- git commit ---\n'
                        f'commit: {commit.hexsha}\n'
                        f'author: {commit.author.name} <{commit.author.email}>\n'
                        f'date: {commit.committed_datetime.isoformat()}\n'
                        f'message:\n{commit.message.rstrip()}\n\n\n'
                        f'--- git status ---\n{repo.git.status()} \n\n\n'
                        f'--- git diff ---\n{repo.git.diff(t)}'
                    )
                    f.write(content)
                file_paths.append(diff_file_name)

            # 上传 diff 文件到外部日志服务
            if self.writer and self.logger_type in ['wandb', 'neptune'] and file_paths:
                for path in file_paths:
                    self.writer.save_file(path)
