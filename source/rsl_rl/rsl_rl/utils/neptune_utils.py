# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neptune 日志写入器封装。"""

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import neptune
except ModuleNotFoundError:
    raise ModuleNotFoundError('记录到 Neptune 需要安装 neptune-client。') from None


class NeptuneSummaryWriter(SummaryWriter):
    """Neptune 摘要写入器。"""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        """初始化 Neptune 写入器。

        参数:
            log_dir: 日志目录路径。
            flush_secs: TensorBoard 刷新间隔（秒）。
            cfg: 训练配置字典，需包含 'neptune_project'。
        """
        super().__init__(log_dir, flush_secs)

        # 获取运行名
        run_name = os.path.split(log_dir)[-1]

        # 获取 Neptune 项目与实体
        try:
            project = cfg['neptune_project']
        except KeyError:
            raise KeyError("请在 runner 配置中指定 neptune_project，例如 'legged_gym'。") from None
        try:
            token = os.environ['NEPTUNE_API_TOKEN']
        except KeyError:
            raise KeyError(
                '未找到 Neptune API Token。请运行或添加到 ~/.bashrc：export NEPTUNE_API_TOKEN=YOUR_API_TOKEN'
            ) from None
        try:
            entity = os.environ['NEPTUNE_USERNAME']
        except KeyError:
            raise KeyError(
                '未找到 Neptune 用户名。请运行或添加到 ~/.bashrc：export NEPTUNE_USERNAME=YOUR_USERNAME'
            ) from None

        # 初始化 Neptune
        neptune_project = entity + '/' + project
        self.run = neptune.init_run(project=neptune_project, api_token=token)
        self.run['log_dir'].log(run_name)

        # 特殊字符名称映射（Neptune 不支持部分字符）
        self.name_map = {
            'Train/mean_reward/time': 'Train/mean_reward_time',
            'Train/mean_episode_length/time': 'Train/mean_episode_length_time',
        }

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """保存训练与环境配置到 Neptune。"""
        self.run['runner_cfg'] = train_cfg
        self.run['policy_cfg'] = train_cfg['policy']
        self.run['alg_cfg'] = train_cfg['algorithm']
        try:
            self.run['env_cfg'] = env_cfg.to_dict()
        except Exception:
            self.run['env_cfg'] = asdict(env_cfg)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """记录标量到 TensorBoard 与 Neptune。"""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.run[self._map_path(tag)].log(scalar_value, step=global_step)

    def stop(self) -> None:
        """停止 Neptune 运行。"""
        self.run.stop()

    def save_model(self, model_path: str, it: int) -> None:
        """上传模型文件到 Neptune。"""
        self.run['model/saved_model_' + str(it)].upload(model_path)

    def save_file(self, path: str) -> None:
        """上传任意文件到 Neptune。"""
        name = path.rsplit('/', 1)[-1].split('.')[0]
        self.run['git_diff/' + name].upload(path)

    def _map_path(self, path: str) -> str:
        """将不兼容 Neptune 的指标路径映射为合法路径。"""
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
