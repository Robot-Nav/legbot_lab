# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Weights & Biases 日志写入器封装。"""

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError('记录到 Weights and Biases 需要安装 wandb。') from None


class WandbSummaryWriter(SummaryWriter):
    """Weights & Biases 摘要写入器。"""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        """初始化 W&B 写入器。

        参数:
            log_dir: 日志目录路径。
            flush_secs: TensorBoard 刷新间隔（秒）。
            cfg: 训练配置字典，需包含 'wandb_project'。
        """
        super().__init__(log_dir, flush_secs)

        # 获取运行名
        run_name = os.path.split(log_dir)[-1]

        # 获取 W&B 项目与实体
        try:
            project = cfg['wandb_project']
        except KeyError:
            raise KeyError("请在 runner 配置中指定 wandb_project，例如 'legged_gym'。") from None
        try:
            entity = os.environ['WANDB_USERNAME']
        except KeyError:
            entity = None

        # 初始化 W&B
        wandb.init(project=project, entity=entity, name=run_name)
        wandb.config.update({'log_dir': log_dir})

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """保存训练与环境配置到 W&B。"""
        wandb.config.update({'runner_cfg': train_cfg})
        wandb.config.update({'policy_cfg': train_cfg['policy']})
        wandb.config.update({'alg_cfg': train_cfg['algorithm']})
        try:
            wandb.config.update({'env_cfg': env_cfg.to_dict()})
        except Exception:
            wandb.config.update({'env_cfg': asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """记录标量到 TensorBoard 与 W&B。"""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        """结束 W&B 运行。"""
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """上传模型文件到 W&B。"""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """上传任意文件到 W&B。"""
        wandb.save(path, base_path=os.path.dirname(path))
