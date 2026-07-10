"""带限制范围的速度指令配置，用于课程学习中逐步扩展指令范围。"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """在 UniformVelocityCommandCfg 基础上增加 limit_ranges，用于课程学习。"""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING  # 指令范围上限，课程调度不可超出该限制
