"""Unitree 电机执行器配置。

实现电机扭矩-转速特性曲线与摩擦模型。
来源：https://github.com/unitreerobotics/unitree_rl_lab
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class UnitreeActuator(DelayedPDActuator):
    """Unitree 电机执行器，按扭矩-转速曲线限制输出扭矩。

    扭矩-转速曲线示意::

            扭矩限制 (N·m)
                ^
        Y2──────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
        ────────┼──────────────┼──────> 转速 (rad/s)
                                  X1   X2

    - Y1：峰值扭矩（扭矩与转速同向）
    - Y2：峰值扭矩（扭矩与转速反向）
    - X1：满扭矩下的最大转速（曲线拐点）
    - X2：空载转速
    - Fs：静摩擦系数
    - Fd：动摩擦系数
    - Va：摩擦完全激活时的转速
    """

    cfg: UnitreeActuatorCfg

    armature: torch.Tensor
    """执行器关节的电惯量，形状为 ``(num_envs, num_joints)``。

    ``armature = J2 + J1 * i2^2 + Jr * (i1 * i2)^2``
    """

    def __init__(self, cfg: UnitreeActuatorCfg, *args, **kwargs):
        """初始化电机参数。"""
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """计算并返回经扭矩-转速曲线与摩擦模型处理后的执行器动作。

        参数:
            control_action: 控制器输出的目标动作。
            joint_pos: 当前关节位置。
            joint_vel: 当前关节转速。

        返回:
            处理后的执行器动作。
        """
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)

        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel)
            + self._friction_dynamic * joint_vel
        )

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """按扭矩-转速曲线裁剪输出扭矩。

        参数:
            effort: 待裁剪的扭矩。

        返回:
            裁剪后的扭矩。
        """
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        """计算转速超过拐点后的线性下降扭矩限制。"""
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    """Unitree 电机执行器配置。"""

    class_type: type = UnitreeActuator

    X1: float = 1e9
    """满扭矩最大转速（扭矩-转速曲线拐点），单位：rad/s。"""

    X2: float = 1e9
    """空载转速，单位：rad/s。"""

    Y1: float = MISSING
    """峰值扭矩（扭矩与转速同向），单位：N·m。"""

    Y2: float | None = None
    """峰值扭矩（扭矩与转速反向），单位：N·m。"""

    Fs: float = 0.0
    """静摩擦系数。"""

    Fd: float = 0.0
    """动摩擦系数。"""

    Va: float = 0.01
    """摩擦完全激活时的转速。"""


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    """Go2 高电压电机的默认参数配置。"""

    X1 = 13.5
    X2 = 30
    Y1 = 20.2
    Y2 = 23.4
