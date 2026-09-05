# Copyright (c) 2026 Robot-Nav
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Legbot wheel-legged quadruped (16-DOF).

Legbot 是四轮足机器人：每条腿 3 个驱动关节（hip/thigh/calf），足端为主动轮
（foot 关节）。总计 16 个自由度（12 腿 + 4 轮）。
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

# 12 个腿关节（动作顺序：FL, FR, RL, RR；与 GO2 保持一致的命名习惯，此处为小写）
LEGBOT_LEG_JOINT_NAMES = [
    "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
    "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
    "rl_hip_joint", "rl_thigh_joint", "rl_calf_joint",
    "rr_hip_joint", "rr_thigh_joint", "rr_calf_joint",
]
# 4 个主动轮（foot 关节，速度控制）
LEGBOT_WHEEL_JOINT_NAMES = ["fl_foot_joint", "fr_foot_joint", "rl_foot_joint", "rr_foot_joint"]

# Legbot 资产配置
LEGBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        # 轮子保留圆柱碰撞体，避免胶囊端面改变滚动接触。
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/legbot_wf/urdf/legbot_WF.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 默认关节角下轮子最低点距 base 约 0.437 m，预留少量落地高度。
        pos=(0.0, 0.0, 0.46),
        # 默认站立姿态（与 legbot 已有项目 thigh=0.9/calf=-1.8 的几何约定一致，
        # 按左右腿轴符号镜像：左腿 -y 轴、右腿 +y 轴）
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*l_thigh_joint": -0.9,
            ".*r_thigh_joint": 0.9,
            ".*l_calf_joint": 1.8,
            ".*r_calf_joint": -1.8,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 髋关节：位置控制（PD）
        "hip": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            stiffness=60.0,
            damping=4.0,
            effort_limit=120.0,
            velocity_limit=20.1,
        ),
        # 大腿关节：位置控制（PD）
        "thigh": IdealPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            stiffness=60.0,
            damping=4.0,
            effort_limit=120.0,
            velocity_limit=20.1,
        ),
        # 小腿关节：位置控制（PD）
        "calf": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            stiffness=60.0,
            damping=4.0,
            effort_limit=175.38,
            velocity_limit=13.76,
        ),
        # 主动轮：速度控制（stiffness=0 的速度伺服，damping 为速度增益）
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            stiffness=0.0,
            damping=1.0,
            effort_limit=28.68,
            velocity_limit=104.72,
        ),
    },
)
