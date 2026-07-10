# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""LegBot 机器人配置。

LegBot 为四足机器人，与 Unitree Go2 具有相同的 12 自由度运动学结构
（4 条腿 × 3 个关节）。本配置用于在 IsaacLab 中替换 Go2 模型进行训练。
参考模型：legbot_description/urdf/legbot_description.urdf
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


LEGBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=f'{ISAACLAB_ASSETS_DATA_DIR}/legbot/urdf/legbot.urdf',
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            '.*_hip_joint': 0.0,
            '.*_thigh_joint': 0.9,
            '.*_calf_joint': -1.8,
        },
        joint_vel={'.*': 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        'hip_thigh': DCMotorCfg(
            joint_names_expr=['.*_hip_joint', '.*_thigh_joint'],
            effort_limit=16.0,
            saturation_effort=16.0,
            velocity_limit=30.0,
            stiffness=50.0,
            damping=3,
            friction=0.0,
        ),
        'calf': DCMotorCfg(
            joint_names_expr=['.*_calf_joint'],
            effort_limit=32.0,
            saturation_effort=32.0,
            velocity_limit=15.7,
            stiffness=50.0,
            damping=3,
            friction=0.0,
        ),
    },
)
