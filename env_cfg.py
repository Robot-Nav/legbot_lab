# ============== Legbot 强化学习环境配置 ==============
# 包含：场景定义、观测、动作、奖励、域随机化、终止条件、课程学习

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 复用Go2的MDP模块（奖励、观测、域随机化、命令、课程学习、地形）
# Legbot与Go2具有相同的12关节运动结构和命名约定
import robot_lab.tasks.go2.mdp as mdp
from robot_lab.assets.legbot import LEGBOT_CFG
from robot_lab.tasks.go2.mdp.terrains import TERRAIN_CFG

# ============== 基础常量定义 ==============
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

BASE_LINK_NAME = "base"
FOOT_LINK_NAME = ".*_foot"
# Legbot站立高度（默认角度 thigh=0.9, calf=-1.8）
# foot_z = -0.1985*cos(0.9) - 0.214*cos(0.9-1.8) + 0.021 ≈ 0.28m
BASE_HEIGHT_TARGET = 0.28  # 目标高度，与真实机器人质心高度(0.277m)对齐

##
# Scene definition - 场景配置
##

@configclass
class LegbotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the Legbot robot."""

    # 地形：使用生成器创建粗糙地形，支持课程学习
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False
    )

    # 机器人：Legbot关节配置
    robot: ArticulationCfg = LEGBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描器：大范围（1.6x1.0m），用于critic观测
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 高度扫描器：小范围（0.4x0.3m），用于奖励计算和critic观测
    height_scanner_small = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 接触力传感器：检测足部接触，用于终止条件和奖励
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 灯光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings - MDP配置（命令、动作、观测、域随机化、奖励、终止条件、课程学习）
##

# ============== 命令配置 ==============
# 定义机器人需要跟踪的目标速度命令
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.Go2RLGymCommandCfg()  # 线速度(xy)和角速度(z)命令

# ============== 动作配置 ==============
# 定义策略输出的动作如何映射到关节位置
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 腿部关节：位置控制，动作缩放系数0.25（动作范围[-1,1]映射到关节偏移[-0.25,0.25]rad）
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={".*_hip_joint": 0.25, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )

# ============== 观测配置 ==============
# Policy观测：带噪声和历史帧（10帧），用于策略网络输入
# Critic观测：无噪声，额外信息（线速度、关节力矩、接触力、高度扫描），用于价值估计
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - 策略网络观测（带噪声和历史帧）"""
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,  # 基座角速度
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 投影重力方向（基座姿态）
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,  # 目标速度命令
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,  # 关节位置（相对于默认位置）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,  # 关节速度（相对于默认速度）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,  # 上一步动作
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        def __post_init__(self):
            self.history_length = 10  # 10帧历史观测，用于时序建模
            self.enable_corruption = True  # 启用噪声扰动
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group - 价值网络观测（无噪声，特权信息）"""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,  # 基座线速度（真实值，策略无法观测）
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,  # 关节加速度（特权信息）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1e-4,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,  # 关节力矩（特权信息）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.01,
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,  # 足部接触力（特权信息）
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
            clip=(-100.0, 100.0),
            scale=1e-3,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,  # 高度扫描（特权信息，用于地形感知）
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=2.5,
        )
        def __post_init__(self):
            self.enable_corruption = False  # Critic不使用噪声
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        """Single timestep observation - 单帧观测（用于MoE-CTS模型）"""
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1  # 仅当前帧，无历史

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    single_obs: SingleObsCfg = SingleObsCfg()  # MoE-CTS模型使用单帧观测

# ============== 域随机化配置 ==============
# 在训练过程中随机化物理参数，提高策略的泛化能力
# mode="startup": 训练开始时随机化一次（质量、惯性、摩擦）
# mode="reset": 每次episode重置时随机化（关节零位、PD增益、基座状态）
# mode="interval": 定期随机化（外力扰动，每4秒）
@configclass
class EventCfg:
    """Configuration for events - 域随机化事件配置"""

    # 质量随机化 - base质量±1kg变化，适应负载变化
    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 1.0),  # ±1kg
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    # 质量随机化 - 其他部件质量±10%变化
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.9, 1.1),  # ±10%
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    # 惯性随机化 - 所有部件惯性±10%变化
    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.9, 1.1),  # ±10%
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # 质心位置随机化 - base质心±5cm偏移
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # ±5cm
        },
    )
    # 关节初始位置随机化 - 重置时关节位置在默认值的50%~150%
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),  # 50%~150%
            "velocity_range": (0.0, 0.0),
        },
    )
    # 执行器增益随机化 - PD参数±10%变化，适应电机差异
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),  # kp ±10%
            "damping_distribution_params": (0.9, 1.1),  # kd ±10%
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # 关节零位偏移随机化 - ±35mrad，模拟编码器误差
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",
            "offset_range": (-0.035, 0.035),  # ±35mrad ≈ ±2°
        },
    )
    # 外力扰动随机化 - 每4秒施加随机速度扰动
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),  # 每4秒扰动一次
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),  # 线速度扰动 ±0.4 m/s
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),  # 角速度扰动 ±0.6 rad/s
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6)
            }
        }
    )
    # 摩擦系数随机化 - 静摩擦0~2，动摩擦0~2，恢复系数0~0.5
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,  # 64种摩擦组合
            "make_consistent": True
        },
    )
    # 基座初始状态随机化 - 重置时随机位置和速度
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14)},  # 位置±0.5m，yaw全范围
            "velocity_range": {
                "x": (-0.5, 0.5),  # 线速度 ±0.5 m/s
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),  # 角速度 ±0.5 rad/s
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


# ============== 奖励配置 ==============
# 正奖励：鼓励跟踪命令速度
# 负奖励：惩罚不期望的行为（垂直速度、高度偏差、关节加速度、力矩、碰撞等）
@configclass
class RewardsCfg:
    """Reward terms for the MDP - 奖励函数配置"""

    # === 跟踪奖励（正奖励） ===
    # 线速度跟踪：指数奖励，std=0.5控制容差，weight=1.0为主要目标
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    # 角速度跟踪：指数奖励，weight=0.5为次要目标
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # === 惩罚奖励（负奖励） ===
    # 垂直速度惩罚：防止机器人上下晃动
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 水平角速度惩罚：防止侧向旋转
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 关节加速度惩罚：鼓励平滑运动
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    )

    # 关节功率惩罚：减少能耗
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    )
    # 关节力矩惩罚：避免过大扭矩
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    )
    # 基座高度惩罚：保持在目标高度0.28m
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        }
    )
    # 动作变化率惩罚：鼓励平滑动作
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 动作平滑性惩罚：三阶导数惩罚
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    # 不期望接触惩罚：大腿和小腿不应接触地面
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 5.0},
    )
    # 关节位置限制惩罚：避免超出关节范围
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    # 足部调节惩罚：足部高度和间距约束
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    # hip关节位置惩罚：静止时保持hip在0附近
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.hip_pos_penalty_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 1.0,
            "command_threshold": 0.1,
        },
    )
    # 大腿/小腿位置惩罚：静止时保持默认姿态
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,
        },
    )

# ============== 终止条件配置 ==============
# 定义何时结束episode
@configclass
class TerminationsCfg:
    """Termination terms for the MDP - 终止条件配置"""
    # 超时终止：episode达到最大时长25秒
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 非法接触终止：base接触地面（摔倒）
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "threshold": 1.0  # 接触力>1N即判定摔倒
        },
    )

# ============== 课程学习配置 ==============
# 随训练进度逐渐增加难度或调整奖励权重
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP - 课程学习配置"""
    # 地形难度课程：随训练进度增加地形复杂度
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_gym)
    # 垂直速度惩罚课程：初期weight=-2.0逐渐减弱到0（1500次迭代）
    base_linear_velocity = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "lin_vel_z_l2", "initial_weight": -2.0, "final_weight": -0.0, "start_it": 0, "end_it": 1500
        })
    # 高度惩罚课程：初期weight=-1.0逐渐增强到-10.0（5000次迭代）
    base_height_l2 = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "base_height_l2", "initial_weight": -1.0, "final_weight": -10.0, "start_it": 0, "end_it": 5000
        })

##
# Environment configuration - 环境主配置
##

@configclass
class LegbotEnvCfg(ManagerBasedRLEnvCfg):
    """Merged configuration for the Legbot robot on rough terrain - Legbot强化学习环境总配置"""

    # Scene settings - 场景设置
    scene: LegbotSceneCfg = LegbotSceneCfg(num_envs=4096, env_spacing=0.5)  # 4096个并行环境，间距0.5m
    # Basic settings - 基础设置
    observations: ObservationsCfg = ObservationsCfg()  # 观测配置
    actions: ActionsCfg = ActionsCfg()  # 动作配置
    commands: CommandsCfg = CommandsCfg()  # 命令配置
    # MDP settings - MDP设置
    rewards: RewardsCfg = RewardsCfg()  # 奖励配置
    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件配置
    events: EventCfg = EventCfg()  # 域随机化配置
    curriculum: CurriculumCfg = CurriculumCfg()  # 课程学习配置

    def __post_init__(self):
        """Post initialization - 后初始化：仿真参数设置"""
        # General settings - 通用设置
        self.decimation = 4  # 控制频率降采样：仿真20ms/控制周期80ms
        self.episode_length_s = 25.0  # 每个episode最长25秒
        # Simulation settings - 仿真设置
        self.sim.dt = 0.005  # 仿真步长5ms（200Hz物理仿真）
        self.sim.render_interval = self.decimation  # 渲染间隔=控制周期

        # Physics material settings from subclass - 物理材质设置
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = int(1 * 1024 * 1024)  # GPU最大刚体patch数：1M
        self.sim.physx.gpu_collision_stack_size = int(512 * 1024 * 1024)  # GPU碰撞堆栈：512MB
        self.sim.physx.enable_external_forces_every_iteration = True  # 每步启用外力

        # Update sensor periods - 传感器更新周期设置
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 80ms更新一次
        if self.scene.height_scanner_small is not None:
            self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt  # 接触力每步更新（5ms）

        # Handle curriculum for terrain generator - 地形课程学习设置
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True  # 启用地形课程
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
