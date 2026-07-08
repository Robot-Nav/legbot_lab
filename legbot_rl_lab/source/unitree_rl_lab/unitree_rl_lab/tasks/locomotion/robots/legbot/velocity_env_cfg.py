# 导入数学库，用于数学运算
import math

# 导入Isaac Lab仿真工具
import isaaclab.sim as sim_utils
# 导入Isaac Lab地形生成模块
import isaaclab.terrains as terrain_gen
# 导入机器人关节配置类
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# 导入基于管理器的强化学习环境配置类
from isaaclab.envs import ManagerBasedRLEnvCfg
# 导入课程学习、事件、观测、奖励、场景实体和终止条件配置类
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
# 导入交互场景配置类
from isaaclab.scene import InteractiveSceneCfg
# 导入接触传感器和射线投射器配置类
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
# 导入地形导入器配置类
from isaaclab.terrains import TerrainImporterCfg
# 导入配置装饰器和资产路径工具
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# 导入噪声配置类
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入机器人配置
from unitree_rl_lab.assets.robots.unitree import UNITREE_LEGBOT_CFG as ROBOT_CFG
# 导入运动控制MDP模块
from unitree_rl_lab.tasks.locomotion import mdp

# 定义鹅卵石道路地形生成器配置
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),                # 地形块大小
    border_width=20.0,               # 边界宽度
    num_rows=10,                     # 行数
    num_cols=20,                     # 列数
    horizontal_scale=0.1,            # 水平缩放
    vertical_scale=0.005,            # 垂直缩放
    slope_threshold=0.75,            # 坡度阈值
    difficulty_range=(0.0, 1.0),     # 难度范围
    use_cache=False,                 # 不使用缓存
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),  # 平坦地形占比
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """足式机器人地形场景配置"""

    # 地形导入器配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",                     # 地形prim路径
        terrain_type="generator",                       # 地形类型为生成器
        terrain_generator=COBBLESTONE_ROAD_CFG,        # 使用鹅卵石地形生成器
        max_init_terrain_level=1,                      # 最大初始地形等级
        collision_group=-1,                             # 碰撞组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",          # 摩擦力组合模式
            restitution_combine_mode="multiply",       # 恢复系数组合模式
            static_friction=1.0,                       # 静摩擦系数
            dynamic_friction=1.0,                      # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,                          # 投影UVW
            texture_scale=(0.25, 0.25),                # 纹理缩放
        ),
        debug_vis=False,                               # 关闭调试可视化
    )
    # 机器人配置，替换prim路径以支持多环境
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描器（射线投射），用于地形感知
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",         # 安装在机器人基座上
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 偏移量
        ray_alignment="yaw",                            # 射线对齐方式
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格模式
        debug_vis=False,                                # 关闭调试可视化
        mesh_prim_paths=["/World/ground"],             # 检测的地面网格
    )
    # 接触力传感器，用于检测脚部接触
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 天空光照配置
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,                            # 光照强度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",  # 天空纹理
        ),
    )


@configclass
class EventCfg:
    """事件配置，用于环境随机化"""

    # 随机化物理材质参数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",                                  # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.5),       # 静摩擦范围（扩大以提升sim2real泛化）
            "dynamic_friction_range": (0.25, 1.5),      # 动摩擦范围
            "restitution_range": (0.0, 0.25),           # 恢复系数范围
            "num_buckets": 64,                          # 桶数量
        },
    )

    # 随机增加基座质量
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # base 7.09kg，±1.0kg随机化
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",                         # 增加操作
        },
    )

    # 随机化基座质心位置
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.005, 0.005), "z": (-0.002, 0.002)},  # ±1cm 覆盖 sim2real COM 偏置
        },
    )

    # 随机化其他部件质量（腿/电池等），缩放因子 0.9~1.1
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.95, 1.05),       # ±5%
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # 随机化所有部件惯性，缩放因子 0.9~1.1
    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.95, 1.05),    # ±5%
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 随机化执行器增益（Kp/Kd），缩放因子 0.9~1.1，startup 模式
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),  # Kp 缩放范围
            "damping_distribution_params": (0.9, 1.1),   # Kd 缩放范围
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 随机化关节零位偏移（编码器误差），±10mrad，startup 模式
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="startup",
        params={
            "action_term_name": "JointPositionAction",
            "offset_range": (-0.01, 0.01),
        },
    )

    # 随机化关节摩擦系数，缩放因子 0.5~3.0（标称值 0.01）
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 3.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 随机化电机力矩限幅（Y1/Y2），缩放因子 0.7~1.0
    randomize_torque_limit = EventTerm(
        func=mdp.randomize_actuator_torque_limit,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "torque_limit_scale": (0.75, 1.05),
        },
    )

    # 施加外部力/力矩（开启轻度扰动）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),               # 力范围（N）
            "torque_range": (-1.0, 1.0),                # 力矩范围（N·m）
        },
    )

    # 复位机器人基座状态
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},  # 位置和偏航范围
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # 复位机器人关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),               # 位置范围
            "velocity_range": (-1.0, 1.0),              # 速度范围
        },
    )

    # 随机推动机器人
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",                                 # 间隔模式
        interval_range_s=(5.0, 10.0),                   # 间隔时间范围（更频繁扰动）
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},  # 速度范围（扩大）
    )


@configclass
class CommandsCfg:
    """MDP指令配置"""

    # 基座速度指令
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),             # 重采样时间
        rel_standing_envs=0.1,                           # 站立环境比例
        debug_vis=True,                                  # 开启调试可视化
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),                      # 线速度X范围
            lin_vel_y=(-0.1, 0.1),                      # 线速度Y范围
            ang_vel_z=(-1, 1)                           # 角速度Z范围
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),                      # 线速度X限制
            lin_vel_y=(-0.6, 0.6),                      # 线速度Y限制
            ang_vel_z=(-1.0, 1.0),                      # 角速度Z限制
        ),
    )


@configclass
class ActionsCfg:
    """MDP动作配置"""

    # 关节位置动作
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",                              # 机器人名称
        joint_names=[".*"],                              # 所有关节
        scale=0.25,                                      # 动作缩放
        use_default_offset=True,                         # 使用默认偏移
        clip={".*": (-100.0, 100.0)},                    # 动作裁剪范围
    )


@configclass
class ObservationsCfg:
    """MDP观测配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组"""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.3, n_max=0.3))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True               # 启用观测扰动
            self.concatenate_terms = True               # 拼接观测项

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """评论家网络观测组"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """MDP奖励函数配置"""

    # -- 任务奖励：跟踪速度指令
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- 基座惩罚
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)        # 垂直速度惩罚
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.12)    # 俯仰/横滚角速度惩罚
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)                # 关节速度惩罚
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5e-7)                # 关节加速度惩罚
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)         # 关节力矩惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.18)              # 动作变化率惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-8.0)        # 关节位置限位惩罚
    energy = RewTerm(func=mdp.energy, weight=-2e-5)                          # 能量消耗惩罚

    # -- 机器人姿态奖励
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0) # 基座姿态偏离惩罚

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,                    # 站立姿势缩放
            "velocity_threshold": 0.3,                   # 速度阈值
        },
    )

    # -- 脚部奖励
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.3,                            # 空中时间阈值
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- 其他惩罚：非期望接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """MDP终止条件配置"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)                    # 超时终止
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )                                                                        # 基座接触终止
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})  # 姿态异常终止


@configclass
class CurriculumCfg:
    """课程学习配置"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)                  # 地形难度课程
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)                   # 速度指令课程


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """速度跟踪运动环境配置"""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)   # 场景配置
    observations: ObservationsCfg = ObservationsCfg()                       # 观测配置
    actions: ActionsCfg = ActionsCfg()                                      # 动作配置
    commands: CommandsCfg = CommandsCfg()                                   # 指令配置
    rewards: RewardsCfg = RewardsCfg()                                      # 奖励配置
    terminations: TerminationsCfg = TerminationsCfg()                       # 终止条件配置
    events: EventCfg = EventCfg()                                           # 事件配置
    curriculum: CurriculumCfg = CurriculumCfg()                             # 课程配置

    def __post_init__(self):
        """后初始化"""
        self.decimation = 4                                                # 控制降采样率
        self.episode_length_s = 20.0                                        # 回合时长
        self.sim.dt = 0.005                                                 # 仿真步长
        self.sim.render_interval = self.decimation                         # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material    # 物理材质

        # PhysX GPU内存配置
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22

        # 传感器更新周期配置
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 地形课程配置
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """部署/测试环境配置"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32                                              # 减少环境数量
        self.scene.terrain.terrain_generator.num_rows = 2                    # 减少地形行数
        self.scene.terrain.terrain_generator.num_cols = 1                    # 减少地形列数
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges  # 使用全部速度范围