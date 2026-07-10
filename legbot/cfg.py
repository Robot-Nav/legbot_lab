# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 本文件定义 Legbot 机器人各导航任务的环境配置参数，供训练与仿真复用。

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

@dataclass
class NoiseConfig:
    """观测噪声配置：控制各维度噪声强度，增强策略鲁棒性。"""
    level: float = 1.0                # 噪声总强度
    scale_joint_angle: float = 0.03   # 关节角度噪声尺度
    scale_joint_vel: float = 1.5      # 关节速度噪声尺度
    scale_gyro: float = 0.2           # 陀螺仪噪声尺度
    scale_gravity: float = 0.05       # 重力向量噪声尺度
    scale_linvel: float = 0.1         # 线速度噪声尺度

@dataclass
class ControlConfig:
    """控制配置：动作缩放与 PD 参数说明。"""
    # 刚度单位：牛·米/弧度，实际刚度使用 XML 中 kp 参数，仅作记录
    # 阻尼单位：牛·米·秒/弧度，实际阻尼使用 XML 中 kv 参数，仅作记录
    action_scale = 0.25  # 平地导航动作缩放系数，兼顾响应速度与稳定性
    # 力矩限制单位：牛·米，实际范围使用 XML 中 forcerange 参数

@dataclass
class InitState:
    """机器人初始状态：位姿、随机化范围与默认关节角度。"""
    # 机器人在世界坐标系中的初始位置
    pos = [0.0, 0.0, 0.5]

    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在地面范围内随机分散 20m×20m

    # 所有关节的默认目标角度：键为关节名，值为弧度
    # 采用移动站立姿态配置
    default_joint_angles = {
        "FR_hip_joint": -0.0,     # 右前髋关节
        "FR_thigh_joint": 0.9,    # 右前大腿
        "FR_calf_joint": -1.8,    # 右前小腿
        "FL_hip_joint": 0.0,      # 左前髋关节
        "FL_thigh_joint": 0.9,    # 左前大腿
        "FL_calf_joint": -1.8,    # 左前小腿
        "RR_hip_joint": -0.0,     # 右后髋关节
        "RR_thigh_joint": 0.9,    # 右后大腿
        "RR_calf_joint": -1.8,    # 右后小腿
        "RL_hip_joint": 0.0,      # 左后髋关节
        "RL_thigh_joint": 0.9,    # 左后大腿
        "RL_calf_joint": -1.8,    # 左后小腿
    }

@dataclass
class Commands:
    """目标指令范围：相对于机器人初始位置的位姿偏移。"""
    # 目标位置相对于机器人初始位置的偏移范围 [x_min, y_min, 偏航_min, x_max, y_max, 偏航_max]
    # 横向/纵向：相对机器人初始位置的偏移（米）
    # 偏航：目标绝对朝向（弧度），水平方向随机
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

@dataclass
class Normalization:
    """观测归一化系数：将原始物理量缩放到合理输入范围。"""
    lin_vel = 2.0    # 线速度归一化系数
    ang_vel = 0.25   # 角速度归一化系数
    dof_pos = 1.0    # 关节位置归一化系数
    dof_vel = 0.05   # 关节速度归一化系数

@dataclass
class Asset:
    """资产标识：机器人基座、足部与终止碰撞体名称。"""
    body_name = "base"                    # 机器人基座名称
    foot_names = ["FR", "FL", "RR", "RL"]  # 足部名称，用于接触检测
    terminate_after_contacts_on = ["collision_middle_box"]  # 基座碰撞体，触地则终止
    ground_subtree = "C_"  # 地形根节点，用于子树接触检测

@dataclass
class Sensor:
    """传感器名称：线速度、陀螺仪与足部接触。"""
    base_linvel = "base_linvel"          # 基座线速度传感器
    base_gyro = "base_gyro"              # 基座角速度传感器
    feet = ["FR", "FL", "RR", "RL"]       # 足部接触力传感器名称

@dataclass
class RewardConfig:
    """奖励函数权重：导航精度、稳定性与能量消耗的综合配置。"""
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 2.0,      # 位置误差奖励（提高10倍）
            "fine_position_tracking": 2.0,  # 精细位置奖励（提高10倍）
            "heading_tracking": 1.0,        # 朝向跟踪奖励（新增）
            "forward_velocity": 0.5,        # 前进速度奖励（鼓励朝目标移动）
            
            # ===== 移动稳定性奖励（保持但降低权重） =====
            "orientation": -0.05,           # 姿态稳定（降低权重）
            "lin_vel_z": -0.5,              # 垂直速度惩罚
            "ang_vel_xy": -0.05,            # XY轴角速度惩罚
            "torques": -1e-5,               # 扭矩惩罚
            "dof_vel": -5e-5,               # 关节速度惩罚
            "dof_acc": -2.5e-7,             # 关节加速度惩罚
            "action_rate": -0.01,           # 动作变化率惩罚
            
            # ===== 终止惩罚 =====
            "termination": -200.0,          # 终止惩罚
        }
    )

@registry.envcfg("legbot_navigation_flat")
@dataclass
class VBotEnvCfg(EnvCfg):
    """Legbot 平地导航环境的基础配置。"""
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10
    max_episode_steps: int = 1000
    sim_dt: float = 0.01    # 仿真步长 10ms，对应 100Hz
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


@registry.envcfg("legbot_navigation_stairs")
@dataclass
class VBotStairsEnvCfg(VBotEnvCfg):
    """Legbot 在楼梯地形上的导航配置，继承平地配置。"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
    max_episode_steps: int = 2000
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 楼梯导航使用 0.2，足够转向但比平地更谨慎
    
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("VBotStairsMultiTarget-v0")
@dataclass
class VBotStairsMultiTargetEnvCfg(VBotStairsEnvCfg):
    """Legbot楼梯多目标导航配置，继承单目标配置"""
    max_episode_seconds: float = 60.0  # 多目标需要更长时间
    max_episode_steps: int = 6000


@registry.envcfg("legbot_navigation_stairs_obstacles")
@dataclass
class VBotStairsObstaclesEnvCfg(VBotStairsEnvCfg):
    """Legbot楼梯地形带障碍球的导航配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_obstacles.xml"
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000

@registry.envcfg("legbot_navigation_section01")
@dataclass
class VBotSection01EnvCfg(VBotStairsEnvCfg):
    """Legbot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section01.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与legbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("legbot_navigation_section02")
@dataclass
class VBotSection02EnvCfg(VBotStairsEnvCfg):
    """Legbot Section02单独训练配置 - 中间楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section02.xml"
    max_episode_seconds: float = 60.0  # Section02较复杂，需要更多时间
    max_episode_steps: int = 6000
    
    @dataclass
    class InitState:
        # 起始位置：section02 的起始位置（继承自移动任务）
        # pos = [-2.5, 8.5, 1.8]
        # pos = [-2.5, 8.5, 1.8]
        pos = [-2.5, 12.0, 1.8]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 15.0, 3.3]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 21.0, 3.3]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 24.6, 1.8]  # Y坐标对应section02的起点，高度1.8m
        # pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        pos_randomization_range = [-0., -0., 0., 0.]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖section02区域（10-20米）
        pose_command_range = [-3.0, 16.0, 3.14, -3.0, 26.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("legbot_navigation_section03")
@dataclass
class VBotSection03EnvCfg(VBotStairsEnvCfg):
    """Legbot Section03单独训练配置 - 终点楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section03.xml"
    max_episode_seconds: float = 50.0  # 拉长一倍：从25秒增加到50秒
    max_episode_steps: int = 5000  # 拉长一倍：从2500步增加到5000步
    
    @dataclass
    class InitState:
        # 起始位置：section03 的起始位置（继承自移动任务）
        pos = [0.0, 26.0, 1.8]  # Y坐标对应section03的起点，高度1.8m
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖section03区域（20-32米）
        pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("legbot_navigation_long_course")
@dataclass
class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    """Legbot三段地形完整导航配置（比赛任务）- 使用world.xml统一地图"""
    # 使用scene_world.xml作为完整的三段地形地图（集成了world.xml）
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_world.xml"
    max_episode_seconds: float = 60.0  # 优化：减少到60秒，加快训练速度
    max_episode_steps: int = 6000  # 对应60秒 @ 100Hz
    
    @dataclass
    class InitState:
        # 起始位置：section01的中心位置
        pos = [0.0, 0.0, 1.8]  # 高台中心，高度1.8m
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖整个30米路线（section01:0-10m, section02:10-20m, section03:20-30m）
        pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 与楼梯场景保持一致
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("legbot_navigation_section001")
@dataclass
class VBotSection001EnvCfg(VBotStairsEnvCfg):
    """Legbot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section001.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与legbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)