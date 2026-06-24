# Vbot RL Lab 项目文档

> 基于 NVIDIA Isaac Lab / Isaac Sim 与 MuJoCo 的 VBot 四足机器人 **Sim2Real** 研究与部署平台。
> 本项目覆盖从大规模并行仿真训练、策略推理与模型导出，到 MuJoCo 仿真验证、再到 RobStride 实机部署的完整闭环。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Isaac Lab 2.2.0](https://img.shields.io/badge/Isaac%20Lab-2.2.0-green.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![C++17](https://img.shields.io/badge/C++-17-orange.svg)](https://isocpp.org/)

---
## 1. 项目概述

### 1.1 项目简介

Vbot RL Lab 是基于 NVIDIA Isaac Lab 框架构建的四足机器人强化学习控制项目，基于 unitree_rl_lab 开源框架深度定制。该项目包含多个子系统：

1. **Isaac Lab 训练系统**：基于 Isaac Sim 5.1.0 + Isaac Lab 2.3.0，使用 RSL-RL PPO 算法进行大规模并行训练，支持 Locomotion（速度跟踪）和 Mimic（动作模仿）任务
2. **MuJoCo 仿真系统（vbot_mujoco）**：基于 MuJoCo + Unitree SDK2 的 Sim2Real 验证仿真器，支持 C++ 和 Python 两种接口，用于控制器在仿真环境中的验证
3. **真实机器人部署**：C++ ONNX Runtime 推理 + FSM 状态机，直接控制真实 Vbot

项目实现了从仿真训练（Isaac Sim）到仿真验证（MuJoCo）再到真实机器人部署（Sim2Real）的全链路闭环，专为 Vbot 四足机器人优化。

### 1.2 核心特性

- **专为 Vbot 定制**：独立的执行器模型（T-N 曲线）、URDF 使用纯几何体（无外部 mesh 依赖）
- **Locomotion 任务**：支持速度跟踪运动（平面线速度 + 角速度指令跟踪）
- **完整部署链路**：训练 → ONNX 导出 → C++ 部署 → 真实机器人控制
- **高保真仿真**：基于 Isaac Sim 5.1.0 + Isaac Lab 2.3.0，支持 4096 并行环境
- **域随机化**：物理材质、质量、关节默认位置、质心、外力扰动等多维度随机化
- **无头模式训练**：支持 `--headless` 参数在服务器上无 GUI 训练

### 1.3 技术栈

| 类别 | 技术 |
|------|------|
| 仿真平台 | NVIDIA Isaac Sim 5.1.0 / Isaac Lab 2.3.0 |
| RL 算法库 | RSL-RL 2.3.1（PPO） |
| 深度学习 | PyTorch（CUDA 加速） |
| 模型导出 | ONNX Runtime 1.22.0 |
| 部署语言 | C++17 |
| 机器人通信 | Unitree SDK2（DDS） |
| 机器人模型 | URDF（纯几何体，无外部 mesh 依赖） |
| 构建系统 | CMake / setuptools |
| 配置管理 | YAML / dataclass / Hydra |

---

## 2. 项目文件结构

```
vbot_rl_lab/
├── source/unitree_rl_lab/          # Python 训练源码（基于 unitree_rl_lab 框架）
│   └── unitree_rl_lab/
│       ├── assets/robots/          # 机器人资产配置
│       │   ├── unitree.py          # 机器人 Articulation 配置（含 UNITREE_VBOT_CFG）
│       │   └── unitree_actuators.py # 执行器模型（含 T-N 曲线，VbotHipThigh/VbotCalf）
│       ├── tasks/                  # 任务定义
│       │   ├── locomotion/         # 速度跟踪运动任务
│       │   │   ├── mdp/            # MDP 组件
│       │   │   │   └── rewards.py  # 自定义奖励函数（energy, feet_gait 等）
│       │   │   └── robots/         # 各机器人环境配置
│       │   │       ├── vbot/       # ★ Vbot 环境配置
│       │   │       │   └── velocity_env_cfg.py
│       │   │       ├── go2/        # Go2 四足（原框架保留）
│       │   │       ├── g1/29dof/   # G1-29dof 人形（原框架保留）
│       │   │       └── h1/         # H1 人形（原框架保留）
│       │   └── mimic/              # 动作模仿任务（G1 舞蹈，原框架保留）
│       │       ├── agents/         # PPO 算法配置
│       │       ├── mdp/            # MDP 组件
│       │       │   ├── commands.py       # MotionCommand 自适应采样
│       │       │   ├── observations.py   # 模仿观测
│       │       │   ├── rewards.py        # 模仿奖励
│       │       │   ├── events.py         # 域随机化事件
│       │       │   └── terminations.py   # 终止条件
│       │       └── robots/g1_29dof/      # G1 舞蹈追踪
│       │           ├── gangnanm_style/
│       │           └── dance_102/
│       └── utils/                  # 工具函数
│           ├── export_deploy_cfg.py     # 部署配置导出
│           ├── parser_cfg.py            # 环境配置解析
│           └── mjcf_spawner.py          # MJCF 模型加载
├── deploy/                         # C++ 部署代码
│   ├── include/                    # 头文件
│   │   ├── FSM/                    # 有限状态机
│   │   │   ├── BaseState.h         # 状态基类 + REGISTER_FSM 宏
│   │   │   ├── FSMState.h          # FSMState（含手柄 DSL 解析）
│   │   │   ├── CtrlFSM.h           # 状态机控制器（1kHz 循环）
│   │   │   ├── State_Passive.h     # 被动状态
│   │   │   ├── State_FixStand.h    # 站立状态（线性插值）
│   │   │   └── State_RLBase.h      # RL 推理状态
│   │   ├── isaaclab/               # C++ 版 IsaacLab 推理框架
│   │   │   ├── algorithms/algorithms.h     # ONNX Runtime 推理引擎
│   │   │   ├── assets/articulation/articulation.h  # 关节数据结构
│   │   │   ├── envs/
│   │   │   │   ├── manager_based_rl_env.h  # 环境管理器
│   │   │   │   └── mdp/
│   │   │   │       ├── observations/observations.h  # 观测计算
│   │   │   │       ├── actions/joint_actions.h      # 动作处理
│   │   │   │       └── terminations.h               # 终止条件
│   │   │   ├── manager/
│   │   │   │   ├── observation_manager.h  # 观测管理器
│   │   │   │   ├── action_manager.h       # 动作管理器
│   │   │   │   └── manager_term_cfg.h     # 观测项配置
│   │   │   ├── devices/keyboard/keyboard.h  # 键盘控制
│   │   │   └── utils/utils.h               # 工具函数
│   │   ├── LinearInterpolator.h    # 线性插值器
│   │   ├── param.h                 # 参数/配置加载
│   │   ├── unitree_articulation.h  # 机器人关节接口（SDK 数据读取）
│   │   └── unitree_joystick_dsl.hpp # 手柄 DSL 解析器
│   ├── robots/                     # 各机器人部署程序
│   │   ├── vbot/                   # ★ Vbot 部署
│   │   │   ├── config/config.yaml  # FSM 配置
│   │   │   ├── include/Types.h     # 类型定义（复用 Go2 SDK 类型）
│   │   │   ├── src/State_RLBase.cpp # RL 状态逻辑
│   │   │   ├── main.cpp            # 主入口
│   │   │   └── CMakeLists.txt      # 构建配置
│   │   └── go2/                    # Go2 四足部署
│   │       ├── config/config.yaml
│   │       ├── include/Types.h
│   │       ├── src/State_RLBase.cpp
│   │       ├── main.cpp
│   │       └── CMakeLists.txt
│   └── thirdparty/                 # 第三方库（ONNX Runtime 1.22.0）
├── vbot_mujoco/                    # ★ MuJoCo Sim2Real 验证仿真器（Unitree 官方框架）
│   ├── simulate/                   # C++ 仿真器（推荐）
│   │   ├── src/
│   │   │   ├── main.cc             # 仿真器主入口
│   │   │   ├── unitree_sdk2_bridge.h  # SDK2 DDS 桥接（LowCmd/LowState）
│   │   │   ├── physics_joystick.h     # 手柄/键盘输入（Xbox/Switch/Keyboard）
│   │   │   ├── param.h             # YAML 配置解析
│   │   │   └── joystick/           # Linux 游戏手柄驱动
│   │   ├── config.yaml             # 仿真器配置（robot/scene/joystick）
│   │   └── CMakeLists.txt
│   ├── simulate_python/            # Python 仿真器（已移除，仅保留文档）
│   ├── unitree_robots/             # 机器人 MJCF 模型
│   │   ├── go2/                    # Go2 模型 + 场景
│   │   ├── b2/                     # B2 模型 + 场景
│   │   ├── b2w/                    # B2w 轮式模型 + 场景
│   │   └── g1/                     # G1 人形模型（23/29dof）
│   ├── terrain_tool/               # 地形生成工具（楼梯/高程图/杂乱地面）
│   ├── example/                    # Sim2Real 示例代码
│   │   ├── cpp/stand_go2.cpp       # C++ SDK2 示例
│   │   ├── python/stand_go2.py     # Python SDK2 示例
│   │   └── ros2/                   # ROS2 示例
│   ├── doc/                        # 架构图
│   ├── readme_zh.md                # 中文文档
│   └── 键盘.md                      # 键盘替代手柄控制说明
├── vbot/                           # ★ Vbot MuJoCo 导航训练环境（第三方 motrix_envs 框架）
│   ├── xmls/                       # MuJoCo XML 场景定义
│   │   ├── vbot.xml                # Vbot 机器人 MJCF（含 STL mesh 依赖）
│   │   ├── scene.xml               # 平地导航场景
│   │   ├── scene_stairs.xml        # 楼梯导航场景
│   │   ├── scene_section01.xml     # Section01 高台地形
│   │   ├── scene_section02.xml     # Section02 中间楼梯
│   │   ├── scene_section03.xml     # Section03 终点楼梯
│   │   ├── scene_world.xml         # 完整三段地形（比赛地图）
│   │   ├── 0131_V_section00.xml    # 视觉模型（OBJ + PNG 纹理）
│   │   ├── 0131_C_section01.xml    # 碰撞模型
│   │   └── assets/                 # 地形 mesh 资源（OBJ/STL/PNG）
│   ├── cfg.py                      # 导航环境配置（motrix_envs 框架）
│   └── vbot_section001_np.py       # Section01 NumPy 环境包装
├── unitree_ros/                    # 机器人模型文件（Isaac Lab 用）
│   └── robots/
│       ├── vbot_description/       # ★ Vbot 模型
│       │   ├── urdf/vbot_description.urdf  # Vbot URDF（纯几何体，无外部 mesh 依赖）
│       │   ├── meshes/             # STL 网格 + MJCF 定义
│       │   │   ├── vbot.xml        # 原始 MJCF 定义
│       │   │   └── *.STL           # 各部件 STL 网格
│       │   └── vbot_training.xml   # 训练用 MJCF
│       └── go2_description/        # Go2 模型
│           ├── urdf/go2_description.urdf
│           ├── dae/                # DAE 网格
│           ├── meshes/             # DAE 网格
│           ├── xacro/              # xacro 模板
│           └── launch/             # ROS launch 文件
├── scripts/                        # 训练/推理脚本
│   ├── rsl_rl/                     # RSL-RL 训练脚本
│   │   ├── train.py                # 训练入口
│   │   ├── play.py                 # 推理入口（含 ONNX/JIT 导出）
│   │   └── cli_args.py             # 命令行参数
│   ├── mimic/                      # 模仿学习脚本（原框架保留）
│   │   ├── csv_to_npz.py           # 运动数据转换
│   │   └── replay_npz.py           # 运动回放
│   └── list_envs.py                # 环境列表
├── logs/                           # RSL-RL 训练日志
│   └── rsl_rl/
│       └── unitree_vbot_velocity/  # Vbot 训练日志
│           └── YYYY-MM-DD_HH-MM-SS/
│               ├── model_*.pt      # 模型检查点
│               ├── params/         # 训练参数
│               │   ├── env.yaml
│               │   ├── agent.yaml
│               │   └── deploy.yaml # 部署配置
│               └── exported/       # 导出模型
│                   ├── policy.onnx
│                   └── policy.pt
├── outputs/                        # Hydra 训练输出目录
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/             # Hydra 配置
│           │   ├── config.yaml     # 合并后的完整配置
│           │   ├── hydra.yaml      # Hydra 系统配置
│           │   └── overrides.yaml  # 命令行覆盖参数
│           └── hydra.log           # 训练日志
├── unitree_rl_lab.sh               # 项目快捷脚本（train/play/install）
├── pyproject.toml                  # Python 项目配置
└── .vscode/                        # VS Code 配置
```

---

## 3. 技术架构

### 3.1 整体架构

项目包含 **三个独立子系统**，形成完整的 Sim2Real 闭环：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     子系统一：Isaac Lab 训练系统                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐                  │
│  │ Isaac Sim│  │ RSL-RL   │  │ 环境配置(MDP)     │                  │
│  │  仿真器   │←→│ PPO训练  │←→│ 观测/奖励/动作    │                  │
│  └──────────┘  └──────────┘  └──────────────────┘                  │
│        │              │                                              │
│        ▼              ▼                                              │
│  ┌──────────┐  ┌──────────┐                                         │
│  │ 4096并行  │  │ ONNX导出 │                                         │
│  │ 环境      │  │ deploy.yaml│                                       │
│  └──────────┘  └──────────┘                                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   子系统二：MuJoCo    │
              │   Sim2Real 验证      │
              │  (vbot_mujoco)       │
              │  C++ 仿真器 + DDS    │
              └──────────┬──────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                     子系统三：真实机器人部署                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐                  │
│  │ ONNX     │  │ FSM      │  │ Unitree SDK2     │                  │
│  │ Runtime  │→│ 状态机    │→│ 机器人通信        │                  │
│  └──────────┘  └──────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

> **vbot/** 目录是独立的第三方导航训练环境（motrix_envs 框架），与上述三个子系统无直接关联。

### 3.2 训练流程

训练流程的核心入口为 [train.py](file:///home/fatu08/vbot_rl_lab/scripts/rsl_rl/train.py)，其执行流程如下：

1. **解析命令行参数**：任务名、环境数、GPU 设备、种子等
2. **启动 Isaac Sim**：通过 `AppLauncher` 初始化仿真器
3. **创建环境**：`gym.make(task_name, cfg=env_cfg)` 创建 MDP 环境
4. **包装环境**：`RslRlVecEnvWrapper` 将 Isaac Lab 环境适配为 RSL-RL 接口
5. **创建训练器**：`OnPolicyRunner` 管理训练循环
6. **导出部署配置**：`export_deploy_cfg()` 生成 `deploy.yaml`
7. **训练循环**：`runner.learn()` 执行 PPO 训练

### 3.3 推理与导出流程

推理入口为 [play.py](file:///home/fatu08/vbot_rl_lab/scripts/rsl_rl/play.py)：

1. 加载训练好的 checkpoint
2. 获取推理策略 `runner.get_inference_policy()`
3. 提取观测归一化器（`actor_obs_normalizer` 或 `student_obs_normalizer`）
4. **导出为 ONNX 格式**：`export_policy_as_onnx()` 生成 `policy.onnx`
5. **导出为 JIT 格式**：`export_policy_as_jit()` 生成 `policy.pt`
6. 在仿真环境中运行推理循环

### 3.4 部署架构（Vbot）

部署侧采用 **有限状态机（FSM）** 架构，核心组件：

| 组件 | 文件 | 功能 |
|------|------|------|
| CtrlFSM | [CtrlFSM.h](file:///home/fatu08/vbot_rl_lab/deploy/include/FSM/CtrlFSM.h) | 状态机控制器，1kHz 循环检测状态切换 |
| BaseState | [BaseState.h](file:///home/fatu08/vbot_rl_lab/deploy/include/FSM/BaseState.h) | 状态基类 + `REGISTER_FSM` 宏注册机制 |
| FSMState | [FSMState.h](file:///home/fatu08/vbot_rl_lab/deploy/include/FSM/FSMState.h) | FSM 状态基类（含手柄 DSL 条件解析） |
| State_Passive | [State_Passive.h](file:///home/fatu08/vbot_rl_lab/deploy/include/FSM/State_Passive.h) | 被动状态，关节阻尼保持 |
| State_FixStand | [State_FixStand.h](file:///home/fatu08/vbot_rl_lab/deploy/include/FSM/State_FixStand.h) | 站立状态，PD 控制到目标姿态（线性插值） |
| State_RLBase | [State_RLBase.h](file:///home/fatu08/vbot_rl_lab/deploy/include/FSM/State_RLBase.h) | RL 推理状态，独立线程运行策略 |
| ManagerBasedRLEnv | [manager_based_rl_env.h](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h) | C++ 版环境管理器 |
| OrtRunner | [algorithms.h](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/algorithms/algorithms.h) | ONNX Runtime 推理引擎 |
| BaseArticulation | [unitree_articulation.h](file:///home/fatu08/vbot_rl_lab/deploy/include/unitree_articulation.h) | SDK 数据读取（IMU + 关节状态） |
| Keyboard | [keyboard.h](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/devices/keyboard/keyboard.h) | 键盘控制（独立线程读取） |
| LinearInterpolator | [LinearInterpolator.h](file:///home/fatu08/vbot_rl_lab/deploy/include/LinearInterpolator.h) | 线性插值（FixStand 状态使用） |

**状态转换流程**（Vbot 为例）：

```
Passive ──[LT+A]──→ FixStand ──[Start]──→ Velocity(RLBase)
   ↑                   │                      │
   └────────[LT+B]─────┴──────────────────────┘
```

> 注：Vbot 复用 Go2 的 SDK 类型（`unitree::robot::go2`），通过 DDS 通信。

---

## 4. 核心功能模块

### 4.1 Locomotion（速度跟踪运动）

Locomotion 任务的目标是训练机器人跟踪给定的速度指令（线速度 $v_x$, $v_y$ 和角速度 $\omega_z$），实现稳定的运动控制。

#### 4.1.1 观测空间（Vbot）

**策略观测（Policy）**——部署时可用：

| 观测项 | 函数 | 维度 | 缩放 | 噪声 |
|--------|------|------|------|------|
| 基座角速度 | `base_ang_vel` | 3 | ×0.2 | ±0.2 |
| 投影重力 | `projected_gravity` | 3 | - | ±0.05 |
| 速度指令 | `generated_commands` | 3 | - | - |
| 关节相对位置 | `joint_pos_rel` | 12 | - | ±0.01 |
| 关节相对速度 | `joint_vel_rel` | 12 | ×0.05 | ±1.5 |
| 上一步动作 | `last_action` | 12 | - | - |

> Vbot 策略观测总维度：3+3+3+12+12+12 = **45**

**评论家观测（Critic）**——训练时特权信息：

额外包含：基座线速度 `base_lin_vel`（3维）、关节力矩 `joint_effort`（12维，缩放 ×0.01）。

> Vbot 不使用 `height_scanner`、`gait_phase` 和 `history_length`（H1/G1 使用）。

#### 4.1.2 动作空间

采用 **关节位置目标** 控制模式：

```python
JointPositionAction = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=0.25,
    use_default_offset=True,
    clip={".*": (-100.0, 100.0)},
)
```

最终关节目标位置计算公式：

$$q_{target} = q_{default} + 0.25 \times a$$

其中 $a \in \mathbb{R}^{12}$ 为策略网络输出的动作向量，$q_{default}$ 为默认关节角度。

#### 4.1.3 奖励函数设计（Vbot）

奖励函数采用 **加权和** 形式，分为任务奖励和正则化奖励两大类：

**任务奖励（正向）**：

| 奖励项 | 函数 | 权重 | 参数 |
|--------|------|------|------|
| 线速度跟踪 | `track_lin_vel_xy_exp` | 1.5 | std=0.5 |
| 角速度跟踪 | `track_ang_vel_z_exp` | 0.75 | std=0.5 |
| 足部腾空时间 | `feet_air_time` | 0.1 | threshold=0.5 |

**正则化奖励（负向惩罚）**：

| 奖励项 | 函数 | 权重 | 说明 |
|--------|------|------|------|
| Z轴线速度 | `lin_vel_z_l2` | -2.0 | 惩罚垂直方向运动 |
| XY角速度 | `ang_vel_xy_l2` | -0.05 | 惩罚非指令角速度 |
| 关节速度 | `joint_vel_l2` | -0.001 | 惩罚关节速度 |
| 关节加速度 | `joint_acc_l2` | -2.5e-7 | 惩罚关节加速度 |
| 关节力矩 | `joint_torques_l2` | -2e-4 | 惩罚关节力矩 |
| 动作变化率 | `action_rate_l2` | -0.1 | 惩罚动作抖动 |
| 关节位置限制 | `joint_pos_limits` | -10.0 | 惩罚接近关节限位 |
| 能量消耗 | `energy` | -2e-5 | $\sum|\dot{q}_i| \cdot |\tau_i|$ |
| 姿态倾斜 | `flat_orientation_l2` | -2.5 | 惩罚基座倾斜 |
| 关节偏差 | `joint_position_penalty` | -0.7 | stand_still_scale=5.0, velocity_threshold=0.3 |
| 腾空时间方差 | `air_time_variance_penalty` | -1.0 | 惩罚步态不均匀 |
| 足部滑动 | `feet_slide` | -0.1 | 惩罚足部滑动 |
| 非期望接触 | `undesired_contacts` | -1.0 | hip/thigh/calf 接触（threshold=1） |

> **Vbot 与其他机器人的差异**：Vbot 不使用 `alive` 奖励、`feet_gait` 步态奖励、`foot_clearance_reward` 足部离地奖励、`base_height` 高度奖励和 `joint_deviation_*` 关节偏差奖励。这些奖励仅在人形机器人（H1/G1）中使用。

#### 4.1.4 终止条件（Vbot）

| 条件 | 说明 |
|------|------|
| `time_out` | 达到最大 episode 时长（20s） |
| `bad_orientation` | 基座倾斜角超过阈值（$\cos^{-1}(g_z) > 0.8$ rad ≈ 45.8°） |
| `base_contact` | 基座接触地面（threshold=10.0） |

> Vbot 不使用 `base_height` 终止条件（仅人形机器人使用）。

#### 4.1.5 课程学习

项目实现了两级课程学习：

1. **地形课程**：`terrain_levels_vel` — 随训练进展逐步增加地形难度
2. **速度课程**：`lin_vel_cmd_levels` — 逐步扩大速度指令范围

速度课程的核心逻辑：当跟踪奖励超过权重 × 0.8 时，将速度范围扩大 ±0.1：

```python
if reward > reward_term.weight * 0.8:
    ranges.lin_vel_x = clamp(ranges + [-0.1, 0.1], limit_ranges)
```

Vbot 速度指令范围：

| 参数 | 初始范围 | 极限范围 |
|------|----------|----------|
| lin_vel_x | [-0.1, 0.1] | [-1.0, 1.0] |
| lin_vel_y | [-0.1, 0.1] | [-0.4, 0.4] |
| ang_vel_z | [-1, 1] | [-1.0, 1.0] |

#### 4.1.6 域随机化（Vbot）

| 随机化项 | 模式 | 参数范围 |
|----------|------|----------|
| 摩擦系数 | startup | 静摩擦 [0.3, 1.2]，动摩擦 [0.3, 1.2] |
| 恢复系数 | startup | [0.0, 0.15] |
| 基座质量 | startup | 加载 [-1.0, 3.0] kg |
| 基座位置 | reset | x,y ∈ [-0.5, 0.5]，yaw ∈ [-π, π] |
| 关节位置 | reset | 缩放 [1.0, 1.0]，速度 [-1.0, 1.0] |
| 推力扰动 | interval | 5~10s 间隔，速度 ±0.5 m/s |

### 4.2 Mimic（动作模仿）

Mimic 任务的目标是训练机器人模仿参考运动轨迹（如舞蹈动作），实现全身运动跟踪。该任务仅支持 G1-29dof 机器人，包含两种舞蹈风格：`gangnanm_style` 和 `dance_102`。

#### 4.2.1 运动数据管线

```
BVH 动捕数据 (.csv)
    │
    ▼  csv_to_npz.py（Isaac Sim 中重定向）
运动数据 (.npz)
    │  包含：joint_pos, joint_vel, body_pos_w, body_quat_w,
    │        body_lin_vel_w, body_ang_vel_w
    ▼
MotionCommand（训练时实时采样）
    │
    ▼  线性插值
MotionLoader_（部署时 CSV 回放）
```

[csv_to_npz.py](file:///home/fatu08/vbot_rl_lab/scripts/mimic/csv_to_npz.py) 的核心流程：
1. 加载 CSV 格式的动捕数据（基座位置/旋转 + 关节角度）
2. 将输入帧率重采样到输出帧率（默认 50fps），使用 **线性插值（LERP）** 和 **球面线性插值（SLERP）**
3. 通过数值微分计算速度：线速度 $\frac{d\mathbf{p}}{dt}$、关节速度 $\frac{dq}{dt}$、角速度 $\frac{d\mathbf{R}}{dt} \to \omega$
4. 在 Isaac Sim 中重定向到机器人骨骼，记录所有刚体的位姿和速度
5. 保存为 `.npz` 格式

#### 4.2.2 MotionCommand 自适应采样

[MotionCommand](file:///home/fatu08/vbot_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/commands.py) 是 Mimic 任务的核心组件，实现了 **自适应运动采样** 策略：

**基本原理**：将运动序列按时间分箱（bin），统计每个 bin 的失败频率，优先采样失败率高的时间段。

**采样概率计算**：

$$P(i) = \frac{(f_i + \frac{u}{K}) * h}{\sum_j (f_j + \frac{u}{K}) * h}$$

其中：
- $f_i$ 为 bin $i$ 的指数移动平均失败率
- $u$ 为均匀采样比例（默认 0.1）
- $K$ 为总 bin 数
- $h$ 为卷积核 $h[k] = \lambda^k / \sum \lambda^k$（$\lambda = 0.8$）

**失败率更新**（指数移动平均）：

$$f_i^{new} = \alpha \cdot c_i + (1 - \alpha) \cdot f_i^{old}$$

其中 $\alpha = 0.001$，$c_i$ 为当前 episode 在 bin $i$ 的失败计数。

**采样指标**：
- 采样熵 $H_{norm} = -\sum P(i) \log P(i) / \log K$（衡量采样均匀性）
- Top-1 概率和 Top-1 bin 位置

#### 4.2.3 Mimic 观测空间

**策略观测**：

| 观测项 | 函数 | 说明 |
|--------|------|------|
| 运动指令 | `generated_commands` | 参考关节位置 + 速度 |
| 参考锚点朝向 | `motion_anchor_ori_b` | 参考躯干朝向（body frame） |
| 基座角速度 | `base_ang_vel` | 机器人角速度 |
| 关节相对位置 | `joint_pos_rel` | 当前关节位置 |
| 关节相对速度 | `joint_vel_rel` | 当前关节速度 |
| 上一步动作 | `last_action` | 前一步动作 |

**特权观测（Critic）**：

额外包含：参考锚点位置 `motion_anchor_pos_b`、参考锚点朝向 `motion_anchor_ori_b`、机器人身体位置 `robot_body_pos_b`、机器人身体朝向 `robot_body_ori_b`、基座线速度 `base_lin_vel`。

#### 4.2.4 Mimic 奖励函数

所有跟踪奖励均采用 **指数核** 形式：

$$r = \exp\left(-\frac{e^2}{\sigma^2}\right)$$

| 奖励项 | 误差 $e$ | $\sigma$ | 权重 |
|--------|----------|----------|------|
| 锚点位置 | $\|\mathbf{p}_{anchor}^{ref} - \mathbf{p}_{anchor}^{robot}\|^2$ | 0.3 | 0.5 |
| 锚点朝向 | $\|q_{anchor}^{ref} \ominus q_{anchor}^{robot}\|^2$ | 0.4 | 0.5 |
| 身体位置 | $\|\mathbf{p}_{body}^{ref} - \mathbf{p}_{body}^{robot}\|^2$ | 0.3 | 1.0 |
| 身体朝向 | $\|q_{body}^{ref} \ominus q_{body}^{robot}\|^2$ | 0.4 | 1.0 |
| 身体线速度 | $\|\mathbf{v}_{body}^{ref} - \mathbf{v}_{body}^{robot}\|^2$ | 1.0 | 1.0 |
| 身体角速度 | $\|\omega_{body}^{ref} - \omega_{body}^{robot}\|^2$ | 3.14 | 1.0 |

正则化项：关节加速度（-2.5e-7）、关节力矩（-1e-5）、动作变化率（-0.1）、关节限制（-10.0）、非期望接触（-0.1）。

#### 4.2.5 Mimic 域随机化

额外包含：
- **关节默认位置随机化**：`randomize_joint_default_pos`，偏移 ±0.52 rad
- **质心随机化**：`randomize_rigid_body_com`，偏移 x∈[-0.025, 0.025]，y∈[-0.05, 0.05]，z∈[-0.05, 0.05]
- **更频繁的推力扰动**：1~3 秒间隔

---

## 5. 算法原理与数学公式

### 5.1 PPO 算法

项目使用 **Proximal Policy Optimization (PPO)** 算法，由 RSL-RL 库实现。

#### 5.1.1 网络结构

```
Actor (策略网络):  obs → [512] → ELU → [256] → ELU → [128] → ELU → action
Critic (价值网络): obs → [512] → ELU → [256] → ELU → [128] → ELU → value
```

初始标准差 $\sigma_0 = 1.0$，采用可学习的对数标准差参数化。

#### 5.1.2 PPO 核心公式

**优势函数估计（GAE）**：

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}$$

其中 TD 误差：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

参数：$\gamma = 0.99$，$\lambda = 0.95$。

**策略目标函数**：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中重要性采样比：

$$\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

裁剪参数 $\epsilon = 0.2$。

**价值函数损失**：

$$L^{VF}(\theta) = \hat{\mathbb{E}}_t \left[ \left( V_\theta(s_t) - V_t^{targ} \right)^2 \right]$$

使用裁剪的价值损失（`use_clipped_value_loss=True`）。

**熵正则化**：

$$L^{Entropy} = -\mathcal{H}(\pi_\theta) = \hat{\mathbb{E}}_t \left[ \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]$$

熵系数 $\beta = 0.01$（Locomotion）/ $0.005$（Mimic）。

**总目标函数**：

$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 L^{Entropy}(\theta)$$

其中 $c_1 = 1.0$（价值损失系数），$c_2 = \beta$。

#### 5.1.3 自适应学习率

采用 **KL 散度自适应** 学习率调度：

- 目标 KL 散度：$\delta_{KL} = 0.01$
- 当实际 KL 散度 > $1.5 \delta_{KL}$ 时，学习率 × 0.5
- 当实际 KL 散度 < $\delta_{KL} / 1.5$ 时，学习率 × 2.0
- 初始学习率：$\alpha_0 = 1 \times 10^{-3}$

#### 5.1.4 训练超参数

| 参数 | Vbot Locomotion | G1-29dof Mimic |
|------|-----------|-------|
| 每环境步数 | 24 | 24 |
| 最大迭代 | 50000 | 30000 |
| 保存间隔 | 100 | 500 |
| Mini-batch 数 | 4 | 4 |
| 训练 Epoch | 5 | 5 |
| 经验归一化 | False | False |
| 梯度裁剪 | 1.0 | 1.0 |
| 种子 | 42 | - |

### 5.2 步态相位编码

[observations.py](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h) 中 C++ 版步态相位编码：

$$\phi(t) = \frac{(t \cdot \Delta t) \mod T}{T}$$

$$\text{gait\_phase} = [\sin(2\pi\phi), \, \cos(2\pi\phi)]$$

> 该观测仅 H1（周期 0.6s）和 G1（周期 0.8s）使用，Vbot 不使用。

### 5.3 步态奖励

[rewards.py](file:///home/fatu08/vbot_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py) 中的 `feet_gait` 函数实现了基于相位的步态奖励：

$$r_{gait} = \sum_{i} \neg(\text{is\_stance}_i \oplus \text{is\_contact}_i)$$

其中：
- $\text{is\_stance}_i = (\phi_i < \theta_{threshold})$，$\theta_{threshold} = 0.55$
- $\phi_i = (\phi_{global} + \text{offset}_i) \mod 1.0$
- 偏移量：左脚 `offset=0.0`，右脚 `offset=0.5`（交替步态）

该奖励鼓励机器人在步态的支撑相保持足部接触，在摆动相抬起足部。

> Vbot 不使用步态奖励，仅 H1/G1 使用。

### 5.4 执行器模型

[unitree_actuators.py](file:///home/fatu08/vbot_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py) 实现了执行器的 **T-N 曲线模型**（含 Vbot 专用执行器），继承自 `DelayedPDActuator`：

```
    扭矩限制 (N·m)
        ^
  Y2─────|
        |──────────────Y1
        |              │\
        |              │ \
        |              │  \
        |              |   \
  ──────+──────────────|──────> 速度 (rad/s)
                      X1   X2
```

**力矩裁剪逻辑**：

$$\tau_{max} = \begin{cases} Y_1 & \text{if } \text{sgn}(\dot{q}) = \text{sgn}(\tau) \text{ and } |\dot{q}| < X_1 \\ Y_2 & \text{if } \text{sgn}(\dot{q}) \neq \text{sgn}(\tau) \text{ and } |\dot{q}| < X_1 \\ k(|\dot{q}| - X_1) + \tau_{max} & \text{if } |\dot{q}| \geq X_1 \end{cases}$$

其中 $k = -\tau_{max} / (X_2 - X_1)$。

**摩擦模型**：

$$\tau_{applied} = \tau_{PD} - F_s \cdot \tanh\left(\frac{\dot{q}}{V_a}\right) - F_d \cdot \dot{q}$$

其中 $F_s$ 为静摩擦系数，$F_d$ 为动摩擦系数，$V_a$ 为摩擦激活速度（默认 0.01）。

**各执行器参数**：

| 型号 | X1 (rad/s) | X2 (rad/s) | Y1 (N·m) | Y2 (N·m) | Fs | Fd | Armature |
|------|-----------|-----------|----------|----------|-----|-----|----------|
| **VbotHipThigh** ★ | **15.0** | **30.0** | **17.0** | **17.0** | **0.2** | **0.02** | **0.0043** |
| **VbotCalf** ★ | **10.0** | **20.0** | **34.0** | **34.0** | **0.2** | **0.02** | **0.04** |
| Go2HV | 13.5 | 30 | 20.2 | 23.4 | - | - | - |
| N7520-14.3 | 22.63 | 35.52 | 71 | 83.3 | 1.6 | 0.16 | 0.01018 |
| N7520-22.5 | 14.5 | 22.7 | 111 | 131 | 2.4 | 0.24 | 0.02510 |
| N5020-16 | 30.86 | 40.13 | 24.8 | 31.9 | 0.6 | 0.06 | 0.00361 |
| W4010-25 | 15.3 | 24.76 | 4.8 | 8.6 | 0.6 | 0.06 | 0.00425 |
| M107-15 | 14.0 | 25.6 | 150 | 182.8 | - | - | 0.06326 |
| M107-24 | 8.8 | 16 | 240 | 292.5 | - | - | 0.16048 |

### 5.5 关节刚度/阻尼计算（Mimic 模式）

Mimic 模式下，G1-29dof 的关节刚度和阻尼基于 **二阶系统自然频率** 计算：

$$K = J \cdot \omega_n^2$$

$$D = 2 \zeta \cdot J \cdot \omega_n$$

其中：
- $\omega_n = 10 \times 2\pi \approx 62.83$ rad/s（自然频率 10Hz）
- $\zeta = 2.0$（过阻尼）
- $J$ 为转子惯量（armature）

### 5.6 动作缩放计算（Mimic 模式）

Mimic 模式的动作缩放因子根据执行器参数自动计算：

$$\text{scale}_j = 0.25 \times \frac{\tau_{max,j}}{K_j}$$

这确保了动作输出在关节力矩限制范围内。

---

## 6. 部署系统

### 6.1 C++ 推理框架

部署侧在 C++ 中重新实现了 Python 训练侧的核心逻辑，确保 Sim2Real 的一致性：

| Python 组件 | C++ 对应 | 功能 |
|-------------|----------|------|
| ObservationManager | [observation_manager.h](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/manager/observation_manager.h) | 观测计算与缩放（`REGISTER_OBSERVATION` 宏注册） |
| ActionManager | [action_manager.h](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/manager/action_manager.h) | 动作处理与缩放（`REGISTER_ACTION` 宏注册） |
| ManagerBasedRLEnv | [manager_based_rl_env.h](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h) | 环境管理 |
| ONNX 导出 | [OrtRunner](file:///home/fatu08/vbot_rl_lab/deploy/include/isaaclab/algorithms/algorithms.h) | ONNX Runtime 推理 |

### 6.2 部署配置导出

[export_deploy_cfg.py](file:///home/fatu08/vbot_rl_lab/source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py) 在训练时自动导出 `deploy.yaml`，包含：

- **关节映射**：`joint_ids_map` — Isaac Lab 关节顺序到 SDK 关节顺序的映射
- **控制参数**：`step_dt`、`stiffness`、`damping`、`default_joint_pos`
- **命令范围**：速度指令的极限范围
- **动作配置**：缩放因子、裁剪范围、偏移量
- **观测配置**：缩放因子、裁剪范围、历史长度

### 6.3 推理循环

```cpp
// State_RLBase::enter() 中启动独立策略线程
policy_thread = std::thread([this]{
    using clock = std::chrono::high_resolution_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(env->step_dt));
    auto sleepTill = clock::now() + dt;
    env->reset();
    while (policy_thread_running) {
        env->step();  // robot->update() → obs → ONNX → action → joint_cmd
        std::this_thread::sleep_until(sleepTill);
        sleepTill += dt;
    }
});
```

每个 `env->step()` 的执行流程：

1. `robot->update()` — 从 SDK 读取关节状态（IMU + 关节位置/速度）
2. `observation_manager->compute()` — 计算观测向量（含缩放和裁剪）
3. `alg->act(obs)` — ONNX Runtime 前向推理
4. `action_manager->process_action(action)` — 处理动作（含缩放和偏移）

`State_RLBase::run()` 在 FSM 主循环（1kHz）中执行，将处理后的动作写入 SDK 命令：

```cpp
void State_RLBase::run() {
    auto action = env->action_manager->processed_actions();
    for(int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
```

### 6.4 Vbot 部署配置

Vbot 的 FSM 配置（[config.yaml](file:///home/fatu08/vbot_rl_lab/deploy/robots/vbot/config/config.yaml)）：

**Passive 状态**：
- 关节模式：全部为 1（阻尼模式）
- 关节阻尼：kd = [3] × 12

**FixStand 状态**：
- 关节刚度：kp = [40] × 12
- 关节阻尼：kd = [4] × 12
- 关键姿态：站立 `[0.0, 1.0, -1.5] × 4`，默认 `[0.0, 0.8, -1.5] × 4`
- 使用线性插值平滑过渡

**Velocity 状态**：
- 策略目录：`../../../logs/rsl_rl/Unitree_Vbot_Velocity`
- 自动查找最新包含 `exported/` 文件夹的训练运行

**安全检查**：
- State_RLBase 注册了 `bad_orientation` 检查（limit_angle=1.0 rad ≈ 57.3°）
- 所有状态注册了 DDS 超时检查，超时自动回到 Passive

### 6.5 C++ 观测实现

部署侧通过 `REGISTER_OBSERVATION` 宏注册观测函数，与训练侧保持一致：

| 观测项 | C++ 实现 | 说明 |
|--------|----------|------|
| `base_ang_vel` | 读取 IMU 陀螺仪数据 | 3维 |
| `projected_gravity` | IMU 四元数 × 重力向量 | 3维 |
| `joint_pos_rel` | 关节位置 - 默认位置 | 12维 |
| `joint_vel_rel` | 关节速度 | 12维 |
| `last_action` | 上一步动作 | 12维 |
| `velocity_commands` | 手柄摇杆 + 配置范围裁剪 | 3维 |
| `gait_phase` | 全局相位 sin/cos 编码 | 2维（仅 H1） |

### 6.6 Vbot SDK 接口

Vbot 复用 Go2 的 SDK 类型（[Types.h](file:///home/fatu08/vbot_rl_lab/deploy/robots/vbot/include/Types.h)）：

```cpp
using LowCmd_t = unitree::robot::go2::publisher::LowCmd;
using LowState_t = unitree::robot::go2::subscription::LowState;
```

数据读取通过 [unitree_articulation.h](file:///home/fatu08/vbot_rl_lab/deploy/include/unitree_articulation.h) 的 `BaseArticulation` 模板类实现：
- IMU 陀螺仪 → `root_ang_vel_b`
- IMU 四元数 → `root_quat_w` → `projected_gravity_b`
- 关节状态 → `joint_pos` / `joint_vel`（按 `joint_ids_map` 重映射）

---

## 7. MuJoCo Sim2Real 验证仿真器（vbot_mujoco）

`vbot_mujoco/` 是基于 **Unitree SDK2 + MuJoCo** 开发的 Sim2Real 验证仿真器，与 Isaac Lab 训练系统独立运行。用户通过 Unitree SDK2 / ROS2 开发的控制器可以直接接入该仿真器进行验证，实现 **同一套控制代码** 在仿真和真实机器人上无缝切换。

### 7.1 核心功能

- **DDS 通信桥接**：通过 `unitree_sdk2_bridge.h` 实现 MuJoCo 与 Unitree SDK2 的 DDS 消息互通
  - 订阅 `rt/lowcmd`（电机控制指令）→ 写入 MuJoCo `ctrl`
  - 发布 `rt/lowstate`（电机状态）← 读取 MuJoCo `sensordata`
  - 发布 `rt/sportmodestate`（机器人位姿速度）
  - 发布 `rt/wirelesscontroller`（手柄/键盘状态）
- **多机器人支持**：Go2、B2、B2w、G1（23/29dof），通过 `robot` 配置切换
- **三种输入方式**：Xbox 手柄、Switch 手柄、**键盘**（无需物理手柄）
- **虚拟挂带**：人形机器人（H1/G1）初始化吊起/放下模拟
- **地形生成工具**：参数化创建楼梯、杂乱地面、高程图等地形

### 7.2 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     vbot_mujoco 仿真器                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   MuJoCo     │  │ Unitree SDK2 │  │   手柄/键盘       │  │
│  │  物理仿真     │←→│  DDS 桥接    │←→│  输入设备        │  │
│  │  (mjModel)   │  │  (Bridge)    │  │  (Joystick)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                  │
│  │  用户控制器   │  ← 同一套代码，仿真/实物无缝切换      │
│  │ (C++/Python/ │                                  │
│  │  ROS2)       │                                  │
│  └──────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 关键组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 主入口 | [main.cc](file:///home/fatu08/vbot_rl_lab/vbot_mujoco/simulate/src/main.cc) | MuJoCo 仿真循环 + 渲染 + 弹性带控制 |
| SDK2 桥接 | [unitree_sdk2_bridge.h](file:///home/fatu08/vbot_rl_lab/vbot_mujoco/simulate/src/unitree_sdk2_bridge.h) | DDS 消息 ↔ MuJoCo 数据转换 |
| 手柄抽象 | [physics_joystick.h](file:///home/fatu08/vbot_rl_lab/vbot_mujoco/simulate/src/physics_joystick.h) | Xbox/Switch/Keyboard 统一接口 |
| 配置解析 | [param.h](file:///home/fatu08/vbot_rl_lab/vbot_mujoco/simulate/src/param.h) | YAML 配置 + 命令行参数解析 |
| 仿真配置 | [config.yaml](file:///home/fatu08/vbot_rl_lab/vbot_mujoco/simulate/config.yaml) | 机器人/场景/DDS/手柄配置 |

### 7.4 机器人桥接类型

```cpp
// 根据执行器数量自动判断机器人类型
if (m->nu > NUM_MOTOR_IDL_GO) {  // > 20 个电机 → G1
    interface = std::make_unique<G1Bridge>(m, d);
} else {  // ≤ 20 个电机 → Go2/B2/B2w/Go2w
    interface = std::make_unique<Go2Bridge>(m, d);
}
```

| 机器人类型 | 桥接类 | SDK 消息类型 | 特殊功能 |
|-----------|--------|-------------|----------|
| Go2/B2/B2w/Go2w | `Go2Bridge` | `unitree_go` idl | 标准四足 |
| G1 | `G1Bridge` | `unitree_hg` idl | 二级 IMU、BMS、mode_machine |

### 7.5 键盘控制映射

vbot_mujoco 支持 **键盘替代手柄**，通过 `glfwGetKey()` 轮询实现：

| 键盘按键 | 对应手柄功能 | 类型 |
|---------|------------|------|
| **W / S** | 左摇杆 Y 轴 (前进/后退) | 轴 |
| **A / D** | 左摇杆 X 轴 (左/右平移) | 轴 |
| **↑ / ↓** | 右摇杆 Y 轴 (俯仰) | 轴 |
| **← / →** | 右摇杆 X 轴 (偏航) | 轴 |
| **Q** | LT 左扳机 | 轴 |
| **E** | RT 右扳机 | 轴 |
| **Space** | A 按钮 | 按钮 |
| **Left Shift** | B 按钮 | 按钮 |
| **Enter** | Start 按钮 | 按钮 |
| **Tab** | Back/Select 按钮 | 按钮 |

### 7.6 使用方式

```bash
# 1. 编译仿真器
cd vbot_mujoco/simulate/
mkdir build && cd build
cmake ..
make -j4

# 2. 启动仿真器（加载 Go2 + 地形场景）
./unitree_mujoco -r go2 -s scene_terrain.xml

# 3. 在另一个终端运行控制器（同一套代码可直接用于实物）
cd deploy/robots/go2/build
./go2_ctrl --network wlp132s0  # 实物：指定网卡
# 或
./go2_ctrl                     # 仿真：默认 localhost
```

### 7.7 与 Isaac Lab 训练系统的关系

| 维度 | Isaac Lab 训练 | vbot_mujoco 验证 |
|------|---------------|-----------------|
| **目的** | 策略训练（PPO） | 控制器 Sim2Real 验证 |
| **仿真器** | Isaac Sim 5.1.0 | MuJoCo 3.3.6 |
| **并行环境** | 4096 环境 | 单环境 |
| **通信方式** | 内部 Python API | DDS（与实物一致） |
| **控制接口** | 直接输出动作 | 接收 LowCmd（tau/kp/kd/q/dq） |
| **使用场景** | 大规模并行训练 | 单机器人控制算法验证 |

> **vbot_mujoco 不包含 Vbot 机器人模型**。当前支持 Go2、B2、B2w、G1。如需验证 Vbot 控制器，需要将 Vbot 的 MJCF 模型和对应 SDK 桥接添加到 vbot_mujoco 中。

---

## 8. 仿真环境配置（Isaac Lab）

### 8.1 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `sim.dt` | 0.005s | 物理仿真步长（200Hz） |
| `decimation` | 4 | 控制频率降采样比 |
| 控制频率 | 50Hz | 1/(0.005×4) = 50Hz |
| `episode_length_s` | 20s | Locomotion / Mimic |
| `num_envs` | 4096 | 并行环境数 |
| `env_spacing` | 2.5m | 环境间距 |

### 8.2 地形配置

Vbot Locomotion 使用程序化生成的地形：

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),           # 每块地形尺寸
    border_width=20.0,         # 边界宽度
    num_rows=10,               # 行数（课程维度）
    num_cols=20,               # 列数
    difficulty_range=(0.0, 1.0),  # 难度范围
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.1),
    },
)
```

物理材质：静摩擦 1.0，动摩擦 1.0。

### 8.3 机器人配置

各机器人的初始状态和执行器参数在 [unitree.py](file:///home/fatu08/vbot_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) 中定义：

| 机器人 | 自由度 | 初始高度 | 关节数 | 执行器类型 | 默认关节位置 |
|--------|--------|----------|--------|-----------|-------------|
| **Vbot** ★ | **12** | **0.462m** | **12** | **VbotHipThigh / VbotCalf** | **hip=0, thigh=0.9, calf=-1.8** |
| Go2 | 12 | 0.4m | 12 | Go2HV | hip=±0.1, thigh=0.8/1.0, calf=-1.5 |
| Go2W | 16 | 0.45m | 16 | Go2HV | thigh=0.8, calf=-1.5, foot=0 |
| B2 | 12 | 0.58m | 12 | M107-24 | hip=±0.1, thigh=0.8/1.0, calf=-1.5 |
| G1-23dof | 23 | 0.8m | 23 | N7520/N5020 | hip_pitch=-0.1, knee=0.3 |
| G1-29dof | 29 | 0.8m | 29 | N7520/N5020/W4010 | hip_pitch=-0.1, knee=0.3 |
| H1 | 19 | 1.1m | 19 | M107-24/GO2HV | hip_pitch=-0.1, knee=0.3 |

**Vbot 执行器 PD 参数**：stiffness=80.0, damping=6.0, friction=0.01

---

## 9. 设计模式与工程实践

### 9.1 MDP 管理器模式

项目采用 Isaac Lab 的 **Manager-Based RL Env** 架构，将 MDP 各组件解耦为独立管理器：

- `ObservationManager`：管理观测项的计算、缩放、噪声注入
- `ActionManager`：管理动作的处理、缩放、偏移
- `RewardManager`：管理奖励项的加权和计算
- `CommandManager`：管理指令的采样与更新
- `EventManager`：管理域随机化事件的触发
- `TerminationManager`：管理终止条件的判断
- `CurriculumManager`：管理课程学习策略

### 9.2 配置即代码

所有环境配置均使用 Python `@configclass` 装饰器定义，兼具类型安全和可读性：

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    ...
```

### 9.3 训练-部署一致性保障

- 训练时通过 `export_deploy_cfg()` 导出完整的部署配置
- C++ 部署侧使用相同的 `deploy.yaml` 配置
- 观测和动作的缩放、裁剪、偏移在两侧完全一致
- ONNX 模型导出包含观测归一化器
- C++ 观测函数通过 `REGISTER_OBSERVATION` 宏注册，与 Python 侧一一对应

### 9.4 FSM 注册机制

部署侧使用宏注册 FSM 状态类，实现可扩展的状态机：

```cpp
REGISTER_FSM(State_RLBase)  // 自动注册到全局工厂映射
```

`CtrlFSM` 通过 YAML 配置动态创建状态实例，支持灵活的状态转换定义。状态转换条件使用手柄 DSL（[unitree_joystick_dsl.hpp](file:///home/fatu08/vbot_rl_lab/deploy/include/unitree_joystick_dsl.hpp)）解析，如 `LT + A.on_pressed`、`Start.on_pressed` 等。

### 9.5 URDF 资源替换机制

[unitree.py](file:///home/fatu08/vbot_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) 中的 `UnitreeUrdfFileCfg.replace_asset()` 方法实现了 URDF 资源的自动处理：

1. 将 URDF 复制到 `/tmp/IsaacLab/unitree_rl_lab/<robot_name>/`
2. 替换 `package://` 引用为相对路径
3. 符号链接 mesh/dae 目录，确保资源可解析

---

## 10. 与原 unitree_rl_lab / unitree_rl_gym 对比

Vbot RL Lab 基于 unitree_rl_lab 框架深度定制开发。以下是原版 unitree_rl_gym 与 vbot_rl_lab 的对比，帮助理解项目底层框架差异。

### 10.1 仿真平台对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **仿真引擎** | Isaac Gym | Isaac Sim + Isaac Lab |
| **版本要求** | Isaac Gym Preview | Isaac Sim 5.1.0 + Isaac Lab 2.3.0 |
| **渲染方式** | GPU 加速的物理仿真 | 完整物理仿真 + 高保真渲染 |
| **并行环境** | 支持 | 支持（4096 环境） |
| **地形生成** | 基础地形 | 程序化地形生成器（课程学习） |

> **关键差异**：Isaac Gym 是 NVIDIA 较早期的仿真平台，而 Isaac Lab 是基于 Isaac Sim 的新一代框架，提供更真实的物理仿真和更丰富的传感器支持。

### 10.2 部署方式对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **主要部署语言** | Python（deploy_real.py） | C++ |
| **模型格式** | PyTorch `.pt` / LibTorch | ONNX `.onnx` |
| **推理引擎** | LibTorch | ONNX Runtime |
| **C++ 示例** | 仅 G1 有 C++ 示例 | Vbot/Go2 有完整 C++ 部署 |
| **FSM 状态机** | 无 | 完整 FSM 架构（Passive/FixStand/Velocity） |

> **关键差异**：`unitree_rl_gym` 主要使用 Python 部署，依赖 LibTorch；`vbot_rl_lab` 采用纯 C++ 部署，使用 ONNX Runtime，更适合实时控制场景。

### 10.3 任务类型对比

| 任务类型 | unitree_rl_gym | vbot_rl_lab |
|----------|----------------|----------------|
| **Locomotion（速度跟踪）** | ✅ | ✅ |
| **Mimic（动作模仿）** | ❌ | ✅ |
| **舞蹈追踪** | ❌ | ✅（G1-29dof） |

> **关键差异**：`unitree_rl_lab` 新增了 **Mimic 任务**，支持从动捕数据学习复杂动作（如舞蹈），包含自适应运动采样算法。

### 10.4 机器人支持对比

| 机器人 | unitree_rl_gym | vbot_rl_lab |
|--------|----------------|----------------|
| **Vbot** | ❌ | ✅ ★ |
| Go2 | ✅ | ✅ |
| Go2W | ❌ | ✅ |
| B2 | ❌ | ✅ |
| G1 | ✅ | ✅ |
| G1-23dof | ❌ | ✅ |
| G1-29dof | ❌ | ✅ |
| H1 | ✅ | ✅ |
| H1_2 | ✅ | ✅ |

> **关键差异**：`vbot_rl_lab` 专为 Vbot 四足机器人定制，同时保留了对宇树系列机器人的支持。

### 10.5 架构设计对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **基础框架** | legged_gym | Isaac Lab |
| **环境架构** | 传统 RL 环境 | Manager-Based RL Env |
| **配置方式** | Python 配置类 | `@configclass` + YAML |
| **MDP 组件管理** | 集成在环境类中 | 独立 Manager（观测/动作/奖励/命令/事件） |
| **域随机化** | 基础随机化 | 多维度随机化（物理材质/质量/质心/关节偏移/推力） |
| **课程学习** | 有限支持 | 地形课程 + 速度课程 |

### 10.6 RSL-RL 版本对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **RSL-RL 版本** | v1.0.2 | v2.3.1 |
| **分布式训练** | 不支持 | 支持 |
| **模型导出** | `.pt` 文件 | ONNX + JIT |

### 10.7 执行器建模对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **执行器模型** | 理想 PD 控制器 | T-N 曲线 + 摩擦模型 |
| **力矩限制** | 固定限制 | 速度相关力矩限制 |
| **摩擦建模** | 无 | 静摩擦 + 动摩擦 |

> **关键差异**：`vbot_rl_lab` 实现了更真实的执行器模型，包含 T-N 曲线和摩擦特性，有助于 Sim2Real 迁移。

### 10.8 观测空间对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **观测历史** | 无 | 支持（G1-29dof: history_length=5） |
| **步态相位编码** | 无 | 支持（H1: sin/cos 编码） |
| **高度扫描** | 基础 | RayCaster 支持（可选） |
| **特权观测** | 基础 | 完整 Critic 观测 |

### 10.9 奖励函数对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **奖励项数量** | 基础奖励 | 更丰富（能量/步态/足部离地等） |
| **步态奖励** | 无 | 基于相位的步态奖励（H1/G1） |
| **足部离地奖励** | 基础 | 带速度调制的离地奖励（H1/G1） |
| **Mimic 奖励** | 无 | 全身跟踪奖励（位置/朝向/速度） |

### 10.10 部署配置对比

| 特性 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|----------------|
| **配置导出** | 手动配置 | 自动导出 `deploy.yaml` |
| **关节映射** | 手动处理 | 自动生成 `joint_ids_map` |
| **观测/动作缩放** | 手动配置 | 自动导出缩放参数 |

### 10.11 综合对比总结

| 维度 | unitree_rl_gym | vbot_rl_lab |
|------|----------------|-------------|
| **定位** | 入门级、快速原型 | 生产级、完整部署 |
| **仿真平台** | Isaac Gym（较旧） | Isaac Lab（新一代） |
| **部署方式** | Python 为主 | C++ 为主 |
| **模型格式** | PyTorch | ONNX |
| **目标机器人** | Go2 / G1 / H1 | **Vbot**（四足，12DOF） |
| **执行器建模** | 理想模型 | 真实 T-N 曲线（Vbot 专用） |
| **模型格式** | URDF + mesh | URDF（纯几何体，无外部 mesh 依赖） |
| **适合场景** | 研究、学习、快速验证 | Vbot 生产部署、Sim2Real |

**选择建议**：
- 如果你需要宇树系列通用机器人的快速原型，`unitree_rl_gym` 更简单易用
- 如果你需要 Vbot 四足机器人的生产级训练与部署，`vbot_rl_lab` 是专为此定制的方案

---

## 11. 训练状态

### 11.1 当前训练记录

最新训练运行：`2026-05-17_17-49-10`

- 实验名称：`unitree_vbot_velocity`
- 检查点数量：302+（model_0.pt ~ model_30200.pt）
- 保存间隔：每 100 次迭代
- 训练种子：42
- 最大迭代：50000

### 11.2 训练输出结构

```
logs/rsl_rl/unitree_vbot_velocity/<run_name>/
├── model_0.pt, model_100.pt, ...    # 模型检查点
├── params/
│   ├── env.yaml                     # 环境配置
│   ├── agent.yaml                   # 算法配置
│   └── deploy.yaml                  # 部署配置
└── exported/
    ├── policy.onnx                  # ONNX 模型
    └── policy.pt                    # JIT 模型
```

---

## 12. 总结

Vbot RL Lab 是基于 unitree_rl_lab 框架深度定制的四足机器人强化学习训练框架，包含三个核心子系统：

**子系统一：Isaac Lab 训练系统**
1. **端到端闭环**：从仿真训练（Isaac Sim）到真实 Vbot 部署的完整工具链
2. **高保真仿真**：基于 Isaac Sim 的大规模并行仿真，4096 环境同时训练
3. **Vbot 专用执行器建模**：定制的 T-N 曲线模型（VbotHipThigh：±17 N·m，VbotCalf：±34 N·m）、摩擦模型、惯量参数
4. **丰富的域随机化**：物理材质、质量、质心、关节偏移、外力扰动等多维度随机化
5. **自适应课程学习**：地形和速度指令的渐进式难度提升
6. **URDF 纯几何体**：Isaac Lab 训练用的 URDF 使用纯几何体（box/cylinder/sphere），无外部 mesh 依赖，便于分发和加载
7. **无头模式训练**：支持 `--headless` 在远程服务器上进行大规模训练

**子系统二：MuJoCo Sim2Real 验证仿真器（vbot_mujoco）**
8. **DDS 通信桥接**：MuJoCo ↔ Unitree SDK2，同一套控制代码仿真/实物无缝切换
9. **多机器人支持**：Go2、B2、B2w、G1（23/29dof），通过配置切换
10. **键盘替代手柄**：无需物理手柄即可控制仿真器
11. **地形生成工具**：参数化创建楼梯、杂乱地面、高程图等复杂地形

**子系统三：真实机器人部署**
12. **C++ 高性能部署**：ONNX Runtime 推理 + 实时控制线程，满足 50Hz 控制频率要求
13. **灵活的 FSM 架构**：宏注册机制 + YAML 配置，支持动态状态转换和手柄 DSL 条件解析

### 训练命令速查

```bash
# 无头模式训练
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --headless

# 恢复训练
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --resume --headless

# 推理演示（含 ONNX 导出）
python scripts/rsl_rl/play.py --task Unitree-Vbot-Velocity

# 使用快捷脚本
./unitree_rl_lab.sh -t --task Unitree-Vbot-Velocity   # 训练
./unitree_rl_lab.sh -p --task Unitree-Vbot-Velocity   # 推理
./unitree_rl_lab.sh -l                                 # 列出环境
./unitree_rl_lab.sh -i                                 # 安装
```

## 13. 致谢
感谢开源项目unitree_rl_lab 框架，为本项目提供了算法思路以及引导。
