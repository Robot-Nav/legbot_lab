# Legbot Lab

> 基于 NVIDIA Isaac Lab 框架的四足机器人 Legbot 强化学习训练与 Sim2Real 部署项目

[![Platform](https://img.shields.io/badge/Platform-Linux-orange)](https://www.linux.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.2.0-green)](https://isaac-sim.github.io/IsaacLab/)
[![PPO](https://img.shields.io/badge/RL-PPO-red)](https://arxiv.org/abs/1707.06347)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 目录

- [项目简介](#项目简介)
- [算法原理](#算法原理)
  - [PPO 算法概述](#ppo-算法概述)
  - [算法公式](#算法公式)
  - [网络架构](#网络架构)
  - [域随机化与 Sim2Real](#域随机化与-sim2real)
- [项目结构](#项目结构)
- [机器人参数](#机器人参数)
- [快速开始](#快速开始)
  - [1. 安装 Isaac Lab](#1-安装-isaac-lab)
  - [2. 安装本项目](#2-安装本项目)
  - [3. 开始训练](#3-开始训练)
  - [4. 模型推理与导出](#4-模型推理与导出)
- [Sim2Real 部署](#sim2real-部署)
  - [系统架构](#系统架构)
  - [编译与运行](#编译与运行)
- [奖励函数设计](#奖励函数设计)
- [训练参数](#训练参数)
- [MDP 定义](#mdp-定义)
- [技术栈](#技术栈)
- [项目团队](#项目团队)

---

## 项目简介

Legbot Lab 是一个完整的四足机器人强化学习（RL）训练与部署框架，目标是从仿真中训练出高性能的运动控制策略，并将其无缝部署到真实机器人上（Sim2Real）。

**核心特点：**

- **完整的 Sim2Real 闭环**：训练 → ONNX 导出 → C++ 控制器部署 → 实物运行
- **PPO 强化学习**：基于 RSL-RL 实现的 Proximal Policy Optimization 算法
- **DDS 通信架构**：控制器与硬件解耦，仿真与实物共用同一套 C++ 控制代码
- **丰富的域随机化**：质量、惯性、摩擦、PD 增益、外力扰动等多维度随机化
- **安全保护机制**：多级限幅、超限保护、安全状态机（FSM）
- **4096 并行环境**：GPU 加速训练，充分利用 Isaac Sim 的并行仿真能力

---

## 算法原理

### PPO 算法概述

本项目使用 **Proximal Policy Optimization (PPO)** 算法训练四足机器人的运动控制策略。PPO 由 OpenAI 于 2017 年提出，是一种基于策略梯度的 Actor-Critic 强化学习算法。

PPO 的核心思想是通过**裁剪策略更新幅度**来限制新旧策略之间的差异，从而稳定训练过程，同时保持较高的样本效率。

### 算法公式

#### 1. 策略梯度目标函数（Clipped Surrogate Objective）

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是优势函数估计值
- $\epsilon$ 是裁剪参数（本项目设为 **0.2**）

#### 2. 广义优势估计 (GAE)

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 残差：
- $\gamma$ 是折扣因子（本项目设为 **0.99**）
- $\lambda$ 是 GAE 参数（本项目设为 **0.95**）

#### 3. 价值函数损失

$$L^{VF}(\phi) = \hat{\mathbb{E}}_t \left[ \frac{1}{2} \left( V_\phi(s_t) - \hat{R}_t \right)^2 \right]$$

其中 $\hat{R}_t = A_t + V_\phi(s_t)$ 是目标回报。

#### 4. 总损失函数

$$L(\theta, \phi) = L^{CLIP}(\theta) - c_1 \cdot L^{VF}(\phi) + c_2 \cdot S[\pi_\theta](s_t)$$

其中 $S[\pi_\theta]$ 是策略熵，鼓励探索。本项目设置：
- 价值损失系数 $c_1 = 1.0$
- 熵系数 $c_2 = 0.01$

#### 5. 自适应学习率（KL 散度控制）

PPO 使用自适应学习率，基于 KL 散度动态调整：

$$\text{if } KL > \text{desired_kl} \times 2: \ \alpha \leftarrow \alpha \times 1.5$$
$$\text{if } KL < \text{desired_kl} \times 0.5: \ \alpha \leftarrow \alpha \times 0.67$$

本项目目标 KL 散度设为 **0.01**。

#### 6. PD 控制器

策略输出关节位置命令，通过 PD 控制器转换为力矩：

$$\tau = K_p \cdot (q_{des} - q) + K_d \cdot (\dot{q}_{des} - \dot{q})$$

其中 $K_p = 60$ N·m/rad，$K_d = 4.0$ N·m·s/rad。

#### 7. 动作映射

$$q_{des} = q_{default} + 0.25 \cdot a$$

其中 $a \in [-1, 1]^{12}$ 是策略网络输出，$q_{default}$ 是默认站立姿态：
- Hip 关节：$0.0$ rad
- Thigh 关节：$0.9$ rad  
- Calf 关节：$-1.8$ rad

### 网络架构

| 组件 | 架构 |
|------|------|
| Actor（策略网络） | MLP: `[obs_dim] → 512 → 256 → 128 → 12`，ELU 激活 |
| Critic（价值网络） | MLP: `[critic_obs_dim] → 512 → 256 → 128 → 1`，ELU 激活 |
| 初始化 | 正交初始化，初始噪声标准差 = 1.0 |

策略网络的观测包含 **10 帧历史**（`history_length=10`），用于捕捉时序动态信息。

### 域随机化与 Sim2Real

为使策略在真实机器人上也能良好工作，训练中引入了大量域随机化：

| 参数 | 范围 | 模式 |
|------|------|------|
| 基座质量 | ±1.0 kg | 启动时一次性 |
| 连杆质量 | ×0.9~1.1 | 启动时一次性 |
| 转动惯量 | ×0.9~1.1 | 启动时一次性 |
| 质心位置 | ±5 cm | 启动时一次性 |
| 摩擦系数 | 0.0~2.0 | 启动时一次性 |
| 恢复系数 | 0.0~0.5 | 启动时一次性 |
| PD 增益 | ×0.9~1.1 | 每次重置 |
| 关节零位偏移 | ±0.035 rad | 每次重置 |
| 初始关节位置 | 50%~150% 默认值 | 每次重置 |
| 外力扰动 | ±0.4 m/s | 每4秒一次 |
| 基座初始状态 | 位置±0.5m, 速度±0.5m/s | 每次重置 |
| 观测噪声 | 多种（见下表） | 每步 |

**观测噪声配置：**

| 观测量 | 噪声范围 | 缩放 |
|--------|---------|------|
| 角速度 | U(-0.2, 0.2) | 0.25 |
| 重力投影 | U(-0.05, 0.05) | 1.0 |
| 关节位置 | U(-0.03, 0.03) | 1.0 |
| 关节速度 | U(-2.0, 2.0) | 0.05 |

---

## 项目结构

```
legbot_mujoco/
├── env_cfg.py                        # 主环境配置（4096并行环境PPO训练）
├── legbot_rl_lab/                    # 核心 RL 训练与部署系统
│   ├── source/unitree_rl_lab/        # Python 训练源码（基于 Isaac Lab）
│   │   └── unitree_rl_lab/
│   │       ├── assets/robots/        # Legbot 机器人配置
│   │       │   ├── unitree.py        # UNITREE_LEGBOT_CFG
│   │       │   └── unitree_actuators.py # 执行器 T-N 曲线模型
│   │       ├── tasks/locomotion/
│   │       │   ├── agents/           # PPO 算法配置
│   │       │   ├── mdp/              # MDP 组件（奖励/观测/域随机化/命令）
│   │       │   └── robots/legbot/    # Legbot 环境配置
│   │       └── utils/                # 工具函数（部署配置导出）
│   ├── scripts/                      # 训练/推理/测试脚本
│   │   └── rsl_rl/
│   │       ├── train.py              # 训练入口（headless 模式）
│   │       ├── play.py               # 推理/ONNX 导出
│   │       └── cli_args.py           # 命令行参数
│   ├── deploy/                       # C++ 部署代码（Sim2Real）
│   │   ├── include/
│   │   │   ├── FSM/                  # 有限状态机
│   │   │   │   ├── CtrlFSM.h         # 1kHz 主状态机
│   │   │   │   ├── State_FixStand.h  # 站姿状态
│   │   │   │   ├── State_Passive.h   # 被动阻尼状态
│   │   │   │   └── State_RLBase.h    # RL 策略运行状态
│   │   │   ├── deploy_safety.h       # 安全保护（力矩/温度/姿态超限）
│   │   │   ├── deploy_csv_logger.h   # 50Hz CSV 诊断日志
│   │   │   └── param.h               # 命令行参数解析
│   │   └── robots/legbot/
│   │       ├── config/config.yaml    # FSM 配置 & 安全参数
│   │       ├── include/Types.h       # DDS 接口定义
│   │       ├── main.cpp              # 控制器入口
│   │       └── src/State_RLBase.cpp  # ONNX Runtime 推理
│   ├── unitree_ros/robots/legbot_description/ # URDF & MJCF 模型
│   └── logs/rsl_rl/unitree_legbot_velocity/   # 训练日志 & 模型
├── simulate/                         # MuJoCo DDS 仿真器
│   ├── src/
│   │   ├── main.cc                   # 仿真主循环
│   │   ├── legbot_bridge.h           # DDS↔MuJoCo 桥接
│   │   └── physics_joystick.h        # 手柄/键盘驱动
│   └── config.yaml                   # 仿真配置
├── serial_dds_gateway/               # 串口⇔DDS 硬件网关
│   ├── src/legbot_rt_gait_pd.cpp     # 网关主程序（500Hz）
│   ├── include/                      # IMU 帧解析/电机协议
│   └── start_gateway.sh              # 一键启动脚本
├── legbot/                           # NumPy 训练环境（独立导航任务）
│   ├── cfg.py                        # 场景配置
│   ├── legbot_section001_np.py       # 训练入口
│   └── xmls/
│       ├── legbot.xml                # Legbot MJCF 定义
│       ├── scene_stairs.xml          # 楼梯场景
│       └── scene_world.xml           # 完整赛道场景
├── terrain_tool/                     # 地形生成工具
└── unitree_sdk2/                     # 宇树 DDS 通信库
```

---

## 机器人参数

| 参数 | 值 |
|------|-----|
| 机器人名称 | Legbot |
| 电机型号 | RobStride RS02（12 个） |
| 自由度 | 12（四足 × 髋/大腿/小腿） |
| 站立高度 | ~0.462 m |
| 基座质量 | 9.016 kg |
| 髋/大腿电机力矩 | ±17 N·m |
| 小腿电机力矩 | ±34 N·m |
| 小腿减速比 | 1:2（模型转电机时乘 2） |
| 板载电脑 | Orange Pi 6 |
| 传感器 | 自制串口 IMU（加速度+陀螺仪） |
| 通信总线 | USB-CAN × 2 + USB-串口（IMU） |

---

## 快速开始

### 1. 安装 Isaac Lab

严格按照 [Isaac Lab 官方指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 完成安装。

### 2. 安装本项目

```bash
# 克隆仓库
git clone https://github.com/Robot-Nav/legbot_lab.git
cd legbot_lab

# 激活 Isaac Lab 的 conda 环境
conda activate env_isaaclab

# 安装本项目为 editable 模式
pip install -e legbot_rl_lab/source/unitree_rl_lab
```

### 3. 开始训练

#### 3.1 主环境训练（4096 并行环境，PPO）

```bash
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --headless \
    --num_envs 4096 \
    --max_iterations 50000
```

**命令行参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 训练任务名称 | Unitree-Legbot-Velocity |
| `--headless` | 无渲染模式（加速训练） | False |
| `--num_envs` | 并行环境数量 | 4096 |
| `--max_iterations` | 最大训练迭代次数 | 50000 |
| `--seed` | 随机种子 | 随机 |
| `--resume` | 从断点恢复训练 | False |
| `--load_run` | 恢复时指定运行目录 | - |
| `--checkpoint` | 恢复时指定模型文件名 | - |

#### 3.2 恢复中断的训练

```bash
# 自动查找最新检查点恢复
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --resume \
    --headless

# 手动指定实验目录和检查点
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --resume \
    --load_run 2026-06-25_17-11-16 \
    --checkpoint model_108.pt \
    --headless
```

#### 3.3 NumPy 导航环境训练

```bash
python legbot/legbot_section001_np.py
```

### 4. 模型推理与导出

```bash
# 运行推理并自动导出 ONNX
python legbot_rl_lab/scripts/rsl_rl/play.py --task Unitree-Legbot-Velocity
```

推理时自动导出以下文件到 `logs/rsl_rl/unitree_legbot_velocity/<run_name>/exported/`：
- `policy.pt` — TorchScript 模型（Python 推理）
- `policy.onnx` — ONNX 模型（C++ 部署）
- `deploy.yaml` — 部署配置（关节映射、缩放因子等）

---

## Sim2Real 部署

### 系统架构

```
┌────────────────────┐  DDS (rt/lowcmd, rt/lowstate)  ┌──────────────────────┐  串口     ┌──────────────┐
│   legbot_ctrl       │◄──────────────────────────────►│  serial_dds_gateway  │◄────────►│ 12电机 + IMU │
│   (RL策略控制器)   │         CycloneDDS             │  (DDS↔串口协议网关)  │ type1-4  │ (凌足USB-CAN)│
└────────────────────┘                                └──────────────────────┘          └──────────────┘
         │                                                      │
   运行在香橙派上                                          运行在香橙派上
   1kHz 控制循环                                          500Hz 协议转换
   └──────────── 共享 lo 网卡 ────────────┘
```

**核心设计思想：控制器代码在仿真和实物上完全一致**，DDS 抽象层使得切换只需更换通信对端（MuJoCo 仿真器 或 serial_dds_gateway 硬件网关）。

### 编译与运行

#### 1. 编译网关（硬件桥接程序）

```bash
cd serial_dds_gateway
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)
```

#### 2. 编译控制器

```bash
cd legbot_rl_lab/deploy/robots/legbot
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)
```

#### 3. 运行（两终端模式）

**终端 1 — 启动网关（必须先启动）：**

```bash
cd serial_dds_gateway
./build/dds_to_serial_gateway \
    --serial-port-a /dev/myttyCAN0 \
    --serial-port-b /dev/myttyCAN1 \
    --imu-port /dev/myttyIMU \
    --network lo \
    --tick-hz 500 \
    --joint-bias-load-file config/joint_prone_bias.fatu.txt \
    --send-disable-on-exit
```

**终端 2 — 启动控制器（网关就绪后）：**

```bash
cd legbot_rl_lab/deploy/robots/legbot
./build/legbot_ctrl --network lo
```

#### 4. FSM 状态切换（手柄操作）

```
Passive（阻尼模式）──LT+A──► FixStand（站姿）──Start──► Velocity（RL接管）
      ▲                        ▲                              │
      │                        │                              │
      └──────── LT+B ──────────┴────────── LT+B ─────────────┘
```

---

## 奖励函数设计

训练中的总奖励为以下各项的加权和：

### 正奖励（跟踪目标）

| 奖励项 | 权重 | 公式 | 说明 |
|--------|------|------|------|
| 线速度跟踪 | +1.0 | $\exp(-\|v_{xy} - v_{cmd}\|^2 / 0.5)$ | 指数核，σ=0.5 |
| 角速度跟踪 | +0.5 | $\exp(-\|\omega_z - \omega_{cmd}\|^2 / 0.5)$ | 指数核，σ=0.5 |

### 负奖励（惩罚项）

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| 垂直线速度 | -2.0 | 防止上下晃动 |
| 水平角速度 | -0.05 | 抑制 roll/pitch |
| 基座高度偏差 | -1.0~-10.0 | 课程学习，保持 0.28m 目标高度 |
| 关节加速度 | -1e-7 | 鼓励平滑运动 |
| 关节功率 | -2e-5 | 降低能耗 |
| 关节力矩 | -1e-4 | 避免过大扭矩 |
| 动作变化率 | -0.01 | 鼓励平滑控制 |
| 不期望接触 | -1.0 | 大腿/小腿不应触地 |
| 关节限位 | -2.0 | 避免超出关节范围 |
| 足部调节 | -0.05 | 足部高度和间距约束 |
| Hip 位置 | -0.05 | 静止时 Hip 保持 0 附近 |
| 关节位置 | -0.01 | 静止时保持默认姿态 |

### 课程学习

| 课程项 | 起始值 | 终值 | 迭代区间 |
|--------|--------|------|----------|
| 垂直速度惩罚权重 | -2.0 | 0.0 | 0 → 1500 |
| 高度惩罚权重 | -1.0 | -10.0 | 0 → 5000 |
| 地形难度 | Level 0 | Level 5 | 随进度递增 |

---

## 训练参数

### PPO 超参数

| 参数 | 值 |
|------|-----|
| 算法 | PPO (Proximal Policy Optimization) |
| 学习率 | $1.0 \times 10^{-3}$（自适应） |
| 裁剪参数 $\epsilon$ | 0.2 |
| 折扣因子 $\gamma$ | 0.99 |
| GAE 参数 $\lambda$ | 0.95 |
| 熵系数 | 0.01 |
| 价值损失系数 | 1.0 |
| 最大梯度范数 | 1.0 |
| 每环境步数 | 24 |
| 学习轮数 | 5 |
| Mini-batch 数量 | 4 |
| 目标 KL 散度 | 0.01 |
| 最大迭代次数 | 50,000 |

### 训练规模

| 参数 | 值 |
|------|-----|
| 并行环境数 | 4,096 |
| 物理步长 | 0.005 s (200 Hz) |
| 解耦系数 | 4（控制 = 50 Hz） |
| Episode 长度 | 25 s（1250 控制步） |
| 每次迭代样本数 | 4096 × 24 = 98,304 |
| 总训练步数上限 | ~4.9 × 10⁹ |

### 网络架构

| 参数 | 值 |
|------|-----|
| 隐藏层维度 | [512, 256, 128] |
| 激活函数 | ELU |
| 策略观测维度 | 45（含 10 帧历史） |
| 策略输出维度 | 12（关节位置偏移） |
| Critic 观测维度 | ~263（含高度扫描 187 维） |

---

## MDP 定义

### 观测空间（策略网络）

| 观测项 | 维度 | 缩放 | 噪声 |
|--------|------|------|------|
| 基座角速度（IMU 陀螺仪） | 3 | 0.25 | U(-0.2, 0.2) |
| 投影重力方向 | 3 | 1.0 | U(-0.05, 0.05) |
| 速度命令 | 3 | 1.0 | - |
| 关节位置（相对默认值） | 12 | 1.0 | U(-0.03, 0.03) |
| 关节速度 | 12 | 0.05 | U(-2.0, 2.0) |
| 上一动作 | 12 | 1.0 | - |
| **总计** | **45** | | |

策略网络还包含 **10 帧历史观测**（`history_length=10`），用于时序建模。

### 行动空间

| 参数 | 值 |
|------|-----|
| 控制方式 | 关节位置控制 (PD) |
| 动作范围 | $[-1, 1]^{12}$ |
| 动作缩放 | ×0.25 rad |
| PD 刚度 $K_p$ | 60 N·m/rad |
| PD 阻尼 $K_d$ | 4.0 N·m·s/rad |

### Critic 特权信息

Critic 网络可观测到策略无法获取的信息：

| 特权信息 | 维度 | 说明 |
|----------|------|------|
| 基座线速度 | 3 | 真实速度值 |
| 关节加速度 | 12 | 加速度值 |
| 关节力矩 | 12 | 真实力矩 |
| 足部接触力 | 4 | 四足法向力 |
| 高度扫描（大范围） | 187 | 1.6×1.0m 地形高度图 |

### 终止条件

| 条件 | 阈值 |
|------|------|
| 超时 | Episode 达到 25 秒 |
| 摔倒 | 基座接触力 > 1.0 N |

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 仿真平台 | NVIDIA Isaac Sim 5.0.0 / Isaac Lab 2.2.0 |
| 物理引擎 | PhysX GPU (4096 并行) |
| RL 算法 | RSL-RL 2.3.1（PPO） |
| 深度学习 | PyTorch（CUDA 加速） |
| 模型导出 | ONNX Runtime 1.22.0 |
| 部署语言 | C++17 |
| 通信中间件 | CycloneDDS |
| 机器人模型 | URDF / MJCF |
| 配置管理 | Hydra / YAML |
| 仿真验证 | MuJoCo + DDS 桥接 |
| 板载电脑 | Orange Pi 6 (aarch64) |
| 电机 | RobStride RS02 |

---

## 安全保护机制

### 命令侧限幅（不影响运行）

| 保护项 | 限值 | 说明 |
|--------|------|------|
| Action Clip | ±100 | 策略原始输出裁剪 |
| 关节角度限位 | 可配置 | 硬限位保护 |
| 力矩限幅 | ±40 Nm | 力矩裁剪 |
| 角度变化限幅 | 0.05 rad/tick | 防神经网络突跳 |
| 速度变化限幅 | 1.0 rad/s/tick | 防速度突变 |

### 反馈侧超限保护（切到 Passive 阻尼模式）

| 监测项 | 阈值 | 动作 |
|--------|------|------|
| 通信超时 | - | → Passive |
| 关节速度 | 30 rad/s | → Passive |
| 反馈力矩 | 45 Nm | → Passive |
| 电机温度 | 80°C | → Passive |
| IMU Roll | ±0.5 rad (~28°) | → Passive |
| IMU Pitch | ±0.5 rad (~28°) | → Passive |
| 急停标志 | - | → Passive |

---

## 项目团队

本项目由 **Robot-Nav** 团队开发与维护。

GitHub: [https://github.com/Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab)

---

## 许可证

MIT License

---

## 引用

```
@misc{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
  year={2017},
  eprint={1707.06347},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/1707.06347},
}

@misc{rudin2022advanced,
  title={Advanced Skills by Learning Locomotion and Local Navigation End-to-End},
  author={Nikhil Rudin and David Hoeller and Marko Bjelonic and Marco Hutter},
  year={2022},
  eprint={2209.12827},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2209.12827},
}
```
