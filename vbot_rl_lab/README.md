# VBot RL Lab — Isaac Lab 训练与 Sim2Real 部署

> 本项目是基于 NVIDIA Isaac Lab 框架的 VBot 四足机器人强化学习训练与部署子仓库，
> 实现从仿真训练到实物部署的完整闭环（Sim2Real）。支持标准 PPO 和 MoE+CTS 两种训练范式。

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 项目结构](#2-项目结构)
- [3. VBot 机器人模型](#3-vbot-机器人模型)
- [4. 算法原理](#4-算法原理)
  - [4.1 PPO 基线](#41-ppo-基线)
  - [4.2 MoE + CTS 师生协同](#42-moe--cts-师生协同)
- [5. MDP 设计](#5-mdp-设计)
  - [5.1 观测空间](#51-观测空间)
  - [5.2 动作空间](#52-动作空间)
  - [5.3 奖励函数](#53-奖励函数)
  - [5.4 域随机化与课程学习](#54-域随机化与课程学习)
- [6. 安装与环境准备](#6-安装与环境准备)
- [7. 训练](#7-训练)
  - [7.1 标准 PPO](#71-标准-ppo)
  - [7.2 MoE+CTS](#72-moects)
  - [7.3 恢复训练](#73-恢复训练)
  - [7.4 命令行参数](#74-命令行参数)
- [8. 推理与模型导出](#8-推理与模型导出)
  - [8.1 标准 PPO](#81-标准-ppo)
  - [8.2 MoE+CTS](#82-moects)
- [9. Sim2Sim 仿真验证](#9-sim2sim-仿真验证)
- [10. Sim2Real 实机部署](#10-sim2real-实机部署)
  - [10.1 硬件连接与配置](#101-硬件连接与配置)
  - [10.2 编译](#102-编译)
  - [10.3 运行](#103-运行)
  - [10.4 安全与故障排查](#104-安全与故障排查)
- [11. 测试脚本](#11-测试脚本)
- [12. 技术栈](#12-技术栈)

---

## 1. 项目定位

`vbot_rl_lab` 是 VBot 四足机器人强化学习研究的**核心子仓库**，提供：

- 基于 Isaac Lab `ManagerBasedRLEnv` 的速度跟踪任务；
- 标准 PPO 与 MoE+CTS 两种强化学习训练流程；
- 训练完成后自动导出 ONNX / TorchScript 策略与 `deploy.yaml`；
- C++17 部署程序，支持 Sim2Sim（DDS + MuJoCo）与 Sim2Real（USB 转 CAN + IMU）两种模式。

> 仓库根目录 README 提供项目全景、快速开始与子项目导航，建议先阅读 [../README.md](../README.md)。

---

## 2. 项目结构

```
vbot_rl_lab/
├── source/unitree_rl_lab/          # Python 训练源码
│   └── unitree_rl_lab/
│       ├── assets/robots/          # VBot / Go2 机器人配置
│       │   ├── unitree.py          # UNITREE_VBOT_CFG（URDF 加载 + 执行器配置）
│       │   └── unitree_actuators.py# VBot 执行器 T-N 曲线模型
│       ├── tasks/locomotion/       # 速度跟踪运动任务
│       │   ├── agents/             # 算法配置
│       │   │   ├── rsl_rl_ppo_cfg.py      # 标准 PPO 配置
│       │   │   └── rsl_rl_moe_cts_cfg.py  # MoE+CTS 配置
│       │   ├── mdp/                # MDP 组件（观测/奖励/课程学习/命令）
│       │   └── robots/vbot/        # VBot 环境配置
│       │       ├── velocity_env_cfg.py          # 标准 PPO 环境配置
│       │       └── velocity_env_moe_cts_cfg.py  # MoE+CTS 环境配置
│       ├── rl/                     # 算法实现
│       │   ├── modules.py          # MoE 网络模块（MLP/L2Norm/SimNorm/Experts/MoE）
│       │   ├── actor_critic_moe_cts.py  # ActorCriticMoECTS（教师/学生双编码器）
│       │   ├── moe_cts.py          # MoECTS 算法（双优化器 + 知识蒸馏）
│       │   ├── rollout_storage_cts.py   # CTS 经验回放存储
│       │   └── on_policy_runner_cts.py  # CTS 训练运行器
│       └── utils/                  # 工具函数（部署配置导出）
│           ├── export_deploy_cfg.py
│           └── parser_cfg.py
├── scripts/rsl_rl/                 # 训练/推理脚本
│   ├── train.py                    # 标准 PPO 训练入口
│   ├── play.py                     # 标准 PPO 推理/导出入口
│   ├── train_moe_cts.py            # MoE+CTS 训练入口
│   ├── play_moe_cts.py             # MoE+CTS 推理/导出入口
│   ├── cli_args.py                 # 命令行参数
│   ├── test_motor.py               # 电机硬件测试
│   └── test_imu.py                 # IMU 硬件测试
├── deploy/                         # C++ 部署代码
│   └── robots/vbot/
│       ├── config/config.yaml      # FSM 状态机配置与 Sim2Real 硬件配置
│       ├── include/                # 部署类型定义（FSM、ONNX、接口抽象）
│       ├── src/                    # RL 状态逻辑
│       ├── tests/                  # 电机/IMU 协议单元测试
│       └── CMakeLists.txt          # 部署构建配置
├── unitree_ros/                    # 机器人模型文件
│   └── robots/vbot_description/
│       ├── urdf/vbot_description.urdf  # VBot URDF（STL mesh）
│       └── meshes/vbot.xml             # 原始 MJCF 定义
├── tests/                          # Python 测试
└── outputs/                        # 训练输出（Hydra 日志 & 配置）
```

---

## 3. VBot 机器人模型

VBot 是一款 12 自由度四足机器人，仿真模型位于 `unitree_ros/robots/vbot_description/`。

| 参数 | 值 |
|------|-----|
| 站立高度 | 0.462 m |
| 基座质量 | 9.016 kg |
| 自由度 | 12（4 条腿 × 3 关节） |
| 关节顺序 | FR_hip → FR_thigh → FR_calf → FL_hip → ... → RL_calf |
| 髋/大腿电机力矩 | ±17 N·m |
| 小腿电机力矩 | ±34 N·m |
| 默认站立姿态 | thigh=0.9 rad, calf=-1.8 rad, hip=0 rad |

执行器使用自定义 `UnitreeActuator`，在 `unitree_actuators.py` 中建模真实电机的 **T-N 曲线**与摩擦：

- `VbotHipThigh`：Y1=Y2=17 N·m, X1=15 rad/s, X2=30 rad/s
- `VbotCalf`：Y1=Y2=34 N·m, X1=10 rad/s, X2=20 rad/s

控制律为 PD + 前馈：

$$
\tau = \tau_{ff} + k_p(q_{target}-q) + k_d(\dot{q}_{target}-\dot{q}) - F_s\tanh(\dot{q}/V_a) - F_d\dot{q}
$$

---

## 4. 算法原理

### 4.1 PPO 基线

标准 PPO 基于 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 实现，配置见 `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`。

| 超参数 | 值 |
|--------|-----|
| Actor/Critic 隐藏层 | `[512, 256, 128]` |
| 激活函数 | ELU |
| 学习率 | 1e-3（自适应 KL） |
| clip | 0.2 |
| entropy 系数 | 0.01 |
| value loss 系数 | 1.0 |
| learning epochs | 5 |
| mini-batches | 4 |
| gamma | 0.99 |
| lambda (GAE) | 0.95 |
| desired KL | 0.01 |

定义重要性采样比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
$$

带剪裁的替代损失：

$$
\mathcal{L}^{\text{CLIP}}(\theta)= -\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\; \operatorname{clip}\left(r_t(\theta),1-\varepsilon,1+\varepsilon\right)A_t\right)\right]
$$

带剪裁的值函数损失：

$$
\mathcal{L}^{V}(\theta)=\mathbb{E}_t\left[\max\Bigl(\bigl(V_\theta(s_t)-R_t\bigr)^2,\; \bigl(V_\theta^{\text{clip}}(s_t)-R_t\bigr)^2\Bigr)\right]
$$

其中：

$$
V_\theta^{\text{clip}}(s)=V_{\text{old}}(s)+\operatorname{clip}\left(V_\theta(s)-V_{\text{old}}(s),-\varepsilon,\varepsilon\right)
$$

总损失：

$$
\mathcal{L}^{\text{PPO}}=\mathcal{L}^{\text{CLIP}}+c_1\mathcal{L}^{V}-c_2\mathcal{L}^{\text{entropy}}
$$

优势函数由 GAE 计算：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

### 4.2 MoE + CTS 师生协同

MoE+CTS 实现于 `rl/` 目录，配置见 `tasks/locomotion/agents/rsl_rl_moe_cts_cfg.py`。

| 超参数 | 值 |
|--------|-----|
| Actor/Critic 隐藏层 | `[512, 256, 128]` |
| 教师编码器 | `[512, 256] → latent_dim=32` |
| 学生 MoE 编码器 | `[512, 256, 256] → latent_dim=32` |
| 专家数量 | 8 |
| 历史长度 | 5 |
| 教师环境比例 | 0.75 |
| 学生编码器学习率 | 1e-3 |
| Load Balance 系数 | 0.01 |
| Norm 类型 | L2Norm（可选 SimNorm） |

#### 网络结构

```
Teacher:  privileged_obs  → MLP → z_T  → L2Norm/SimNorm
Student:  [o_{t-H+1},...,o_t] → MoE → z_S → L2Norm/SimNorm
Actor:    concat(z, o_t) → MLP → μ(a)
Critic:   concat(z, privileged_obs) → MLP → V(s)
```

#### Mixture of Experts

门控网络：

$$
g(x)=\operatorname{softmax}\left(W_g x\right),\qquad \sum_i g_i(x)=1
$$

专家输出加权：

$$
z_S=\sum_{i=1}^{N}g_i(h_t)\,E_i(h_t)
$$

负载均衡损失：

$$
\bar{u}_j=\frac{1}{B}\sum_{b=1}^{B}g_j(x_b),\qquad
\mathcal{L}_{\text{lb}}=\frac{1}{N}\sum_{j=1}^{N}\left(\bar{u}_j-\frac{1}{N}\right)^2
$$

#### 师生蒸馏

教师编码器在特权观测上训练，学生编码器在历史观测上训练，蒸馏损失为：

$$
\mathcal{L}_{\text{latent}}=\mathbb{E}\left[\left\|z_T-z_S\right\|^2\right]
$$

学生编码器总损失：

$$
\mathcal{L}_{\text{student}}=\mathcal{L}_{\text{latent}}+\beta\,\mathcal{L}_{\text{lb}}
$$

#### 训练流程

1. 将 `num_envs` 个环境按 `teacher_env_ratio` 划分为教师/学生环境；
2. 教师环境使用特权观测 $o^{\text{priv}}$ 计算 $z_T$；学生环境使用历史观测 $h_t$ 计算 $z_S$；
3. Actor/Critic 在两类样本上共同优化 PPO 损失；
4. 学生编码器使用独立优化器，最小化 $\mathcal{L}_{\text{student}}$；
5. 部署时只保留学生 MoE 编码器 + Actor，ONNX 导出显式维护 `history` 状态。

---

## 5. MDP 设计

### 5.1 观测空间

#### Policy 观测（标准 PPO / MoE+CTS 共用，45 维）

| 项 | 维度 | 说明 |
|----|------|------|
| base_ang_vel | 3 | 基座角速度（本体坐标系） |
| projected_gravity | 3 | 重力向量在本体坐标系的投影 |
| velocity_commands | 3 | 目标线速度 x/y 与角速度 z |
| joint_pos_rel | 12 | 相对默认位置的关节角度 |
| joint_vel_rel | 12 | 关节速度 |
| last_action | 12 | 上一步动作 |

#### Critic 特权观测

| 项 | 维度 | 说明 |
|----|------|------|
| base_lin_vel | 3 | 基座线速度（本体坐标系） |
| joint_effort | 12 | 关节力矩 |
| height_measurements | 187 | 高度扫描（1.6 m × 1.0 m，分辨率 0.1 m） |
| foot_contact_forces | 4 | 足部接触力范数（MoE+CTS 配置） |
| joint_acc | 12 | 关节加速度（MoE+CTS 配置） |

MoE+CTS 中，学生编码器输入为最近 `history_length=5` 帧 Policy 观测，拼接后为 $5 \times 45 = 225$ 维。

### 5.2 动作空间

动作维度 12，对应 12 个关节的目标位置偏移，通过 `JointPositionAction` 映射：

$$
q_{target}=q_{default} + \alpha \cdot a
$$

其中动作缩放 `scale=0.25`。

### 5.3 奖励函数

奖励函数定义于 `tasks/locomotion/mdp/rewards.py` 与环境配置中，核心项包括：

| 奖励项 | 权重（PPO / MoE+CTS） | 说明 |
|--------|------------------------|------|
| track_lin_vel_xy | 1.5 / 1.0 | 线速度跟踪 |
| track_ang_vel_z | 0.75 / 0.5 | 偏航角速度跟踪 |
| base_linear_velocity (lin_vel_z) | -2.0 | 抑制上下跳动 |
| base_angular_velocity (ang_vel_xy) | -0.05 | 抑制横滚/俯仰角速度 |
| joint_vel | -0.001 | 关节速度惩罚 |
| joint_acc | -2.5e-7 | 关节加速度惩罚 |
| joint_torques | -2e-4 / -2.5e-5 | 力矩惩罚（已按 VBot 刚度校准） |
| action_rate | -0.1 / -0.01 | 动作变化率惩罚 |
| energy | -2e-5 | 能耗惩罚 |
| feet_air_time | 0.1 / 1.0 | 腾空时间奖励 |
| air_time_variance | -1.0 | 四足腾空时间方差惩罚 |
| feet_slide | -0.1 | 足部滑动惩罚 |
| undesired_contacts | -1 | 非足部接触惩罚 |
| feet_height_body | -0.5 | 抬脚高度奖励 |
| hip_to_default | -0.05 | 髋关节回归默认位置 |
| feet_regulation | -0.05 | 低高度时抑制水平足速 |
| action_smoothness | -0.01 | 二阶动作平滑惩罚 |

### 5.4 域随机化与课程学习

#### 域随机化（EventCfg）

- 刚体摩擦/恢复系数
- 基座附加质量
- PD 增益缩放
- 基座质心偏移
- 连杆质量缩放
- 关节零位偏移
- 电机力矩缩放
- 动作延迟 0~20 ms
- 随机外部推力

#### 课程学习

- `terrain_levels_vel`：根据速度跟踪表现逐步提升地形难度；
- `lin_vel_cmd_levels`：逐步扩大线速度命令范围。

---

## 6. 安装与环境准备

1. 按 [Isaac Lab 官方指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
2. 激活 conda 环境并安装本包：

```bash
conda activate env_isaaclab
pip install -e source/unitree_rl_lab
```

3. VBot 的 URDF 模型已内置于 `unitree_ros/robots/vbot_description/urdf/vbot_description.urdf`，依赖同目录 `meshes/` 下的 STL 文件。

---

## 7. 训练

### 7.1 标准 PPO

```bash
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --headless
```

### 7.2 MoE+CTS

```bash
python scripts/rsl_rl/train_moe_cts.py --task Unitree-Vbot-Velocity-MoECTS --headless
```

### 7.3 恢复训练

```bash
# 自动查找最新检查点
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --resume --headless

# 手动指定
python scripts/rsl_rl/train.py \
  --task Unitree-Vbot-Velocity \
  --resume \
  --load_run 2026-05-15_10-00-00 \
  --checkpoint model_108.pt \
  --headless
```

MoE+CTS 恢复训练同理：

```bash
python scripts/rsl_rl/train_moe_cts.py --task Unitree-Vbot-Velocity-MoECTS --resume --headless
```

### 7.4 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_envs` | 并行环境数 | 4096 |
| `--max_iterations` | 最大训练迭代 | 50000 |
| `--seed` | 随机种子 | 随机 |
| `--resume` | 恢复训练 | 无 |
| `--load_run` | 指定实验目录 | 最新 |
| `--checkpoint` | 指定检查点 | 最新 |
| `--distributed` | 多 GPU 分布式训练 | False |
| `--video` | 训练过程录制视频 | False |

训练日志与检查点保存在 `logs/rsl_rl/unitree_vbot_velocity/<timestamp>/` 或 `unitree_vbot_velocity_moects/<timestamp>/`。

---

## 8. 推理与模型导出

### 8.1 标准 PPO

```bash
python scripts/rsl_rl/play.py --task Unitree-Vbot-Velocity
```

推理时会自动导出 `policy.onnx` 与 `policy.pt`。

### 8.2 MoE+CTS

```bash
python scripts/rsl_rl/play_moe_cts.py --task Unitree-Vbot-Velocity-MoECTS
```

MoE+CTS 导出包括：

- `exported/policy.onnx`：ONNX Runtime 模型，输入 `obs` + `history`，输出 `actions` + `new_history`；
- `exported/policy.pt`：TorchScript 模型，内部维护 history buffer；
- `params/deploy.yaml`：PD 增益、默认关节角、观测缩放、命令范围等。

---

## 9. Sim2Sim 仿真验证

### 9.1 编译

```bash
# MuJoCo 仿真器
cd ../simulate
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# VBot 控制器
cd ../../vbot_rl_lab/deploy/robots/vbot
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 9.2 运行

需要同时开启两个终端：

```bash
# 终端 1：启动 MuJoCo 仿真器
cd ../simulate/build
./unitree_mujoco

# 终端 2：启动 VBot 控制器（默认 sim2sim 模式）
cd vbot_rl_lab/deploy/robots/vbot/build
./vbot_ctrl --mode sim2sim --network lo
```

### 9.3 操作流程

| 操作 | 状态切换 |
|------|----------|
| `Q` (LT) + `Space` (A) | Passive → FixStand |
| `Enter` (Start) | FixStand → Velocity（启动 RL） |
| `W/S/A/D` | 前进/后退/平移 |
| `←/→` | 转向 |
| `Q` (LT) + `Shift` (B) | Velocity → Passive |

> MuJoCo 仿真器不加载策略模型，所有推理由 `vbot_ctrl` 中的 ONNX Runtime 完成。高度扫描通过 UDP `127.0.0.1:19876` 从仿真器发送到控制器。

---

## 10. Sim2Real 实机部署

### 10.1 硬件连接与配置

Sim2Real 模式通过 USB 转 CAN 串口协议驱动 RobStride 电机，并通过串口读取 IMU。
硬件配置位于 `deploy/robots/vbot/config/config.yaml` 的 `sim2real` 字段：

| 项目 | 默认配置 |
|------|----------|
| 电机串口 1 | `/dev/ttyUSB0`，6 个电机 |
| 电机串口 2 | `/dev/ttyUSB1`，6 个电机 |
| IMU 串口 | `/dev/ttyUSB2` |
| 波特率 | 921600 |
| CAN 主站 ID | `0xFF` |
| 执行器类型 | `2`（ROBSTRIDE_02） |

**请根据实际硬件修改 `motor_ids` 与 `zero_offsets`**，使其与机器人关节顺序一一对应。

### 10.2 编译

```bash
cd deploy/robots/vbot
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 10.3 运行

```bash
cd deploy/robots/vbot/build
./vbot_ctrl --mode sim2real
```

控制器启动后默认进入 Passive 状态：

1. 长按 `LT + A` 进入 FixStand 站立状态；
2. 按 `Start` 进入 Velocity 状态，启动 RL 策略；
3. 按 `LT + B` 回到 Passive 状态。

> Sim2Real 模式下暂无 DDS 手柄输入，可通过键盘或未来扩展的远程控制接口发送命令。

### 10.4 安全与故障排查

- 首次部署前务必使用 `scripts/test_motor.py` 与 `scripts/test_imu.py` 验证硬件通信；
- 机器人应悬空或固定，避免策略异常导致摔倒；
- 若电机抖动，降低 `config.yaml` 中的 `kp/kd`；
- 若 IMU 无数据，检查串口设备名与波特率；
- 发生异常时立即按 `LT + B` 或发送失能指令切换到 Passive。

---

## 11. 测试脚本

### 电机测试

```bash
# 扫描串口上的电机
python scripts/test_motor.py --port /dev/ttyUSB0 --scan

# 测试单个电机（索引 0，默认 motor_id=0x11）
python scripts/test_motor.py --port /dev/ttyUSB0 --motor 0
```

### IMU 测试

```bash
# 协议验证（无需硬件）
python scripts/test_imu.py --validate-only

# 持续读取 IMU 数据
python scripts/test_imu.py --port /dev/ttyUSB2 --count 500
```

### C++ 协议单元测试

```bash
./scripts/build_and_run_tests.sh --cpp-only
```

---

## 12. 技术栈

| 类别 | 技术 |
|------|------|
| 仿真平台 | NVIDIA Isaac Sim 5.0.0 / Isaac Lab 2.2.0 |
| RL 算法 | RSL-RL 2.3.1（PPO）+ MoE+CTS 扩展 |
| 深度学习 | PyTorch（CUDA 加速） |
| 模型导出 | ONNX Runtime、TorchScript |
| 部署语言 | C++17 |
| 机器人模型 | URDF + STL meshes |
| 网络架构 | MoE + MLP + L2Norm/SimNorm |
| 配置管理 | Hydra / YAML |
| 通信协议 | DDS（Sim2Sim）、USB 转 CAN 串口协议（Sim2Real） |

---

> 更多项目级介绍、算法公式与快速导航请返回阅读 [../README.md](../README.md)。
