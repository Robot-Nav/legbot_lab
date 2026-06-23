# VBot Lab/MuJoCo Sim2Real 四足机器人强化学习平台

> 基于 NVIDIA Isaac Lab / Isaac Sim 与 MuJoCo 的 VBot 四足机器人 **Sim2Real** 研究与部署平台。
> 本项目覆盖从大规模并行仿真训练、策略推理与模型导出，到 MuJoCo 仿真验证、再到 RobStride 实机部署的完整闭环。
> 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Isaac Lab 2.2.0](https://img.shields.io/badge/Isaac%20Lab-2.2.0-green.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![C++17](https://img.shields.io/badge/C++-17-orange.svg)](https://isocpp.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.3-blueviolet.svg)](https://mujoco.org/)
[![Robot-Nav](https://img.shields.io/badge/Robot--Nav-Integration-brightgreen.svg)](https://blog.csdn.net/qq_56908984/article/details/161678983?spm=1001.2014.3001.5501)

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 仓库结构](#2-仓库结构)
- [3. 核心特性](#3-核心特性)
- [4. 算法原理](#4-算法原理)
  - [4.1 PPO（标准基线）](#41-ppo标准基线)
  - [4.2 MoE + CTS 师生协同](#42-moe--cts-师生协同)
- [5. 快速开始](#5-快速开始)
  - [5.1 环境准备](#51-环境准备)
  - [5.2 训练 VBot 速度跟踪策略](#52-训练-vbot-速度跟踪策略)
  - [5.3 推理与导出](#53-推理与导出)
  - [5.4 Sim2Sim 仿真验证](#54-sim2sim-仿真验证)
  - [5.5 Sim2Real 实机部署](#55-sim2real-实机部署)
- [6. 子项目导航](#6-子项目导航)
- [7. 技术栈](#7-技术栈)
- [8. 引用与许可](#8-引用与许可)

---

演示：

https://github.com/user-attachments/assets/229b4f40-bd35-47e3-a2bd-81c5a3710f71

https://github.com/user-attachments/assets/86f6ec8c-f933-436e-ba22-2dc1b9986bbd

---

## 1. 项目简介

VBot 是一款面向研究与教育的四足机器人平台，具备 12 个主动关节（3×4 腿），全身采用 RobStride（灵足）系列电机驱动。
本项目旨在为 VBot 等四足机器人 提供一套开源、可复现的 **端到端强化学习 Sim2Real 工具链**：

- **仿真训练**：在 NVIDIA Isaac Lab 中运行数千个并行环境，训练速度跟踪策略。
- **两种算法范式**：
  - 标准 **PPO**（RSL-RL），作为高性能基线；
  - **MoE + CTS**（Mixture of Experts + Concurrent Teacher-Student），在策略端使用历史观测的 MoE 学生编码器，在训练端使用特权信息的教师编码器进行蒸馏。
- **模型导出**：训练结束后自动导出 `deploy.yaml`、`policy.onnx`（ONNX Runtime）与 `policy.pt`（TorchScript）。
- **Sim2Sim 验证**：通过自定义 MuJoCo 仿真器与控制器进行 DDS 通信，验证策略在物理仿真中的稳定性。
- **Sim2Real 部署**：C++ 控制器通过 USB 转 CAN 串口协议直接驱动 RobStride 电机，并读取外部 IMU，实现实机闭环控制。

> 详细的训练参数、MDP 设计、奖励函数与部署说明请见 [`vbot_rl_lab/README.md`](./vbot_rl_lab/README.md)。

---

## 2. 仓库结构

```
vbot_mujoco (moe+cts)/
├── vbot_rl_lab/                    # Isaac Lab 训练 + C++ Sim2Real 部署
│   ├── source/unitree_rl_lab/      # Python 训练源码
│   │   ├── assets/robots/          # VBot / Go2 机器人配置
│   │   ├── tasks/locomotion/       # 速度跟踪任务（MDP、奖励、课程学习）
│   │   ├── rl/                     # MoE+CTS 算法实现
│   │   └── utils/                  # 部署配置导出
│   ├── scripts/rsl_rl/             # train.py / play.py / train_moe_cts.py / play_moe_cts.py
│   ├── deploy/robots/vbot/         # C++ 控制器（FSM + ONNX Runtime + RobStride 驱动）
│   ├── deploy/robots/go2/          # Go2 控制器示例
│   ├── unitree_ros/                # URDF / MJCF / ROS 包
│   └── README.md                   # 详细训练与部署文档
│
├── simulate/                       # MuJoCo + DDS Sim2Sim 仿真器
│   ├── src/                        # 主程序、DDS 桥、手柄、PNG 编码
│   ├── CMakeLists.txt
│   └── config.yaml
│
├── vbot/                           # 基于 motrix_envs 的 VBot 导航环境（NumPy/MuJoCo）
│   ├── cfg.py
│   ├── vbot_section001_np.py
│   └── xmls/                       # MuJoCo 场景与 VBot 模型
│
├── unitree_robots/                 # Go2 机器人 MJCF/OBJ 资源
├── terrain_tool/                   # MuJoCo 地形生成工具
└── README.md                       # 本文件
```

---

## 3. 核心特性

| 特性 | 说明 |
|------|------|
| **大规模并行训练** | 默认 4096 环境，单 GPU 即可训练；支持多卡分布式。 |
| **标准 PPO 基线** | RSL-RL 2.3.1，Actor/Critic `[512,256,128]`，自适应 KL 学习率。 |
| **MoE+CTS 高级范式** | 教师编码器使用特权观测，学生 MoE 编码器使用历史观测；部署时只运行学生网络。 |
| **丰富的域随机化** | 摩擦、基座质量、连杆质量、COM、关节零偏、电机力矩、PD 增益、动作延迟、外部推力。 |
| **课程学习** | 地形难度与速度命令范围随训练逐步增加。 |
| **Sim2Sim 闭环** | MuJoCo 仿真器通过 DDS `LowCmd/LowState` 与控制器 1 kHz 通信。 |
| **Sim2Real 闭环** | USB 转 CAN 串口协议（18 B 定长帧）驱动 RobStride 电机；32 B 定长帧读取外部 IMU。 |
| **一键导出部署配置** | 训练脚本自动输出 `deploy.yaml`、ONNX、TorchScript。 |

---

## 4. 算法原理

### 4.1 PPO（标准基线）

项目中的标准 PPO 实现基于 [RSL-RL](https://github.com/leggedrobotics/rsl_rl)。
定义重要性采样比：

$$r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$$

带剪裁的替代损失：

$$
\mathcal{L}^{\text{CLIP}}(\theta)=
-\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,\;
\text{clip}\left(r_t(\theta),1-\varepsilon,1+\varepsilon\right)A_t
\right)
\right]
$$

其中优势函数 $A_t$ 由 GAE 计算：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

带剪裁的值函数损失：

$$
V_\theta^{\text{clip}}(s)=V_{\text{old}}(s)+\text{clip}\left(V_\theta(s)-V_{\text{old}}(s),-\varepsilon,\varepsilon\right)
$$

$$
\mathcal{L}^{V}(\theta)=\mathbb{E}_t\left[
\max\Bigl(
\bigl(V_\theta(s_t)-R_t\bigr)^2,\;
\bigl(V_\theta^{\text{clip}}(s_t)-R_t\bigr)^2
\Bigr)
\right]
$$

总损失：

$$
\mathcal{L}^{\text{PPO}}=
\mathcal{L}^{\text{CLIP}}+
c_1\mathcal{L}^{V}-
c_2\mathcal{L}^{\text{entropy}}
$$

当 KL 散度超过阈值时，学习率会自适应调整：

- $\overline{\mathrm{KL}}>2\,\mathrm{KL}_{\text{target}}$：$\alpha \leftarrow \alpha/1.5$
- $\overline{\mathrm{KL}}<\mathrm{KL}_{\text{target}}/2$：$\alpha \leftarrow \alpha\times1.5$

### 4.2 MoE + CTS 师生协同

MoE+CTS 参考 [Concurrent Teacher-Student (CTS)](https://arxiv.org/abs/2405.10830) 与 [Switch Transformers](https://arxiv.org/abs/2101.03961) 设计，
用于在部署时仅用**历史观测**推理，而在训练时利用**特权观测**监督。

#### 网络结构

- **教师编码器** $f_T$：输入 Critic/特权观测 $o^{\text{priv}}$，输出潜向量 $z_T\in\mathbb{R}^{d_z}$。
- **学生 MoE 编码器** $f_S^{\text{MoE}}$：输入历史观测序列 $h_t=[o_{t-H+1},\dots,o_t]$，输出潜向量 $z_S$。
- **策略网络** $\mu_\theta$：输入 $\text{concat}(z, o_t)$，输出关节目标位置均值。
- **价值网络** $V_\phi$：输入 $\text{concat}(z, o^{\text{priv}})$，输出状态价值。

#### Mixture of Experts

门控网络输出专家权重：

$$
g(x)=\text{softmax}\left(W_g x\right),\qquad
g_i(x)\in[0,1],\; \sum_i g_i=1
$$

专家输出加权求和：

$$
z_S = \sum_{i=1}^{N} g_i(h_t)\,E_i(h_t)
$$

为鼓励专家负载均衡，引入 **Load Balance Loss**：

$$
\bar{u}_j=\frac{1}{B}\sum_{b=1}^{B}g_j(x_b),\qquad
\mathcal{L}_{\text{lb}}=\frac{1}{N}\sum_{j=1}^{N}\left(\bar{u}_j-\frac{1}{N}\right)^2
$$

#### 师生蒸馏

学生编码器通过最小化与教师潜向量的 L2 距离进行学习：

$$
\mathcal{L}_{\text{latent}}=\mathbb{E}\left[\left\|z_T-z_S\right\|^2\right]
$$

学生编码器的总损失为：

$$
\mathcal{L}_{\text{student}}=\mathcal{L}_{\text{latent}}+\beta\,\mathcal{L}_{\text{lb}}
$$

#### 并发训练

训练时将并行环境按固定比例划分：

- 教师环境（默认 75%）：使用特权观测 $o^{\text{priv}}$ 生成 $z_T$；
- 学生环境（默认 25%）：使用历史观测 $h_t$ 生成 $z_S$。

PPO 策略/价值损失在两类样本上共同优化，学生编码器则通过独立的 Adam 优化器进行蒸馏优化。

部署时仅保留学生 MoE 编码器 + 策略网络，ONNX 导出显式维护 `history` 状态，保证 Sim2Real 推理与仿真一致。

---

## 5. 快速开始

### 5.1 环境准备

1. 安装 NVIDIA Isaac Lab（参考[官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)）。
2. 激活 Isaac Lab 的 conda 环境：

```bash
conda activate env_isaaclab   # 或你的 Isaac Lab 环境名
```

3. 安装本项目为 editable 包：

```bash
cd vbot_rl_lab
pip install -e source/unitree_rl_lab
```

### 5.2 训练 VBot 速度跟踪策略

#### 标准 PPO

```bash
cd vbot_rl_lab
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --headless
```

#### MoE + CTS

```bash
cd vbot_rl_lab
python scripts/rsl_rl/train_moe_cts.py --task Unitree-Vbot-Velocity-MoECTS --headless
```

常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_envs` | 并行环境数 | 4096 |
| `--max_iterations` | 最大训练迭代 | 50000 |
| `--seed` | 随机种子 | 随机 |
| `--resume` | 从最新检查点恢复 | 无 |

训练日志保存在 `vbot_rl_lab/logs/rsl_rl/unitree_vbot_velocity/<timestamp>/`。

### 5.3 推理与导出

#### 标准 PPO

```bash
cd vbot_rl_lab
python scripts/rsl_rl/play.py --task Unitree-Vbot-Velocity
```

#### MoE + CTS

```bash
cd vbot_rl_lab
python scripts/rsl_rl/play_moe_cts.py --task Unitree-Vbot-Velocity-MoECTS
```

推理结束后会在检查点目录生成：

- `exported/policy.onnx`：ONNX Runtime 部署模型
- `exported/policy.pt`：TorchScript 模型
- `params/deploy.yaml`：PD 增益、默认关节角、观测缩放等部署配置

### 5.4 Sim2Sim 仿真验证

需要编译并同时运行 MuJoCo 仿真器和控制器。

```bash
# 1. 编译 MuJoCo 仿真器
cd simulate
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 2. 编译 VBot 控制器
cd ../../vbot_rl_lab/deploy/robots/vbot
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

运行（两个终端）：

```bash
# 终端 1：启动 MuJoCo 仿真器
cd simulate/build
./unitree_mujoco

# 终端 2：启动控制器（Sim2Sim 模式）
cd vbot_rl_lab/deploy/robots/vbot/build
./vbot_ctrl --mode sim2sim --network lo
```

默认键盘映射（仿真器窗口需处于焦点）：

| 按键 | 状态切换 |
|------|----------|
| `Q` (LT) + `Space` (A) | Passive → FixStand |
| `Enter` (Start) | FixStand → Velocity（启动 RL） |
| `Q` (LT) + `Shift` (B) | Velocity → Passive |

> 仿真器通过 DDS `rt/lowcmd`、`rt/lowstate` 与控制器通信；高度扫描通过 UDP `127.0.0.1:19876` 传输。

### 5.5 Sim2Real 实机部署

1. 确认电机串口（默认 `/dev/ttyUSB0`、`/dev/ttyUSB1`）与 IMU 串口（默认 `/dev/ttyUSB2`）存在，并调整 [`deploy/robots/vbot/config/config.yaml`](./vbot_rl_lab/deploy/robots/vbot/config/config.yaml) 中的 `motor_ids`、`zero_offsets` 与实际硬件一致。
2. 编译控制器（同上）。
3. 启动实机模式：

```bash
cd vbot_rl_lab/deploy/robots/vbot/build
./vbot_ctrl --mode sim2real
```

> **安全提示**：首次实机测试前，建议先运行 `python scripts/test_motor.py` 与 `python scripts/test_imu.py` 验证电机与 IMU 通信。机器人应悬空或固定，避免突然运动造成损坏。

---

## 6. 子项目导航

| 子目录 | 内容 |
|--------|------|
| [`vbot_rl_lab/`](./vbot_rl_lab) | 强化学习训练、MoE+CTS 算法、C++ 部署控制器、Sim2Real 配置与测试。 |
| [`simulate/`](./simulate) | MuJoCo 物理仿真器，提供与实机一致的 DDS 通信接口。 |
| [`vbot/`](./vbot) | 基于 `motrix_envs` 的导航任务训练环境。 |
| [`terrain_tool/`](./terrain_tool) | 参数化地形生成工具。 |

---

## 7. 技术栈

| 类别 | 技术 |
|------|------|
| 仿真平台 | NVIDIA Isaac Sim 5.0.0 / Isaac Lab 2.2.0 |
| 物理引擎 | PhysX（Isaac Sim）、MuJoCo（Sim2Sim） |
| RL 算法 | RSL-RL 2.3.1（PPO）+ MoE+CTS 扩展 |
| 深度学习 | PyTorch（CUDA） |
| 模型导出 | ONNX Runtime、TorchScript |
| 部署语言 | C++17 |
| 机器人模型 | URDF + STL meshes |
| 网络架构 | MLP / MoE / L2Norm / SimNorm |
| 配置管理 | Hydra / YAML |

---

## 8. 引用与许可

如果你在研究中使用了本项目，请引用相关基础工作：

- PPO: Schulman et al. "Proximal Policy Optimization Algorithms", 2017.
- GAE: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2015.
- MoE: Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017.
- Switch Transformers: Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", 2021.
- CTS: "Concurrent Teacher-Student Reinforcement Learning for Legged Robots", arXiv:2405.10830.

本项目基于 Isaac Lab / RSL-RL 的开源框架构建，相关源文件保留原始 BSD-3-Clause 许可证声明。

---

## 9. 致谢

感谢开源项目 https://github.com/wty-yy/go2_rl_gym，为本项目提供了算法思路以及引导。

```bibtex
@inproceedings{wu2026robogauge,
    title={Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion},
    author={Tianyang Wu and Hanwei Guo and Yuhang Wang and Junshu Yang and Xinyang Sui and Jiayi Xie and Xingyu Chen and Zeyang Liu and Xuguang Lan},
    booktitle={Proceedings of Robotics: Science and Systems},
    year={2026}
}
```
