# LegBot RL RobotLab 技术文档

> 本文档详细说明 LegBot RL RobotLab 项目的算法原理、数学公式、项目含义、代码结构与运行步骤。

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目意义](#2-项目意义)
3. [整体架构](#3-整体架构)
4. [算法原理：MoE-CTS](#4-算法原理moe-cts)
5. [网络结构与公式](#5-网络结构与公式)
6. [环境与 MDP 定义](#6-环境与-mdp-定义)
7. [奖励函数设计](#7-奖励函数设计)
8. [域随机化与电机模型](#8-域随机化与电机模型)
9. [课程学习](#9-课程学习)
10. [训练流程](#10-训练流程)
11. [部署流程（Sim2Sim）](#11-部署流程sim2sim)
12. [配置参数详解](#12-配置参数详解)
13. [关键文件索引](#13-关键文件索引)

---

## 1. 项目概述

LegBot RL RobotLab 是一个基于 NVIDIA IsaacLab 框架的四足机器人（LegBot）强化学习训练与部署项目。它实现了 **MoE-CTS（Mixture of Experts - Concurrent Teacher-Student）** 算法，是 [CTS 论文](https://arxiv.org/pdf/2405.10830) 的 IsaacLab 实现，同时也是 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym)（基于 IsaacGym）的复现与改进版本。

LegBot 是一个自定义四足机器人，具有与 Unitree Go2 相同的12关节运动结构，便于复用已有的MDP模块。

核心闭环：
```
IsaacLab 训练  →  导出策略 (TorchScript/ONNX)  →  MuJoCo Sim2Sim 验证  →  真实 LegBot 部署
```

---

## 2. 项目意义

### 2.1 解决的问题
足式机器人运动控制面临三大挑战：
1. **感知不完整**：真实部署时机器人无法获取完整的状态信息（如精确的线速度、地形高程图等），但训练时若不利用这些特权信息会导致性能下降。
2. **Sim2Real 差距**：仿真环境与真实世界存在动力学差异（电机特性、摩擦、质量分布等）。
3. **地形泛化**：需要在楼梯、斜坡、不平地面等多种地形上稳健行走。

### 2.2 本项目的贡献
- **并发师生架构（CTS）**：训练时同时运行教师网络（使用特权信息）和学生网络（仅用可部署信息），学生通过蒸馏学习教师的隐空间表示，部署时仅使用学生网络。
- **混合专家（MoE）**：学生编码器采用 MoE 结构，8 个专家网络通过门控机制动态组合，提升对复杂地形和运动模式的建模能力。
- **真实电机模型**：使用 Unitree 官方电机扭矩-转速曲线模型，而非简单 PD 控制器，缩小 Sim2Real 差距。
- **RoboGauge 基准测试**：在 150k 训练步内达到 0.6828 分，超越 CTS 原始版本、HIM、DreamWaQ 等方法。

---

## 3. 整体架构

### 3.1 代码目录结构

```
go2_rl_robotlab/
├── scripts/                          # 训练与评估入口脚本
│   └── rsl_rl/
│       ├── train.py                  # 训练入口
│       ├── play.py                   # 评估入口，同时导出策略
│       ├── cli_args.py               # 命令行参数定义
│       └── utils.py                  # 日志与策略导出工具
├── source/
│   ├── robot_lab/                    # 环境与任务定义
│   │   └── robot_lab/
│   │       ├── assets/               # 机器人资产与执行器配置
│   │       │   ├── unitree.py        # Go2 机器人配置
│   │       │   └── unitree_actuator.py  # Unitree 电机模型
│   │       └── tasks/go2/            # Go2 任务定义
│   │           ├── env/go2_env.py    # 环境类
│   │           ├── env_cfg.py        # 环境配置（场景/观测/奖励/事件）
│   │           ├── rsl_rl_cfg.py     # RL 算法配置
│   │           ├── manager/action_manager.py  # 动作管理器
│   │           └── mdp/              # MDP 组件
│   │               ├── commands.py   # 指令生成器
│   │               ├── observations.py  # 观测函数
│   │               ├── rewards.py    # 奖励函数
│   │               ├── events.py     # 域随机化事件
│   │               ├── curriculums.py  # 课程学习
│   │               └── terrains.py   # 地形生成器
│   └── rsl_rl/                       # 定制版 RSL-RL 算法库
│       └── rsl_rl/
│           ├── algorithms/
│           │   └── moe_cts.py        # MoE-CTS 算法核心
│           ├── modules/
│           │   └── actor_critic_moe_cts.py  # 师生网络模块
│           ├── networks/
│           │   └── moe.py            # MoE 网络实现
│           ├── runners/
│           │   └── on_policy_runner_cts.py  # CTS 训练循环
│           ├── storage/
│           │   └── rollout_storage_cts.py   # CTS 数据存储
│           └── utils/
│               └── exporter_cts.py   # 策略导出器
├── deploy/                           # 部署相关
│   └── deploy_mujoco/
│       ├── deploy_go2.py             # MuJoCo Sim2Sim 部署
│       ├── utils.py                  # 部署工具函数
│       └── configs/go2.yaml          # 部署配置
└── resources/                        # 机器人模型与场景资源
    └── go2/
        ├── urdf/go2.urdf             # Go2 URDF 模型
        ├── flat.xml / stairs.xml ... # MuJoCo 场景
        └── assets/                   # 网格与纹理资源
```

### 3.2 模块依赖关系

```
train.py / play.py
    │
    ├── robot_lab.tasks.go2.__init__  (注册 gym 环境 RobotLab-Go2-v0)
    │       │
    │       ├── Go2EnvCfg  (env_cfg.py) ── 场景/观测/奖励/事件/课程配置
    │       ├── Go2Env     (go2_env.py) ── 重写 ActionManager
    │       └── MoECTSRunnerCfg (rsl_rl_cfg.py) ── 算法超参
    │
    └── rsl_rl.runners.OnPolicyRunnerCTS
            │
            ├── MoECTS 算法 (algorithms/moe_cts.py)
            │       │
            │       ├── ActorCriticMoECTS 策略 (modules/actor_critic_moe_cts.py)
            │       │       ├── TeacherEncoder (MLP + L2Norm)
            │       │       ├── StudentMoEEncoder (MoE + L2Norm)
            │       │       ├── Actor MLP
            │       │       └── Critic MLP
            │       └── RolloutStorageCTS (storage/rollout_storage_cts.py)
            │
            └── networks/moe.py  (MoE 与 Experts 实现)
```

---

## 4. 算法原理：MoE-CTS

### 4.1 核心思想

MoE-CTS 是 **PPO** 与 **并发师生蒸馏** 的结合，并在学生编码器中引入 **混合专家（MoE）** 机制。

**关键点**：
- 训练时将所有并行环境按 `teacher_env_ratio=0.75` 划分为两组：
  - **教师环境（75%）**：使用教师编码器，输入特权观测（critic obs，包含线速度、高度扫描等）
  - **学生环境（25%）**：使用学生 MoE 编码器，输入仅可部署观测（actor obs，无特权信息）
- 两组环境共享同一个 Actor 和 Critic，仅编码器不同
- 学生编码器通过 **隐空间蒸馏损失** 模仿教师编码器的输出
- 部署时仅使用学生分支（无需特权信息）

### 4.2 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│  每个训练迭代：                                               │
│                                                              │
│  1. Rollout 阶段（num_steps_per_env=24 步）：                │
│     ├─ 教师环境：obs → TeacherEncoder → latent → Actor → action
│     └─ 学生环境：obs → StudentMoEEncoder → latent → Actor → action
│     环境执行 action，存储 transition                          │
│                                                              │
│  2. 计算回报（GAE）：                                         │
│     δ_t = r_t + γ·V(s_{t+1}) - V(s_t)                       │
│     A_t = δ_t + γ·λ·A_{t+1}                                  │
│     R_t = A_t + V(s_t)                                       │
│                                                              │
│  3. PPO 更新（5 epochs × 4 mini-batches）：                  │
│     ├─ 策略损失：L_policy = -E[min(r·A, clip(r)·A)]          │
│     ├─ 价值损失：L_value = E[(V - R)²]                       │
│     └─ 熵奖励：L_entropy = -β·H[π]                           │
│                                                              │
│  4. 学生编码器蒸馏更新：                                       │
│     ├─ 隐空间损失：L_latent = ||f_student(x) - f_teacher(x)||²
│     ├─ 负载均衡损失：L_balance = ||mean(gates) - 1/N||²      │
│     └─ L_student = L_latent + α·L_balance                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 与原始 CTS 的区别

| 特性 | 原始 CTS | MoE-CTS（本项目） |
|------|---------|------------------|
| 学生编码器 | 单一 MLP | 混合专家（8 个专家） |
| 编码器归一化 | - | L2Norm / SimNorm |
| 负载均衡 | 无 | 负载均衡损失 |
| 激活函数 | ELU | ELU / CatELU（可选） |

---

## 5. 网络结构与公式

### 5.1 教师编码器（Teacher Encoder）

教师编码器接收**特权观测** $o_c$（critic obs），输出隐向量 $z_t$：

$$
z_t = \text{L2Norm}(\text{MLP}_{\text{teacher}}(o_c))
$$

- 输入：critic 观测（base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, joint_acc, joint_torque, contact_force, height_scan）
- 网络结构：`MLP[512, 256]` → `L2Norm`
- 输出维度：`latent_dim = 32`

```python
# source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py
self.teacher_encoder = nn.Sequential(
    MLP(mlp_input_dim_t, latent_dim, teacher_encoder_hidden_dims, activation),
    L2Norm()  # 或 SimNorm
)
```

### 5.2 学生 MoE 编码器（Student MoE Encoder）

学生编码器接收**可部署观测** $o_a$（actor obs，无特权信息），通过 MoE 输出隐向量 $z_s$：

$$
\begin{aligned}
g &= \text{Softmax}(\text{MLP}_{\text{gate}}(o_a)) \\
e_i &= \text{Expert}_i(\text{Backbone}(o_a)), \quad i=1,\dots,N \\
z_s &= \text{L2Norm}\left(\sum_{i=1}^{N} g_i \cdot e_i\right)
\end{aligned}
$$

- 专家数量：`expert_num = 8`
- 网络结构：
  - 共享主干：`MLP[512, 256, 256]`
  - 每个专家：`Conv1d(groups=8)` 分组卷积，输出 `output_dim=32`
  - 门控网络：`MLP[512, 256, 256]` → `Softmax`
- 输出维度：`latent_dim = 32`

```python
# source/rsl_rl/rsl_rl/networks/moe.py
class MoE(nn.Module):
    def forward(self, x):
        weights = self.gating_network(x)          # (B, expert_num)
        expert_outs = self.experts(x)             # (B, expert_num, output_dim)
        output = torch.sum(weights.unsqueeze(-1) * expert_outs, dim=1)
        return output, weights
```

**专家结构（Experts）**：采用共享主干 + 分组卷积的高效实现，而非独立的 N 个 MLP：
```python
class Experts(nn.Module):
    # backbone: MLP(input → expert_num * expert_hidden_dim)
    # experts: Conv1d(in=expert_num*hidden, out=expert_num*output, kernel=1, groups=expert_num)
```

### 5.3 Actor（策略网络）

Actor 接收隐向量与单帧观测的拼接，输出动作均值：

$$
a \sim \mathcal{N}(\mu, \sigma^2), \quad \mu = \text{MLP}_{\text{actor}}([z, o_{\text{single}}])
$$

- 输入：`[latent(32), single_obs(45)]` = 77 维
- 网络结构：`MLP[512, 256, 128]` → 12（动作维度）
- 动作标准差：可学习参数 `std`（标量或对数参数化）

### 5.4 Critic（价值网络）

Critic 接收隐向量与特权观测的拼接，输出状态价值：

$$
V(s) = \text{MLP}_{\text{critic}}([z.\text{detach}(), o_c])
$$

- 输入：`[latent(32), critic_obs]`
- 网络结构：`MLP[512, 256, 128]` → 1

### 5.5 L2Norm 与 SimNorm

隐向量归一化用于稳定蒸馏训练：

$$
\text{L2Norm}(x) = \frac{x}{\|x\|_2}
$$

$$
\text{SimNorm}(x) = \text{Softmax}(x_{\text{reshape}[-1, 8]}) \quad \text{(Simplicial Normalization)}
$$

### 5.6 CatELU 激活函数（可选）

受 [Concat ReLU](https://arxiv.org/pdf/2303.07507) 启发，CatELU 将特征维度翻倍：

$$
\text{CatELU}(x) = [\text{ELU}(x), \text{ELU}(-x)]
$$

输出维度为输入的 2 倍，保留正负信息。

---

## 6. 环境与 MDP 定义

### 6.1 场景配置

- **机器人**：Unitree Go2（URDF 模型，12 个关节：4 腿 × 3 关节 [hip, thigh, calf]）
- **地形**：程序化生成的多类型地形（平坦、楼梯、斜坡、障碍物、间隙、台阶等）
- **并行环境数**：默认 8192
- **仿真参数**：
  - 物理步长 `dt = 0.005s`
  - 控制频率 `decimation = 4`（即 50Hz 控制频率，`step_dt = 0.02s`）
  - 单回合时长 `episode_length_s = 25s`

### 6.2 观测空间

观测分为三组，定义在 [env_cfg.py](source/robot_lab/robot_lab/tasks/go2/env_cfg.py)：

| 观测组 | 用途 | 历史长度 | 是否加噪 | 包含项 |
|--------|------|---------|---------|--------|
| **policy**（actor obs） | 学生编码器输入 | 10 | 是 | base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions |
| **critic**（critic obs） | 教师/Critic 输入 | 1 | 否 | 上述全部 + base_lin_vel, joint_acc, joint_torque, contact_force, height_scan |
| **single_obs** | Actor 拼接输入 | 1 | 是 | 同 policy 但仅当前帧 |

**关节顺序**（JOINT_NAMES）：
```python
["FL_hip", "FL_thigh", "FL_calf",
 "FR_hip", "FR_thigh", "FR_calf",
 "RL_hip", "RL_thigh", "RL_calf",
 "RR_hip", "RR_thigh", "RR_calf"]
```

### 6.3 动作空间

- **类型**：关节位置控制（`JointPositionActionCfg`）
- **维度**：12
- **缩放**：0.25（动作 × 0.25 + 默认关节角 = 目标关节角）
- **裁剪**：[-100, 100]

### 6.4 指令空间

指令由 `Go2RLGymCommand` 生成，维度为 3：`[lin_vel_x, lin_vel_y, ang_vel_yaw]`

- **重采样时间**：5s
- **动态重采样**：根据剩余距离与剩余回合时间调整速度下界，确保机器人在回合结束前接近目标
- **地形相关范围**：不同地形类型对应不同的速度上限（如平坦地形允许 ±2.0 m/s，楼梯限制 ±1.0 m/s）
- **指令课程**：在 20k 和 50k 迭代时扩展速度范围
- **特殊指令**：
  - 极限速度指令（limit_vel_prob=0.2）：从速度边界采样
  - 零指令（zero_command_curriculum）：训练初期为 0，逐步增加到 0.1 概率

### 6.5 终止条件

- **超时**（time_out）：回合达到最大时长
- **非法接触**（illegal_contact）：base 链接接触力超过阈值 1.0

---

## 7. 奖励函数设计

奖励函数定义在 [rewards.py](source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py)，采用多项加权求和。

### 7.1 跟踪奖励（正向）

**线速度跟踪**（指数核）：
$$
r_{\text{lin}} = \exp\left(-\frac{\|v_{\text{cmd}}^{xy} - v_{\text{base}}^{xy}\|^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=1.0
$$

**角速度跟踪**（指数核）：
$$
r_{\text{ang}} = \exp\left(-\frac{(\omega_{\text{cmd}}^{z} - \omega_{\text{base}}^{z})^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=0.5
$$

### 7.2 惩罚项（负向）

| 奖励项 | 公式 | 权重 | 说明 |
|--------|------|------|------|
| lin_vel_z_l2 | $\|v_z\|^2$ | -2.0 → 0（课程） | 惩罚垂直速度 |
| ang_vel_xy_l2 | $\|\omega_{xy}\|^2$ | -0.05 | 惩罚 roll/pitch 角速度 |
| joint_acc_l2 | $\sum \ddot{q}_i^2$ | -1e-7 | 关节加速度（Lab 物理步级，权重极小） |
| joint_power | $\sum \|\dot{q}_i \cdot \tau_i\|$ | -2e-5 | 关节功率 |
| joint_torques_l2 | $\sum \tau_i^2$ | -1e-4 | 关节力矩 |
| base_height_l2 | $(h - 0.38)^2$ | -1.0 → -10.0（课程） | 基座高度（使用高度扫描估计地面） |
| action_rate_l2 | $\|a_t - a_{t-1}\|^2$ | -0.01 | 动作变化率 |
| action_smoothness_l2 | $\|a_t - 2a_{t-1} + a_{t-2}\|^2$ | -0.01 | 动作平滑性（二阶差分） |
| undesired_contacts | $\sum \mathbb{1}(\|F\| > 5)$ | -1.0 | 大腿/小腿非法接触 |
| joint_pos_limits | 关节超限惩罚 | -2.0 | 关节位置限制 |
| feet_regulation | $\sum v_{\text{foot}}^{xy\,2} \cdot e^{-h_{\text{foot}}/(0.025 \cdot h_{\text{target}})}$ | -0.05 | 惩罚近地脚横向滑动 |
| hip_pos_penalty_l1 | $\sum \|q_{\text{hip}} - q_{\text{default}}\|_1$ | -0.05 | 髋关节偏离默认位置 |
| joint_pos_penalty_l1 | $\sum \|q_{\text{thigh,calf}} - q_{\text{default}}\|_1$ | -0.01 | 大腿/小腿偏离默认位置 |

### 7.3 基座高度估计

`base_height_l2` 与 `feet_regulation` 使用高度扫描器估计地面高度，而非直接使用世界坐标 z：

```python
base_height = base_z - mean(ray_hits_z)  # 减去估计的地面高度
```

对于无效射线（NaN/Inf），回退到 `base_z - base_height_target`。

---

## 8. 域随机化与电机模型

### 8.1 域随机化（Events）

定义在 [env_cfg.py](source/robot_lab/robot_lab/tasks/go2/env_cfg.py) 的 `EventCfg`，用于缩小 Sim2Real 差距：

| 随机化项 | 模式 | 参数 | 说明 |
|---------|------|------|------|
| 刚体质量（base） | startup | add [-1, 1] kg | 基座质量扰动 |
| 刚体质量（其他） | startup | scale [0.9, 1.1] | 其他链接质量缩放 |
| 质心位置 | startup | ±0.03m | 基座质心偏移 |
| 关节重置 | reset | scale [0.5, 1.5] | 重置时关节位置随机 |
| 执行器增益 | reset | scale [0.9, 1.1] | PD 刚度/阻尼随机 |
| 电机零偏 | reset | ±0.035 rad | 电机零位偏移 |
| 推力扰动 | interval(4s) | ±0.4 m/s, ±0.6 rad/s | 随机推机器人 |
| 摩擦系数 | startup | [0, 2.0] | 静/动摩擦 |
| 基座重置 | reset | 位置±0.5m, 姿态±π | 随机初始位姿 |

### 8.2 Unitree 电机模型

定义在 [unitree_actuator.py](source/robot_lab/robot_lab/assets/unitree_actuator.py)，实现真实的扭矩-转速曲线：

```
    扭矩上限 (N·m)
        ^
Y2──────|
        |──────────Y1
        |          │\
        |          │ \
        |          │  \
        |          |   \
--------+----------|----> 速度 (rad/s)
                 X1   X2
```

**Go2 HV 参数**：
- `X1 = 13.5` rad/s：满扭矩时最大转速（T-N 曲线拐点）
- `X2 = 30` rad/s：空载转速
- `Y1 = 20.2` N·m：同向（扭矩与转速同向）峰值扭矩
- `Y2 = 23.4` N·m：反向（扭矩与转速反向）峰值扭矩

**摩擦模型**：
$$
\tau_{\text{applied}} = \tau_{\text{PD}} - F_s \cdot \tanh\left(\frac{\dot{q}}{V_a}\right) - F_d \cdot \dot{q}
$$

**扭矩裁剪**：
- 当 $|\dot{q}| < X1$：限制为 $Y1$（同向）或 $Y2$（反向）
- 当 $|\dot{q}| \geq X1$：线性下降至 0（在 $X2$ 处）

**电机延迟**：`min_delay=0, max_delay=4` 步（电机级延迟，非动作级延迟）

### 8.3 动作管理器

`ActionManagerGo2` 额外维护 `_prev_prev_action` 用于二阶动作平滑性奖励计算：
```python
# source/robot_lab/robot_lab/tasks/go2/manager/action_manager.py
class ActionManagerGo2(ActionManager):
    def process_action(self, action):
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action
```

---

## 9. 课程学习

定义在 [curriculums.py](source/robot_lab/robot_lab/tasks/go2/mdp/curriculums.py) 与 [env_cfg.py](source/robot_lab/robot_lab/tasks/go2/env_cfg.py)：

### 9.1 地形课程（terrain_levels_vel_gym）

根据机器人移动距离动态调整地形难度：
- `move_up`：最大移动距离 > 地形长度/2 → 提升难度等级
- `move_down`：最大移动距离 < 目标距离×0.5 → 降低难度等级

### 9.2 奖励权重课程（gradual_reward_weight_modification）

线性插值修改奖励权重：
- `lin_vel_z_l2`：权重从 -2.0 线性衰减至 0.0（0→1500 迭代）
- `base_height_l2`：权重从 -1.0 线性增强至 -10.0（0→5000 迭代）

### 9.3 指令范围课程（command_range_curriculum）

在指定迭代步扩展速度指令范围：
```python
# 20000 迭代：lin_vel_x [-1,1], lin_vel_y [-1,1], ang_vel [-1.5,1.5]
# 50000 迭代：lin_vel_x [-2,2], lin_vel_y [-1,1], ang_vel [-2,2]
```

---

## 10. 训练流程

### 10.1 训练入口

```bash
python scripts/rsl_rl/train.py --task=RobotLab-Legbot-v0 --headless
```

### 10.2 训练循环（OnPolicyRunnerCTS）

定义在 [on_policy_runner_cts.py](source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py)：

```python
for it in range(max_iterations):
    # 1. Rollout：采集 num_steps_per_env=24 步
    for _ in range(num_steps_per_env):
        actions = alg.act(obs)              # 教师/学生分别推理
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)
    
    # 2. 计算 GAE 回报
    alg.compute_returns(obs)
    
    # 3. PPO + 蒸馏更新
    loss_dict = alg.update()
    
    # 4. 保存检查点（每 save_interval=500 步）
    if it % save_interval == 0:
        runner.save(f"model_{it}.pt")
```

### 10.3 MoECTS 算法核心

定义在 [moe_cts.py](source/rsl_rl/rsl_rl/algorithms/moe_cts.py)：

**环境划分**：
```python
teacher_env_ratio = 0.75
teacher_env_idxs = [i for i in range(num_envs) if i % 4 != 0]  # 75%
student_env_idxs = [i for i in range(num_envs) if i % 4 == 0]  # 25%
```

**动作推理（act）**：教师与学生环境分别推理，再重排回原始顺序。

**PPO 损失**：
$$
L_{\text{PPO}} = L_{\text{surrogate}} + c_v L_{\text{value}} - c_e H[\pi]
$$

其中：
$$
L_{\text{surrogate}} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$
$$
r_t = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{\text{old}}}(a_t|s_t))
$$

**学生蒸馏损失**：
$$
L_{\text{latent}} = \|\text{StudentEncoder}(o_a) - \text{TeacherEncoder}(o_c).\text{detach()}\|^2
$$

**负载均衡损失**（鼓励专家均匀使用）：
$$
L_{\text{balance}} = \left\|\frac{1}{B}\sum_{b=1}^{B} g^{(b)} - \frac{1}{N}\mathbf{1}\right\|^2
$$

**总学生损失**：
$$
L_{\text{student}} = L_{\text{latent}} + \alpha \cdot L_{\text{balance}}, \quad \alpha = 0.01
$$

**自适应学习率**：基于 KL 散度调整：
- 若 $KL > 2 \cdot \text{desired\_kl}$：`lr /= 1.5`
- 若 $KL < \text{desired\_kl}/2$：`lr *= 1.5`

### 10.4 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_envs | 8192 | 并行环境数 |
| num_steps_per_env | 24 | 每次迭代步数 |
| max_iterations | 300000 | 最大迭代数 |
| save_interval | 500 | 检查点保存间隔 |
| num_learning_epochs | 5 | PPO epoch 数 |
| num_mini_batches | 4 | mini-batch 数 |
| learning_rate | 1e-3 | 学习率（自适应） |
| student_encoder_learning_rate | 1e-3 | 学生编码器学习率 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.95 | GAE 参数 |
| clip_param | 0.2 | PPO 裁剪参数 |
| entropy_coef | 0.01 | 熵系数 |
| value_loss_coef | 1.0 | 价值损失系数 |
| load_balance_coef | 0.01 | 负载均衡系数 |
| teacher_env_ratio | 0.75 | 教师环境比例 |
| desired_kl | 0.01 | 目标 KL 散度 |
| max_grad_norm | 1.0 | 梯度裁剪 |

---

## 11. 部署流程（Sim2Sim）

### 11.1 策略导出

运行 `play.py` 时自动导出 TorchScript 与 ONNX 格式：

```bash
python scripts/rsl_rl/play.py --task=RobotLab-Legbot-v0 --checkpoint logs/rsl_rl/legbot_moe_cts/<run_name>/model_<iter>.pt
```

导出器 [exporter_cts.py](source/rsl_rl/rsl_rl/utils/exporter_cts.py) 将学生分支（StudentMoEEncoder + Actor）与归一化器封装为单输入模型：

```python
class _TorchPolicyExporter:
    def forward(self, single_obs):
        # 1. 更新内部观测历史（history_len=10）
        self._update_history(single_obs)
        # 2. 归一化
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(self.obs_history)
        # 3. 学生 MoE 编码
        latent, _ = self.student_moe_encoder(obs_a)
        # 4. Actor 输出动作
        return self.actor(torch.cat([latent, single_obs], dim=-1))
```

**关键特性**：导出的策略**内部维护观测历史**，部署时只需输入当前帧观测。

### 11.2 MuJoCo 部署

部署脚本 [deploy_legbot.py](deploy/deploy_mujoco/deploy_legbot.py)：

```bash
python deploy/deploy_mujoco/deploy_legbot.py
```

**部署配置**（[legbot.yaml](deploy/deploy_mujoco/configs/legbot.yaml)）：
- 仿真步长：`dt = 0.002s`
- 控制频率：`decimation = 10`（50Hz）
- PD 控制：`kps = [20.0, ...]`, `kds = [0.5, ...]`（12个关节）
- 默认关节角：`[0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]`

**部署循环**：
```python
while viewer.is_running():
    # 1. PD 控制计算扭矩
    data.ctrl[:] = pd_control(target_pos, qpos, kps, target_vel, qvel, kds)
    # 2. MuJoCo 物理步进
    mujoco.mj_step(model, data)
    # 3. 每 decimation 步查询策略
    if counter % decimation == 0:
        features = build_features(data, action, cmd, cfg)
        single_obs = build_single_obs(features, layout)
        action = policy(single_obs)
        target_pos = default_angles + action * 0.25
```

**观测构建**（与训练对齐）：
```python
features = {
    "ang_vel": qvel[3:6] * 0.25,
    "gravity": gravity_from_quat(qpos[3:7]),
    "cmd": cmd * [1.0, 1.0, 1.0],
    "joint_pos": (qpos[7:] - default_angles) * 1.0,
    "joint_vel": qvel[6:] * 0.05,
    "last_action": action,
}
```

### 11.3 场景切换

修改 `legbot.yaml` 中的 `xml_path`：
- `flat.xml`：平坦地形
- `stairs.xml`：楼梯
- `boxes.xml`：箱子地形
- `stairs_and_slope.xml`：楼梯与斜坡组合

### 11.4 控制器支持

- 自动检测手柄连接
- 手柄映射：`LX/LY` → 前向/侧向速度，`RX` → 角速度
- 无手柄时使用配置文件中的 `cmd_init: [1.0, 0.0, 0.0]`

---

## 12. 配置参数详解

### 12.1 环境配置（env_cfg.py）

```python
LegbotSceneCfg:
    num_envs: 4096
    env_spacing: 0.5
    terrain: 程序化地形（10 行 × 20 列，含楼梯/斜坡/障碍等）
    robot: LEGBOT_CFG（LegBot机器人配置）
    height_scanner: 1.6×1.0m 网格，分辨率 0.1m
    height_scanner_small: 0.4×0.3m 网格（用于基座高度估计）

LegbotEnvCfg:
    decimation: 4
    episode_length_s: 25.0
    sim.dt: 0.005
    BASE_HEIGHT_TARGET: 0.28  # LegBot站立高度
```

### 12.2 算法配置（rsl_rl_cfg.py）

```python
RslRlMoeCtsActorCriticCfg:
    class_name: "ActorCriticMoECTS"
    expert_num: 8                    # 专家数量
    latent_dim: 32                   # 隐向量维度
    norm_type: 'l2norm'              # 归一化类型
    teacher_encoder_hidden_dims: [512, 256]
    student_encoder_hidden_dims: [512, 256, 256]
    actor_hidden_dims: [512, 256, 128]
    critic_hidden_dims: [512, 256, 128]
    activation: "elu"

RslRlMoeCtsAlgorithmCfg:
    class_name: "MoECTS"
    load_balance_coef: 0.01
    teacher_env_ratio: 0.75
    student_encoder_learning_rate: 1e-3
    # ... PPO 标准参数
```

### 12.3 命令行参数

```bash
python scripts/rsl_rl/train.py \
    --task=RobotLab-Legbot-v0 \
    --headless \
    --num_envs <N> \
    --max_iterations <N> \
    --experiment_name <name> \
    --run_name <name> \
    --checkpoint <path>
```

---

## 13. 关键文件索引

| 功能 | 文件路径 |
|------|---------|
| 训练入口 | [scripts/rsl_rl/train.py](scripts/rsl_rl/train.py) |
| 评估与导出 | [scripts/rsl_rl/play.py](scripts/rsl_rl/play.py) |
| 环境配置 | [source/robot_lab/robot_lab/tasks/legbot/env_cfg.py](source/robot_lab/robot_lab/tasks/legbot/env_cfg.py) |
| 算法配置 | [source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py](source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py) |
| 环境类 | [source/robot_lab/robot_lab/tasks/legbot/env/legbot_env.py](source/robot_lab/robot_lab/tasks/legbot/env/legbot_env.py) |
| 任务注册 | [source/robot_lab/robot_lab/tasks/legbot/__init__.py](source/robot_lab/robot_lab/tasks/legbot/__init__.py) |
| MoE-CTS 算法 | [source/rsl_rl/rsl_rl/algorithms/moe_cts.py](source/rsl_rl/rsl_rl/algorithms/moe_cts.py) |
| 师生网络模块 | [source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py](source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py) |
| MoE 网络实现 | [source/rsl_rl/rsl_rl/networks/moe.py](source/rsl_rl/rsl_rl/networks/moe.py) |
| CTS 训练循环 | [source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py](source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py) |
| 策略导出器 | [source/rsl_rl/rsl_rl/utils/exporter_cts.py](source/rsl_rl/rsl_rl/utils/exporter_cts.py) |
| 奖励函数（复用Go2） | [source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py](source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py) |
| 观测函数（复用Go2） | [source/robot_lab/robot_lab/tasks/go2/mdp/observations.py](source/robot_lab/robot_lab/tasks/go2/mdp/observations.py) |
| 指令生成（复用Go2） | [source/robot_lab/robot_lab/tasks/go2/mdp/commands.py](source/robot_lab/robot_lab/tasks/go2/mdp/commands.py) |
| 域随机化（复用Go2） | [source/robot_lab/robot_lab/tasks/go2/mdp/events.py](source/robot_lab/robot_lab/tasks/go2/mdp/events.py) |
| 课程学习（复用Go2） | [source/robot_lab/robot_lab/tasks/go2/mdp/curriculums.py](source/robot_lab/robot_lab/tasks/go2/mdp/curriculums.py) |
| LegBot 机器人配置 | [source/robot_lab/robot_lab/assets/legbot.py](source/robot_lab/robot_lab/assets/legbot.py) |
| 动作管理器（复用Go2） | [source/robot_lab/robot_lab/tasks/go2/manager/action_manager.py](source/robot_lab/robot_lab/tasks/go2/manager/action_manager.py) |
| MuJoCo 部署 | [deploy/deploy_mujoco/deploy_legbot.py](deploy/deploy_mujoco/deploy_legbot.py) |
| 部署配置 | [deploy/deploy_mujoco/configs/legbot.yaml](deploy/deploy_mujoco/configs/legbot.yaml) |
| LegBot URDF 模型 | [legbot_description.urdf](legbot_description.urdf) |
| LegBot MuJoCo 模型 | [resources/legbot/legbot.xml](resources/legbot/legbot.xml) |

---

## 附录：与 go2_rl_gym 的主要差异

| 方面 | go2_rl_gym (IsaacGym) | go2_rl_robotlab (IsaacLab) |
|------|----------------------|---------------------------|
| 电机模型 | 简单 PD 控制器 | Unitree 官方扭矩-转速曲线模型 |
| 跟踪奖励 | 动态 sigma | 固定 sigma=0.5 |
| joint_acc_l2 权重 | 较大 | 极小（-1e-7），因 Lab 物理步级计算更精确 |
| joint_pos_penalty_l1 | 无 | 有（性能更优） |
| 动作延迟 | 随机动作延迟 | 电机级延迟（min_delay=0, max_delay=4） |
| 电机强度随机化 | 有 | 无（Lab 实现约束） |
| 历史长度 | 5 | 10（Lab 中更优） |
| 算法 | CTS | MoE-CTS（增加混合专家） |
