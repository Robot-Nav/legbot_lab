# LegBot RL RobotLab

<div align="center">

**MoE-CTS：基于混合专家的并发师生强化学习——用于四足机器人运动控制**

[English README](README.md) | [技术文档 (Technical Doc)](TECHNICAL_DOC_zh.md)

</div>

---

## 项目概述

**LegBot RL RobotLab** 是基于 NVIDIA IsaacLab 框架的四足机器人（LegBot）强化学习训练与部署项目。本项目实现了 **MoE-CTS（Mixture of Experts - Concurrent Teacher-Student）** 算法，是 [CTS 论文](https://arxiv.org/pdf/2405.10830) 的 IsaacLab 实现，同时也是 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) 的复现与改进版本。

LegBot 是一个自定义四足机器人，具有与 Unitree Go2 相同的 12 关节运动结构（4 条腿 × 3 个关节：髋关节、大腿、小腿），可直接复用已有的 MDP 模块。

<p align="center">
  <b>IsaacLab 训练 → MuJoCo Sim2Sim 验证 → 真实机器人部署</b>
</p>

<p align="center">
  <img src="resources/results/isaaclab_scene.png" width="70%"/>
</p>

---

## 项目意义

足式机器人运动控制面临三大挑战：

1. **感知不完整**：真实部署时机器人无法获取完整的状态信息（如精确线速度、地形高程图等），但训练时若不利用这些特权信息会导致性能下降。
2. **Sim2Real 差距**：仿真环境与真实世界存在动力学差异（电机特性、摩擦、质量分布等）。
3. **地形泛化**：需要在楼梯、斜坡、不平地面等多种地形上稳健行走。

本项目的贡献：

- **并发师生架构（CTS）**：训练时同时运行教师网络（使用特权信息）和学生网络（仅用可部署信息），学生通过蒸馏学习教师的隐空间表示，部署时仅使用学生网络。
- **混合专家（MoE）**：学生编码器采用 MoE 结构，8 个专家网络通过门控机制动态组合，提升对复杂地形和运动模式的建模能力。
- **真实电机模型**：使用 Unitree 官方电机扭矩-转速曲线模型，而非简单 PD 控制器，缩小 Sim2Real 差距。
- **RoboGauge 基准测试**：在 150k 训练步内达到 0.6828 分，超越 CTS 原始版本、HIM、DreamWaQ 等方法。

---

## 算法原理：MoE-CTS

### 核心思想

MoE-CTS 是 **PPO** 与 **并发师生蒸馏** 的结合，并在学生编码器中引入 **混合专家（MoE）** 机制。

**关键点**：
- 训练时将并行环境按 `teacher_env_ratio=0.75` 划分为两组：
  - **教师环境（75%）**：使用教师编码器，输入特权观测（critic obs：含线速度、高度扫描、关节力矩等）
  - **学生环境（25%）**：使用学生 MoE 编码器，输入仅可部署观测（actor obs：角速度、重力方向、关节状态、速度指令）
- 两组环境共享同一个 Actor 和 Critic，仅编码器不同
- 学生编码器通过 **隐空间蒸馏损失** 模仿教师编码器的输出
- 部署时仅使用学生分支（无需特权信息）

### 算法流程

```
每个训练迭代：

1. Rollout 阶段（num_steps_per_env=24 步）：
   ├─ 教师环境：obs → TeacherEncoder → latent → Actor → action
   └─ 学生环境：obs → StudentMoEEncoder → latent → Actor → action
   环境执行 action，存储 transition

2. 计算 GAE 回报：
   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
   A_t = δ_t + γ·λ·A_{t+1}
   R_t = A_t + V(s_t)

3. PPO 更新（5 epochs × 4 mini-batches）：
   ├─ 策略损失：L_policy = -E[min(r·A, clip(r)·A)]
   ├─ 价值损失：L_value = E[(V - R)²]
   └─ 熵奖励：L_entropy = -β·H[π]

4. 学生编码器蒸馏更新：
   ├─ 隐空间损失：L_latent = ||StudentEncoder(o_a) - TeacherEncoder(o_c).detach()||²
   ├─ 负载均衡损失：L_balance = ||mean(gates) - 1/N||²
   └─ L_student = L_latent + α·L_balance, α=0.01
```

### 网络结构

#### 教师编码器

$$z_t = \text{L2Norm}(\text{MLP}_{\text{teacher}}(o_c))$$

- 输入：critic 观测（特权信息）
- 结构：MLP[512, 256] → L2Norm
- 输出维度：latent_dim = 32

#### 学生 MoE 编码器

$$g = \text{Softmax}(\text{MLP}_{\text{gate}}(o_a))$$
$$e_i = \text{Expert}_i(\text{Backbone}(o_a)), \quad i=1,\dots,N$$
$$z_s = \text{L2Norm}\left(\sum_{i=1}^{N} g_i \cdot e_i\right)$$

- 专家数量：N = 8
- 共享主干：MLP[512, 256, 256]
- 每个专家：Conv1d(groups=8) 分组卷积
- 门控网络：MLP[512, 256, 256] → Softmax
- 输出维度：latent_dim = 32

#### Actor（策略网络）

$$a \sim \mathcal{N}(\mu, \sigma^2), \quad \mu = \text{MLP}_{\text{actor}}([z, o_{\text{single}}])$$

- 输入：[latent(32), single_obs(45)] = 77 维
- 结构：MLP[512, 256, 128] → 12（动作维度）
- 标准差：可学习参数

#### Critic（价值网络）

$$V(s) = \text{MLP}_{\text{critic}}([z.\text{detach}(), o_c])$$

- 输入：[latent(32), critic_obs]
- 结构：MLP[512, 256, 128] → 1

### PPO 目标函数

$$L_{\text{PPO}} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right] + c_v \cdot L_{\text{value}} - c_e \cdot H[\pi]$$

其中：

$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, \quad \epsilon = 0.2$$

---

## 项目结构

```
legbot_lab/
├── scripts/rsl_rl/              # 训练与评估入口脚本
│   ├── train.py                 # 训练入口
│   ├── play.py                  # 评估 + 策略导出
│   ├── cli_args.py              # 命令行参数
│   └── rsl_rl_utils.py          # 日志与导出工具
├── source/
│   ├── robot_lab/               # 环境与任务定义
│   │   └── robot_lab/
│   │       ├── assets/          # 机器人资产与执行器配置
│   │       │   ├── legbot.py    # LegBot 机器人配置
│   │       │   └── unitree_actuator.py  # Unitree 电机模型
│   │       └── tasks/
│   │           ├── go2/         # Go2 MDP 模块（LegBot 复用）
│   │           │   ├── mdp/     # 奖励/观测/指令等
│   │           │   └── manager/ # 动作管理器
│   │           └── legbot/      # LegBot 专属配置
│   │               ├── env_cfg.py      # 环境配置
│   │               ├── rsl_rl_cfg.py   # 算法配置
│   │               └── env/legbot_env.py  # 环境类
│   └── rsl_rl/                  # 定制版 RSL-RL 算法库
│       └── rsl_rl/
│           ├── algorithms/moe_cts.py        # MoE-CTS 算法核心
│           ├── modules/actor_critic_moe_cts.py  # 师生网络模块
│           ├── networks/moe.py             # MoE 网络实现
│           ├── runners/on_policy_runner_cts.py  # CTS 训练循环
│           ├── storage/rollout_storage_cts.py   # CTS 数据存储
│           └── utils/exporter_cts.py       # 策略导出器
├── deploy/deploy_mujoco/        # MuJoCo Sim2Sim 部署
│   ├── deploy_legbot.py         # 部署脚本
│   ├── utils.py                 # 部署工具
│   └── configs/legbot.yaml      # 部署配置
├── resources/legbot/            # MuJoCo 场景与 URDF
│   ├── urdf/legbot.urdf         # LegBot URDF 模型
│   ├── meshes/                  # STL 网格文件
│   ├── flat.xml                 # 平坦地形
│   ├── stairs.xml               # 楼梯场景
│   ├── boxes.xml                # 箱子障碍物
│   └── legbot.xml               # LegBot MuJoCo 模型
├── src/                         # C++ 仿真与桥接
│   ├── legbot_bridge.h          # DDS 桥接（MuJoCo）
│   ├── param.h                  # 仿真配置
│   └── main.cc                  # MuJoCo 仿真的入口
├── TECHNICAL_DOC_zh.md          # 详细技术文档（中文）
├── README.md                    # 英文 README
└── README_cn.md                 # 本文件
```

---

## MDP 定义

### 观测空间

| 观测组 | 用途 | 历史帧 | 加噪 | 包含项 |
|--------|------|--------|------|--------|
| **policy**（actor obs） | 学生编码器输入 | 10 | 是 | base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action |
| **critic**（critic obs） | 教师/Critic 输入 | 1 | 否 | 上述全部 + base_lin_vel, joint_acc, joint_torque, contact_force, height_scan |
| **single_obs** | Actor 拼接输入 | 1 | 是 | 同 policy，仅当前帧 |

**关节顺序：**

```python
["FL_hip", "FL_thigh", "FL_calf",
 "FR_hip", "FR_thigh", "FR_calf",
 "RL_hip", "RL_thigh", "RL_calf",
 "RR_hip", "RR_thigh", "RR_calf"]
```

### 动作空间

- 类型：关节位置控制（`JointPositionAction`）
- 维度：12
- 缩放：0.25（动作 × 0.25 + 默认关节角 = 目标关节角）

### 指令空间

- 维度：3 — `[lin_vel_x, lin_vel_y, ang_vel_yaw]`
- 重采样间隔：5s
- 指令范围课程：在 20k 和 50k 迭代时扩展速度范围

### 终止条件

- 超时：25s / episode
- 非法接触：base 接触力 > 1.0N

---

## 奖励函数设计

### 跟踪奖励（正向）

**线速度跟踪**（指数核）：

$$r_{\text{lin}} = \exp\left(-\frac{\|v_{\text{cmd}}^{xy} - v_{\text{base}}^{xy}\|^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=1.0$$

**角速度跟踪**：

$$r_{\text{ang}} = \exp\left(-\frac{(\omega_{\text{cmd}}^{z} - \omega_{\text{base}}^{z})^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=0.5$$

### 惩罚项（负向）

| 奖励项 | 公式 | 权重 | 说明 |
|--------|------|------|------|
| 垂直速度 | $\|v_z\|^2$ | -2.0 → 0（课程） | 防止上下晃动 |
| Roll/Pitch 角速度 | $\|\omega_{xy}\|^2$ | -0.05 | 防止侧向旋转 |
| 关节加速度 | $\sum \ddot{q}_i^2$ | -1e-7 | 鼓励平滑运动 |
| 关节功率 | $\sum \|\dot{q}_i \cdot \tau_i\|$ | -2e-5 | 减少能耗 |
| 关节力矩 | $\sum \tau_i^2$ | -1e-4 | 避免过大扭矩 |
| 基座高度 | $(h - 0.28)^2$ | -1.0 → -10.0（课程） | 保持站立高度 |
| 动作变化率 | $\|a_t - a_{t-1}\|^2$ | -0.01 | 鼓励平滑动作 |
| 动作平滑性 | $\|a_t - 2a_{t-1} + a_{t-2}\|^2$ | -0.01 | 二阶平滑 |
| 不期望接触 | $\sum \mathbb{1}(\|F\| > 5)$ | -1.0 | 大腿/小腿触地 |
| 关节限位 | 关节超限惩罚 | -2.0 | 避免超出关节范围 |
| 足部滑动 | $\sum v_{\text{foot}}^{xy\,2} \cdot e^{-h_{\text{foot}}/\text{threshold}}$ | -0.05 | 减少脚部滑移 |
| 髋关节位置 | $\sum \|q_{\text{hip}} - q_{\text{default}}\|_1$ | -0.05 | 髋关节保持在默认位置 |

---

## 域随机化与 Unitree 电机模型

### 域随机化

| 随机化项 | 模式 | 范围 | 说明 |
|---------|------|------|------|
| 基座质量 | startup | ±1 kg | 适应负载变化 |
| 其他部件质量 | startup | ×[0.9, 1.1] | 质量分布变化 |
| 质心位置 | startup | ±0.05 m | 质心偏移 |
| 关节重置位置 | reset | ×[0.5, 1.5] | 初始姿态随机 |
| 执行器增益 (kp/kd) | reset | ×[0.9, 1.1] | PD 参数扰动 |
| 电机零偏 | reset | ±0.035 rad | 编码器零位误差 |
| 推力扰动 | interval (4s) | ±0.4 m/s, ±0.6 rad/s | 随机外力 |
| 摩擦系数 | startup | [0, 2.0] | 不同地面摩擦 |
| 基座初始状态 | reset | pos ±0.5m, yaw ±π | 随机初始位姿 |

### Unitree 电机模型

本项目使用 Unitree 官方扭矩-转速曲线模型（Go2 HV 参数）：

| 参数 | 值 | 说明 |
|------|-----|------|
| X1 | 13.5 rad/s | 满扭矩最大转速 |
| X2 | 30 rad/s | 空载转速 |
| Y1 | 20.2 N·m | 正向峰值扭矩 |
| Y2 | 23.4 N·m | 反向峰值扭矩 |

摩擦模型：

$$\tau_{\text{applied}} = \tau_{\text{PD}} - F_s \cdot \tanh\left(\frac{\dot{q}}{V_a}\right) - F_d \cdot \dot{q}$$

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_envs | 4096 | 并行环境数 |
| num_steps_per_env | 24 | 每次迭代步数 |
| max_iterations | 300000 | 最大迭代次数 |
| save_interval | 500 | 检查点保存间隔 |
| num_learning_epochs | 5 | PPO epoch 数 |
| num_mini_batches | 4 | Mini-batch 数 |
| learning_rate | 1e-3 | 自适应学习率 |
| student_encoder_lr | 1e-3 | 学生编码器学习率 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.95 | GAE λ |
| clip_param | 0.2 | PPO 裁剪范围 |
| entropy_coef | 0.01 | 熵系数 |
| value_loss_coef | 1.0 | 价值损失系数 |
| load_balance_coef | 0.01 | MoE 负载均衡系数 |
| teacher_env_ratio | 0.75 | 教师环境比例 |
| desired_kl | 0.01 | 目标 KL 散度 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| expert_num | 8 | MoE 专家数量 |
| latent_dim | 32 | 隐向量维度 |
| history_length | 10 | 观测历史长度 |
| sim_dt | 0.005 s | 物理步长 |
| control_decimation | 4 | 50Hz 控制频率 |
| episode_length_s | 25.0 | 最大回合时长 |

---

## 安装指南

### 1. 安装 IsaacLab

按照[官方指南](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/isaaclab_pip_installation.html)安装：

```bash
conda create -n legbot_lab python=3.11
conda activate legbot_lab
pip install --upgrade pip
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2. 安装定制版 RSL-RL 和 RobotLab

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

### 3. 安装 MuJoCo（可选，Sim2Sim 需要）

```bash
pip install mujoco pygame
```

---

## 训练与评估

```bash
# 训练
python scripts/rsl_rl/train.py --task=RobotLab-Legbot-v0 --headless

# 评估
python scripts/rsl_rl/play.py --task=RobotLab-Legbot-v0
```

### 结合 RoboGauge 评估

[RoboGauge](https://github.com/wty-yy/robogauge) 提供异步评估平台：

```bash
# 终端 1：启动 RoboGauge 服务
python robogauge/scripts/server.py --port 9973 --num-processes 32

# 终端 2：带评估的训练
python scripts/rsl_rl/train.py --task=RobotLab-Legbot-v0 --headless --robogauge --robogauge_port 9973
```

### 命令行参数

```bash
python scripts/rsl_rl/train.py \
    --task=RobotLab-Legbot-v0 \
    --headless \
    --num_envs <N> \
    --max_iterations <N> \
    --experiment_name <名称> \
    --run_name <名称> \
    --checkpoint <路径>
```

---

## MuJoCo Sim2Sim 部署

### 导出策略

运行 `play.py` 会同时导出 TorchScript（`.pt`）和 ONNX（`.onnx`）格式：

```bash
python scripts/rsl_rl/play.py \
    --task=RobotLab-Legbot-v0 \
    --checkpoint logs/rsl_rl/legbot_moe_cts/<run_name>/model_<iter>.pt
```

导出的策略**内部维护观测历史**，部署时只需输入当前帧观测。

### MuJoCo 中运行

编辑 `deploy/deploy_mujoco/configs/legbot.yaml` 中的策略路径：

```yaml
policy_path: "{ROOT_DIR}/logs/rsl_rl/legbot_moe_cts/<timestamp>/exported/policy.pt"
```

运行：

```bash
python deploy/deploy_mujoco/deploy_legbot.py
```

### 切换仿真场景

修改 `legbot.yaml` 中的 `xml_path`：

```yaml
# 平坦地形
xml_path: "{ROOT_DIR}/resources/legbot/flat.xml"
# 楼梯
xml_path: "{ROOT_DIR}/resources/legbot/stairs.xml"
# 箱子障碍
xml_path: "{ROOT_DIR}/resources/legbot/boxes.xml"
```

### 手柄控制

| 输入 | 功能 |
|------|------|
| LX | 前进/后退速度 |
| LY | 左移/右移速度 |
| RX | 角速度（转向） |

- 自动检测手柄连接，无手柄时使用配置文件的默认指令

---

## RoboGauge 基准测试结果

<p align="center">
  <img src="resources/results/robogauge_compare.png" width="100%"/>
</p>

### 150k 训练步数最优分数

| 模型 | 总分 | 跟踪 | 安全 | 质量 | 关卡 |
|------|------|------|------|------|------|
| **go2_moe_cts（本项目）** | **0.6828** | **0.6785** | 0.7552 | **0.7645** | **8.17** |
| go2_moe_cts (go2_rl_gym) | 0.6713 | 0.6669 | **0.7857** | 0.7392 | 7.85 |
| CTS（原始版本） | 0.5786 | 0.5755 | 0.7066 | 0.6624 | 6.83 |
| HIM | 0.5379 | 0.5453 | 0.6476 | 0.6050 | 6.19 |
| DreamWaQ | 0.5054 | 0.5105 | 0.6149 | 0.5730 | 5.74 |

---

## 与 go2_rl_gym 的区别

- **电机模型**：使用 Unitree 官方扭矩-转速曲线模型，替代简单 PD 控制器
- **奖励函数**：固定 sigma 跟踪奖励、降低 joint_acc_l2 权重、增加 joint_pos_penalty_l1
- **域随机化**：电机级延迟替代随机动作延迟
- **观测历史**：10 帧（对比 Gym 版 5 帧）
- **算法**：MoE-CTS，学生编码器含 8 个专家网络

---

## 致谢

本项目基于以下开源项目：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — NVIDIA Isaac Sim 统一机器人学习框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — 强化学习算法库
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — IsaacLab 机器人 RL 扩展
- [MuJoCo](https://github.com/google-deepmind/mujoco) — 高性能物理仿真器
- [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) — IsaacGym 版 Go2 RL 训练

相关论文：

- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)

---

## 许可证

本项目基于 Apache 2.0 许可证发布。详见各源文件的许可证声明。
