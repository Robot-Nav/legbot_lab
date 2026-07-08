# LegBot RL RobotLab

<div align="center">

**MoE-CTS：基于混合专家的并发师生强化学习——用于四足机器人运动控制**

[English README](README.md) | [技术文档 (Technical Doc)](TECHNICAL_DOC_zh.md)

</div>

---

## 项目概述

**LegBot RL RobotLab** 是基于 NVIDIA IsaacLab 框架的四足机器人（LegBot）强化学习训练与部署项目。本项目实现了 **MoE-CTS（Mixture of Experts - Concurrent Teacher-Student）** 算法，是 [CTS 论文](https://arxiv.org/pdf/2405.10830) 的 IsaacLab 实现，同时也是 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) 的复现与改进版本。

LegBot 是一个自定义四足机器人，具有与 Unitree Go2 相同的 12 关节运动结构（4 条腿 × 3 个关节：髋关节、大腿、小腿），可直接复用已有的 MDP 模块。

**核心闭环：**

<p align="center">
  <b>IsaacLab 训练 → MuJoCo Sim2Sim 验证 → 真实 LegBot 部署</b>
</p>



### 解决的问题与贡献

足式机器人运动控制面临三大挑战：

1. **感知不完整**：真实部署时机器人无法获取完整的状态信息（如精确线速度、地形高程图等），但训练时若不利用这些特权信息会导致性能下降。
2. **Sim2Real 差距**：仿真环境与真实世界存在动力学差异（电机特性、摩擦、质量分布等）。
3. **地形泛化**：需要在楼梯、斜坡、不平地面等多种地形上稳健行走。

本项目的贡献：

- **并发师生架构（CTS）**：训练时同时运行教师网络（使用特权信息）和学生网络（仅用可部署信息），学生通过蒸馏学习教师的隐空间表示，部署时仅使用学生网络。
- **混合专家（MoE）**：学生编码器采用 MoE 结构，8 个专家网络通过门控机制动态组合，提升对复杂地形和运动模式的建模能力。
- **真实电机模型**：使用 Unitree 官方电机扭矩-转速曲线模型，而非简单 PD 控制器，缩小 Sim2Real 差距。

---

## 算法原理：MoE-CTS

### 核心思想

MoE-CTS 是 **PPO** 与 **并发师生蒸馏** 的结合，并在学生编码器中引入 **混合专家（MoE）** 机制。

训练时将并行环境按 `teacher_env_ratio=0.75` 划分为两组：

- **教师环境（75%）**：使用教师编码器，输入特权观测（critic obs：含线速度、高度扫描、关节力矩等）
- **学生环境（25%）**：使用学生 MoE 编码器，输入仅可部署观测（actor obs：角速度、重力方向、关节状态、速度指令）

两组环境共享同一个 Actor 和 Critic，仅编码器不同。学生编码器通过**隐空间蒸馏损失**模仿教师编码器的输出。部署时仅使用学生分支（无需特权信息）。

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

### 与原始 CTS 的区别

| 特性 | 原始 CTS | MoE-CTS（本项目） |
|------|---------|------------------|
| 学生编码器 | 单一 MLP | 混合专家（8 个专家） |
| 编码器归一化 | - | L2Norm / SimNorm |
| 负载均衡 | 无 | 负载均衡损失 |
| 激活函数 | ELU | ELU / CatELU（可选） |

### 网络结构

#### 教师编码器

$$z_t = \text{L2Norm}(\text{MLP}_{\text{teacher}}(o_c))$$

- 输入：critic 观测（特权信息）
- 结构：MLP[512, 256] → L2Norm
- 输出维度：`latent_dim = 32`

```python
# source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py
self.teacher_encoder = nn.Sequential(
    MLP(mlp_input_dim_t, latent_dim, teacher_encoder_hidden_dims, activation),
    L2Norm()  # 或 SimNorm
)
```

#### 学生 MoE 编码器

$$g = \text{Softmax}(\text{MLP}_{\text{gate}}(o_a))$$
$$e_i = \text{Expert}_i(\text{Backbone}(o_a)), \quad i=1,\dots,N$$
$$z_s = \text{L2Norm}\left(\sum_{i=1}^{N} g_i \cdot e_i\right)$$

- 专家数量：N = 8
- 共享主干：MLP[512, 256, 256]
- 每个专家：Conv1d(groups=8) 分组卷积
- 门控网络：MLP[512, 256, 256] → Softmax
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

### L2Norm 与 SimNorm

隐向量归一化用于稳定蒸馏训练：

$$\text{L2Norm}(x) = \frac{x}{\|x\|_2}$$

$$\text{SimNorm}(x) = \text{Softmax}(x_{\text{reshape}[-1, 8]}) \quad \text{(Simplicial Normalization)}$$

### CatELU 激活函数（可选）

受 [Concat ReLU](https://arxiv.org/pdf/2303.07507) 启发，CatELU 将特征维度翻倍：

$$\text{CatELU}(x) = [\text{ELU}(x), \text{ELU}(-x)]$$

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
│   │           │   ├── mdp/     # 奖励/观测/指令/事件/课程
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
│   ├── flat.xml / stairs.xml / boxes.xml  # 地形场景
│   └── legbot.xml               # LegBot MuJoCo 模型
├── src/                         # C++ 仿真与桥接
│   ├── legbot_bridge.h          # DDS 桥接（MuJoCo）
│   ├── param.h                  # 仿真配置
│   └── main.cc                  # MuJoCo 仿真的入口
├── TECHNICAL_DOC_zh.md          # 详细技术文档（中文）
└── README_cn.md                 # 本文件
```

### 模块依赖关系

```
train.py / play.py
    │
    ├── robot_lab.tasks.legbot.__init__  (注册 gym 环境 RobotLab-Legbot-v0)
    │       │
    │       ├── LegbotEnvCfg  (env_cfg.py) ── 场景/观测/奖励/事件/课程配置
    │       ├── LegbotEnv     (legbot_env.py) ── 重写 ActionManager
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

## MDP 定义

### 观测空间

定义于 [env_cfg.py](source/robot_lab/robot_lab/tasks/legbot/env_cfg.py)：

| 观测组 | 用途 | 历史长度 | 是否加噪 | 包含项 |
|--------|------|---------|---------|--------|
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

- 类型：关节位置控制（`JointPositionActionCfg`）
- 维度：12
- 缩放：0.25（动作 × 0.25 + 默认关节角 = 目标关节角）
- 裁剪范围：[-100, 100]

### 指令空间

指令由 `Go2RLGymCommand` 生成，维度为 3：`[lin_vel_x, lin_vel_y, ang_vel_yaw]`

- 重采样间隔：5s
- 动态重采样：根据剩余距离与剩余回合时间调整速度下界
- 地形相关范围：不同地形类型对应不同的速度上限
- 指令范围课程：在 20k 和 50k 迭代时扩展速度范围
- 零指令课程：训练初期零指令概率为 0，逐步增加至 0.1

### 终止条件

- 超时：25s / episode
- 非法接触：base 链接接触力 > 1.0N

---

## 奖励函数设计

定义于 [rewards.py](source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py)，采用多项加权求和。

### 跟踪奖励（正向）

**线速度跟踪**（指数核）：

$$r_{\text{lin}} = \exp\left(-\frac{\|v_{\text{cmd}}^{xy} - v_{\text{base}}^{xy}\|^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=1.0$$

**角速度跟踪**：

$$r_{\text{ang}} = \exp\left(-\frac{(\omega_{\text{cmd}}^{z} - \omega_{\text{base}}^{z})^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=0.5$$

### 惩罚项（负向）

| 奖励项 | 公式 | 权重 | 说明 |
|--------|------|------|------|
| lin_vel_z_l2 | $\|v_z\|^2$ | -2.0 → 0（课程） | 惩罚垂直速度 |
| ang_vel_xy_l2 | $\|\omega_{xy}\|^2$ | -0.05 | 惩罚 roll/pitch 角速度 |
| joint_acc_l2 | $\sum \ddot{q}_i^2$ | -1e-7 | 关节加速度（Lab 物理步级，权重极小） |
| joint_power | $\sum \|\dot{q}_i \cdot \tau_i\|$ | -2e-5 | 关节功率 |
| joint_torques_l2 | $\sum \tau_i^2$ | -1e-4 | 关节力矩 |
| base_height_l2 | $(h - 0.28)^2$ | -1.0 → -10.0（课程） | 基座高度（使用高度扫描估计地面） |
| action_rate_l2 | $\|a_t - a_{t-1}\|^2$ | -0.01 | 动作变化率 |
| action_smoothness_l2 | $\|a_t - 2a_{t-1} + a_{t-2}\|^2$ | -0.01 | 动作平滑性（二阶差分） |
| undesired_contacts | $\sum \mathbb{1}(\|F\| > 5)$ | -1.0 | 大腿/小腿非法接触 |
| joint_pos_limits | 关节超限惩罚 | -2.0 | 关节位置限制 |
| feet_regulation | $\sum v_{\text{foot}}^{xy\,2} \cdot e^{-h_{\text{foot}}/(0.025 \cdot h_{\text{target}})}$ | -0.05 | 惩罚近地脚横向滑动 |
| hip_pos_penalty_l1 | $\sum \|q_{\text{hip}} - q_{\text{default}}\|_1$ | -0.05 | 髋关节偏离默认位置 |
| joint_pos_penalty_l1 | $\sum \|q_{\text{thigh,calf}} - q_{\text{default}}\|_1$ | -0.01 | 大腿/小腿偏离默认位置 |

### 基座高度估计

`base_height_l2` 与 `feet_regulation` 使用高度扫描器估计地面高度，而非直接使用世界坐标 z：

```python
base_height = base_z - mean(ray_hits_z)  # 减去估计的地面高度
```

---

## 域随机化与电机模型

### 域随机化

| 随机化项 | 模式 | 范围 | 说明 |
|---------|------|------|------|
| 基座质量 | startup | ±1 kg | 适应负载变化 |
| 其他部件质量 | startup | ×[0.9, 1.1] | 质量分布变化 |
| 质心位置 | startup | ±0.03 m | 质心偏移 |
| 关节重置位置 | reset | ×[0.5, 1.5] | 初始姿态随机 |
| 执行器增益 (kp/kd) | reset | ×[0.9, 1.1] | PD 参数扰动 |
| 电机零偏 | reset | ±0.035 rad | 编码器零位误差 |
| 推力扰动 | interval (4s) | ±0.4 m/s, ±0.6 rad/s | 随机外力 |
| 摩擦系数 | startup | [0, 2.0] | 不同地面摩擦 |
| 基座初始状态 | reset | pos ±0.5m, yaw ±π | 随机初始位姿 |

### Unitree 电机模型

本项目使用 Unitree 官方扭矩-转速曲线模型（Go2 HV 参数）：

```
    扭矩上限 (N·m)
        ^
Y2──────|
        |──────────Y1
        |          |\
        |          | \
        |          |  \
        |          |   \
--------+----------|----> 速度 (rad/s)
                 X1   X2
```

| 参数 | 值 | 说明 |
|------|-----|------|
| X1 | 13.5 rad/s | 满扭矩最大转速（T-N 曲线拐点） |
| X2 | 30 rad/s | 空载转速 |
| Y1 | 20.2 N·m | 同向峰值扭矩 |
| Y2 | 23.4 N·m | 反向峰值扭矩 |

**摩擦模型：**

$$\tau_{\text{applied}} = \tau_{\text{PD}} - F_s \cdot \tanh\left(\frac{\dot{q}}{V_a}\right) - F_d \cdot \dot{q}$$

**扭矩裁剪：**
- $|\dot{q}| < X1$：限制为 Y1（同向）或 Y2（反向）
- $|\dot{q}| \geq X1$：线性下降至 0（在 X2 处）

**电机延迟**：`min_delay=0, max_delay=4` 步（电机级延迟，非动作级延迟）

### 动作管理器

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

## 课程学习

### 地形课程（terrain_levels_vel_gym）

根据机器人移动距离动态调整地形难度：
- `move_up`：最大移动距离 > 地形长度 / 2 → 提升难度等级
- `move_down`：最大移动距离 < 目标距离 × 0.5 → 降低难度等级

### 奖励权重课程（gradual_reward_weight_modification）

线性插值修改奖励权重：
- `lin_vel_z_l2`：-2.0 → 0.0（0→1500 迭代）
- `base_height_l2`：-1.0 → -10.0（0→5000 迭代）

### 指令范围课程（command_range_curriculum）

在指定迭代步扩展速度指令范围：
```python
# 20000 迭代：lin_vel_x [-1,1], lin_vel_y [-1,1], ang_vel [-1.5,1.5]
# 50000 迭代：lin_vel_x [-2,2], lin_vel_y [-1,1], ang_vel [-2,2]
```

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

# 恢复训练
python scripts/rsl_rl/train.py --task=RobotLab-Legbot-v0 --headless \
    --resume --load_run=2026-07-07_12-09-58 --checkpoint=model_37000.pt

# 评估
python scripts/rsl_rl/play.py --task=RobotLab-Legbot-v0
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

### 训练循环（OnPolicyRunnerCTS）

定义于 [on_policy_runner_cts.py](source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py)：

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

---

## MuJoCo Sim2Sim 部署

### 策略导出

运行 `play.py` 会同时导出 TorchScript（`.pt`）和 ONNX（`.onnx`）格式。导出器将学生分支（StudentMoEEncoder + Actor）与归一化器封装为单输入模型：

```bash
python scripts/rsl_rl/play.py \
    --task=RobotLab-Legbot-v0 \
    --checkpoint logs/rsl_rl/legbot_moe_cts/<run_name>/model_<iter>.pt
```

**关键特性**：导出的策略**内部维护观测历史**，部署时只需输入当前帧观测。

```python
# source/rsl_rl/rsl_rl/utils/exporter_cts.py
class _TorchPolicyExporter:
    def forward(self, single_obs):
        self._update_history(single_obs)         # 更新内部历史
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(self.obs_history)
        latent, _ = self.student_moe_encoder(obs_a)   # MoE 编码
        return self.actor(torch.cat([latent, single_obs], dim=-1))
```

### MuJoCo 中运行

编辑 `deploy/deploy_mujoco/configs/legbot.yaml` 中的策略路径：

```yaml
policy_path: "{ROOT_DIR}/logs/rsl_rl/legbot_moe_cts/<timestamp>/exported/policy.pt"
```

运行：

```bash
python deploy/deploy_mujoco/deploy_legbot.py
```

**部署循环：**

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

### 切换仿真场景

修改 `legbot.yaml` 中的 `xml_path`：

```yaml
# 平坦地形
xml_path: "{ROOT_DIR}/resources/legbot/flat.xml"
# 楼梯
xml_path: "{ROOT_DIR}/resources/legbot/stairs.xml"
# 箱子障碍
xml_path: "{ROOT_DIR}/resources/legbot/boxes.xml"
# 楼梯与斜坡组合
xml_path: "{ROOT_DIR}/resources/legbot/stairs_and_slope.xml"
```

### 手柄控制

| 输入 | 功能 |
|------|------|
| LX | 前进/后退速度 |
| LY | 左移/右移速度 |
| RX | 角速度（转向） |

- 自动检测手柄连接，无手柄时使用配置文件中的默认指令 `cmd_init: [1.0, 0.0, 0.0]`

---

## 与 go2_rl_gym 的主要差异

| 方面 | go2_rl_gym (IsaacGym) | legbot_lab (IsaacLab) |
|------|----------------------|---------------------------|
| 电机模型 | 简单 PD 控制器 | Unitree 官方扭矩-转速曲线模型 |
| 跟踪奖励 | 动态 sigma | 固定 sigma=0.5 |
| joint_acc_l2 权重 | 较大 | 极小（-1e-7），因 Lab 物理步级计算更精确 |
| joint_pos_penalty_l1 | 无 | 有 |
| 动作延迟 | 随机动作延迟 | 电机级延迟（min_delay=0, max_delay=4） |
| 电机强度随机化 | 有 | 无（Lab 实现约束） |
| 历史长度 | 5 | 10（Lab 中更优） |
| 算法 | CTS | MoE-CTS（增加混合专家） |

---

## 致谢

本项目基于以下开源项目：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — NVIDIA Isaac Sim 统一机器人学习框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — 强化学习算法库
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — IsaacLab 机器人 RL 扩展
- [MuJoCo](https://github.com/google-deepmind/mujoco) — 高性能物理仿真器
- [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) — IsaacGym 版 Go2 RL 训练（原始实现）

相关论文：

- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)

---

## 许可证

本项目基于 Apache 2.0 许可证发布。详见各源文件的许可证声明。
