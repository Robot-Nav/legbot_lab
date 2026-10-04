<div align="center">

# WF-CTS-MOE

[English](README.md) | [中文](README_cn.md)

面向 16 自由度轮足四足机器人的强化学习项目，基于 Isaac Lab 与 RSL-RL 构建。

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-3.3.0-orange.svg)](https://github.com/leggedrobotics/rsl_rl)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.4.0-lightgrey.svg)](https://mujoco.org/)

</div>

## 项目概述

WF-CTS-MOE 用于训练 16 自由度的轮足四足机器人 **Legbot-WF**，使其能根据速度指令在崎岖地形上运动。训练算法为 **MoE-CTS**：在 Concurrent Teacher-Student（CTS）强化学习基础上引入 Mixture-of-Experts（混合专家），运行于 Isaac Lab 与定制版 RSL-RL 之上。

机器人构成：

- 12 个腿部关节（髋 / 大腿 / 小腿 × 4 条腿），采用位置控制。
- 4 个主动轮（每条腿末端一个），采用速度控制。
- 单帧策略观测为 57 维，带 4 帧历史（共 228 维），与 W1W 部署控制器一致。
- 每步输出 16 维动作，按每条腿「髋、大腿、小腿、轮」排列（腿位置偏移 + 轮速）。

项目中同时保留了 Go2 任务作为参考环境，两个机器人共用同一套算法核心。

## 项目意义

CTS 将训练拆分为共享同一 Actor 与 Critic 的教师网络和学生网络：

- 教师读取完整、特权化的观测，生成隐变量表示。
- 学生只读取机载观测（含历史），并被训练为逼近教师的隐变量。

实际部署时只使用学生路径，教师仅在训练阶段起到监督作用，不会运行在机器人上。

MoE-CTS 把学生侧的普通 MLP 编码器替换为混合专家编码器。不同专家可以分别学习不同运动模式（支撑、摆动、轮式滚动、攀爬等），由门控网络进行选择；负载均衡损失避免专家退化为单一专家。

## 算法原理

### 符号定义

| 符号 | 含义 |
| --- | --- |
| `o_priv` | 特权观测（critic 组） |
| `o_on` | 含历史的机载观测（policy 组） |
| `o_single` | 单帧机载观测 |
| `f_theta` | 教师编码器：`o_priv -> z_t` |
| `g_phi` | 学生 MoE 编码器：`o_on -> z_s` |
| `pi` | 共享 Actor：`(z, o_single) -> a` |
| `V` | 共享 Critic：`(z, o_priv) -> value` |
| `rho` | 教师环境比例（0.75） |

### 教师-学生拆分

一次 rollout 的 `N` 个环境被划分为 `rho * N` 个教师环境与 `(1 - rho) * N` 个学生环境。两者共用同一 Actor 与 Critic，但隐变量计算方式不同：

- 教师：`z_t = f_theta(o_priv)`
- 学生：`z_s = g_phi(o_on)`

### PPO 目标

教师与学生样本统一使用带裁剪的 PPO 目标优化。

替代损失：

```math
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
```

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
```

价值损失：

```math
L^{VF} = \mathbb{E}_t\left[(V_\theta(s_t) - R_t)^2\right]
```

PPO 总损失：

```math
L^{PPO} = L^{CLIP} + c_1 L^{VF} - c_2\, H[\pi_\theta]
```

优势函数使用广义优势估计（GAE）：

```math
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

学习率依据 `desired_kl = 0.01` 的自适应 KL 调度进行调整。

### MoE 学生编码器

学生编码器为混合专家层：

```math
z_s = \sum_{i=1}^{N} w_i(x)\, e_i(x), \qquad w(x) = \mathrm{softmax}(g(x))
```

其中 `g` 为门控网络，`e_i` 为各专家。默认配置使用 8 个专家。

为使各专家保持均衡，门控权重被正则化到均匀分布：

```math
L^{LB} = \sum_{i=1}^{N} \left(\bar{w}_i - \frac{1}{N}\right)^2, \qquad \bar{w}_i = \mathbb{E}_x\left[w_i(x)\right]
```

学生编码器使用独立优化器与蒸馏损失训练：

```math
L^{student} = \left\| z_t - z_s \right\|_2^2 + \alpha\, L^{LB}
```

实现中 `alpha` 即 `load_balance_coef = 0.01`。

### 默认超参数

| 参数 | 取值 |
| --- | --- |
| `clip_param` | 0.2 |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `value_loss_coef` | 1.0 |
| `entropy_coef` | 0.01 |
| `load_balance_coef` | 0.01 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | 1e-3 |
| `student_encoder_learning_rate` | 1e-3 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `teacher_env_ratio` | 0.75 |
| `expert_num` | 8 |
| `latent_dim` | 32 |

## 四轮足训练配置详表

以下按 2026-10-04 核对的当前代码整理，适用于 `RobotLab-Legbot-Simplified-v0` 和默认任务 `RobotLab-Legbot-v0`，模型为 `resources/w1w_wf/urdf/w1w_wf.urdf`。`RobotLab-Legbot-Original-v0` 使用原始网格模型。命令行覆盖以具体运行的 `params/env.yaml`、`params/agent.yaml` 为准。

### 算法结构与训练参数

算法为 **PPO＋教师—学生联合训练＋MoE混合专家学生编码器**。教师根据特权信息学习隐变量；学生从本体观测历史估计隐变量，两者共用动作网络。部署使用学生分支。

| 项目 | 当前配置 |
| --- | --- |
| 算法 / 训练器 / 策略 | `MoECTS` / `OnPolicyRunnerCTS` / `ActorCriticMoECTS` |
| 教师编码器 | 输入295维特权观测，隐藏层 `[512,256]`，输出32维隐变量 |
| 学生编码器 | 输入228维历史观测，8个专家，隐藏层配置 `[512,256,256]` |
| 专家融合 | Softmax门控加权求和；隐变量做L2归一化 |
| Actor | 32维隐变量＋57维当前观测＝89维输入；隐藏层 `[512,256,128]`；16维动作均值 |
| Critic | 295维特权观测＋32维隐变量＝327维输入；隐藏层 `[512,256,128]`；1维状态价值 |
| 激活函数 | ELU |
| 教师 / 学生环境 | 75% / 25%；2048环境对应1536 / 512 |
| PPO更新对象 | 教师编码器、共享Actor、Critic、动作标准差 |
| 学生编码器更新 | 在学生环境状态上匹配教师隐变量；教师目标停止梯度，独立优化器 |
| 学生损失 | 隐变量MSE＋0.01×平均专家门控权重相对均匀分布的MSE |
| 训练 / 推理动作 | 训练为高斯采样，初始标准差1.0；推理使用学生动作均值 |
| 默认并行环境数 | 2048 |
| 每轮采样 | 每环境24步，共49152个样本 |
| PPO更新epoch / mini-batch数 | 5 / 4 |
| PPO裁剪 / 价值损失 | 裁剪系数0.2；价值系数1.0，启用价值裁剪 |
| 折扣因子γ / GAE λ | 0.99 / 0.95 |
| PPO / 学生编码器学习率 | 初始均为0.001；PPO按KL自适应，目标KL为0.01 |
| 熵系数 / 梯度范数上限 | 0.01 / 1.0 |
| 优化器 | Adam，β=`(0.9,0.999)`，weight decay=0 |
| 仿真 / 控制周期 | 0.005秒（200 Hz）/ 0.02秒（50 Hz），decimation=4 |
| 最大回合长度 | 25秒，即1250个控制步 |
| 保存间隔 / 默认最大轮数 | 每50轮 / 300000轮；此前运行通过命令行指定100000轮 |

### 奖励函数：14项

每个控制步 `r_t = 0.02 × Σ(w_i × f_i)`。下表计算内容为乘权重和时间步长之前的原始项；`q₀`为默认关节位置，`a`为动作管理器中的动作。

| 名称 | 计算内容 | 权重 | 作用 / 范围 |
| --- | --- | --- | --- |
| `track_lin_vel_xy_exp` | `exp(−水平速度误差平方和/0.25)` | +2.0 | 跟踪机身坐标系vx、vy指令 |
| `track_ang_vel_z_exp` | `exp(−yaw角速度误差平方/0.25)` | +1.0 | 跟踪转向指令 |
| `lin_vel_z_l2` | 机身坐标系z向速度平方 | −2→−1 | 抑制上下运动；前1500轮线性调整 |
| `ang_vel_xy_l2` | roll、pitch角速度平方和 | −0.05 | 抑制机身晃动 |
| `joint_acc_l2` | 全部16关节加速度平方和 | −1e−7 | 抑制关节加速度过大 |
| `joint_power` | `Σabs(τ×q̇)` | −2e−5 | 全部16关节绝对机械功率之和 |
| `joint_torques_l2` | 全部16关节力矩平方和 | −1e−4 | 抑制力矩过大 |
| `base_height_l2` | `(离地高度−0.49)²` | −1→−10 | 保持机身高度；前5000轮线性调整 |
| `action_rate_l2` | `Σ(aₜ−aₜ₋₁)²` | −0.01 | 抑制动作突变 |
| `action_smoothness_l2` | `Σ(aₜ−2aₜ₋₁+aₜ₋₂)²` | −0.01 | 抑制动作二阶变化；回合步数大于2时生效 |
| `undesired_contacts` | 接触力超过5 N的大腿、小腿刚体数量 | −1.0 | 使用接触力历史最大值 |
| `joint_pos_limits` | 超出软关节限位的距离之和 | −2.0 | 仅12个腿关节，不含轮子 |
| `hip_pos_penalty_l1` | 4个髋关节相对q₀的绝对偏差和 | −0.05 | 限制髋关节偏移 |
| `joint_pos_penalty_l1` | 8个大腿、小腿关节相对q₀的绝对偏差和 | −0.01 | 限制腿部偏离默认姿态 |

- 离地高度由机身世界高度减去小范围扫描的平均地面高度计算。
- 两个关节姿态偏差项的 `stand_still_scale=1`，静止和运动分支倍率相同。
- 大腿、小腿接触扣分；机身接触属于终止条件。
- 当前未独立启用轮胎打滑、步态相位、抬脚高度或左右对称奖励。
- `Episode return` 是回合总奖励；TensorBoard 的 `Episode_Reward/*` 是分项回合累计值除以最大回合时长25秒，统计口径不同。

### 域随机化与初始状态随机化

范围采用均匀采样。启动时随机化在环境初始化时分配，不会每回合重新抽样。

| 项目 | 时机 | 范围 / 方式 | 对象 / 说明 |
| --- | --- | --- | --- |
| 机身质量 | 启动时 | 原质量加±1 kg | base；同步重新计算惯量 |
| 其他刚体质量 | 启动时 | 原质量乘0.9～1.1 | 除base外刚体；同步更新惯量 |
| 机身质心 | 启动时 | x、y、z各偏移±0.03 m | base |
| 静摩擦系数 | 启动时 | 0～2 | 所有机器人碰撞刚体 |
| 动摩擦系数 | 启动时 | 0～2，限制不大于静摩擦 | `make_consistent=True` |
| 恢复系数 | 启动时 | 0～0.5 | 与摩擦共同使用64组材料桶 |
| 初始关节位置 | 每次重置 | 默认角度乘0.9～1.1 | 默认角为0的关节仍为0 |
| 初始关节速度 | 每次重置 | 0 | 当前没有初始关节速度扰动 |
| Kp | 每次重置 | 标称值乘0.9～1.1 | 轮子Kp为0，随机化后仍为0 |
| Kd | 每次重置 | 标称值乘0.9～1.1 | 全部关节 |
| 电机零位偏差 | 每次重置 | ±0.035 rad | 12个腿关节的位置目标偏移 |
| 初始水平位置 | 每次重置 | x、y各±0.5 m | 相对默认位置及环境原点 |
| 初始高度 | 每次重置 | 0.53 m＋0～0.05 m | 另外叠加环境原点高度 |
| 初始朝向 | 每次重置 | yaw±3.14 rad | roll、pitch没有随机角度扰动 |
| 初始线速度 | 每次重置 | 三轴各±0.5 m/s | 机身根节点速度 |
| 初始角速度 | 每次重置 | 三轴各±0.5 rad/s | 与初始姿态角不同 |
| 水平推扰 | 每4秒 | 世界系vx、vy各增加±0.4 m/s | z速度增量为0 |
| 角速度推扰 | 每4秒 | 世界系三轴角速度各增加±0.6 rad/s | 通过速度跳变实现 |
| 观测噪声 | 生成学生观测时 | 见下表 | 特权观测不加噪声 |

当前未启用随机动作延迟、通信丢包、独立电机强度随机化、重力随机化、独立惯量随机化。惯量仅随质量调整。本机 `push_by_setting_velocity` 实现向当前速度增加扰动。

### 学生观测：57维×4帧＝228维

处理顺序：原始值→噪声→裁剪→缩放→历史拼接。以下项目均在缩放前裁剪到 `[-100,100]`。

| 观测项 | 单帧维度 | 含义 | 缩放 | 均匀噪声（缩放前） |
| --- | --- | --- | --- | --- |
| `base_ang_vel` | 3 | 机身坐标系角速度 | 0.25 | ±0.2 rad/s |
| `projected_gravity` | 3 | 机身坐标系重力方向 | 1 | ±0.05 |
| `velocity_commands` | 3 | vx、vy、yaw速度指令 | `(2,2,0.25)` | 无 |
| `joint_pos` | 16 | 相对默认站姿的位置 | 1 | 腿关节±0.03 rad；轮子列最终清零 |
| `joint_vel` | 16 | 相对默认速度的关节速度 | 0.05 | ±2 rad/s |
| `actions` | 16 | 上一次动作 | 1 | 无 |
| 合计 | 57 | 4帧历史 | 228维 | — |

| 观测组 / 规则 | 实际含义 |
| --- | --- |
| `policy` | 228维历史，输入学生MoE编码器 |
| `single_obs` | 57维，与policy历史最新帧一致，输入共享Actor |
| 历史排列 | 每个观测项保存4帧后，再拼接各项；不是整帧连续堆叠 |
| 轮子位置 | 保留4个位置维度，恒为0，包括清除这些列的噪声 |
| 学生信息范围 | 不直接提供真实机身线速度、力矩、接触力或地形扫描 |

### 特权观测：295维

教师编码器与Critic使用，不加观测噪声，不做整组4帧堆叠。

| 观测项 | 维度 | 缩放 | 说明 |
| --- | --- | --- | --- |
| 机身线速度 | 3 | 2.0 | 机身坐标系 |
| 机身角速度 | 3 | 0.25 | 机身坐标系 |
| 投影重力 | 3 | 1.0 | 重力方向 |
| 速度指令 | 3 | 1.0 | 与学生指令缩放不同 |
| 相对关节位置 | 16 | 1.0 | 轮子位置为0 |
| 相对关节速度 | 16 | 0.05 | 全部关节 |
| 上一次动作 | 16 | 1.0 | 电机顺序 |
| 关节加速度 | 16 | 0.0001 | 全部关节 |
| 实际施加力矩 | 16 | 0.01 | 全部关节 |
| 轮端接触力 | 16 | 0.001 | 4轮×〔3帧力模长＋历史最大值〕 |
| 地形高度扫描 | 187 | 2.5 | 范围1.6×1.0 m，网格0.1 m |
| 总计 | 295 | — | 已与训练启动日志核对 |

高度扫描缩放前裁剪到 `[-1,1]`，其余项缩放前裁剪到 `[-100,100]`。

### 动作空间与执行器

共16维连续动作，训练包装器先将网络动作裁剪到 `[-10,10]`。

| 动作类别 | 维度 | 目标计算 | 控制方式 |
| --- | --- | --- | --- |
| hip髋关节 | 4 | `q目标 = 默认角度＋零位扰动＋0.25a` | 位置PD |
| thigh大腿 | 4 | 同上 | 位置PD |
| calf小腿 | 4 | 同上 | 位置PD |
| foot轮子 | 4 | `ω目标 = clip(20a,−150,150)` rad/s | 速度伺服 |

腿部动作项另有 `[-100,100]` rad的目标裁剪，这与URDF物理关节限位不同。

| 动作下标（从0开始） | 对应关节 |
| --- | --- |
| 0～3 | 左前：hip、thigh、calf、foot |
| 4～7 | 右前：hip、thigh、calf、foot |
| 8～11 | 左后：hip、thigh、calf、foot |
| 12～15 | 右后：hip、thigh、calf、foot |

| 关节 | 标称Kp | 标称Kd | 力矩上限 | 仿真速度限制 |
| --- | --- | --- | --- | --- |
| hip | 100 | 2 | 120 N·m | 15 rad/s |
| thigh | 100 | 2 | 120 N·m | 15 rad/s |
| calf | 100 | 2 | 175.38 N·m | 10.26 rad/s |
| foot | 0 | 1 | 28.68 N·m | 104.72 rad/s |

轮速目标裁剪±150 rad/s与仿真速度限制104.72 rad/s是不同设置。

| 默认站姿关节 | 左侧腿 | 右侧腿 |
| --- | --- | --- |
| hip | 0 | 0 |
| thigh | −0.68 rad | +0.68 rad |
| calf | +1.40 rad | −1.40 rad |
| foot | 0 | 0 |

左右符号差异来自URDF关节轴方向。

### 终止条件：2项

| 条件 | 判定 | 类型 | 结果 |
| --- | --- | --- | --- |
| `time_out` | 达到1250步，即25秒 | 时间截断 | 重置；算法处理超时价值补偿 |
| `illegal_contact` | base净接触力模长在3帧历史中的最大值大于10 N | 失败终止 | 重置环境 |

| 其他情况 | 当前处理 |
| --- | --- |
| 大腿、小腿接触 | 超过5 N扣分，不直接终止 |
| 关节超限 | 软限位惩罚及物理限位 |
| 机身倾角过大 | 没有独立倾角终止 |
| 机身高度过低 | 高度惩罚，没有独立高度终止 |
| 速度跟踪失败 | 跟踪奖励降低 |
| 轮胎打滑 | 没有独立终止项 |

base判定使用净接触力，没有进一步区分接触对象。

### 速度课程与地形

| 训练轮数 | vx（m/s） | vy（m/s） | yaw角速度（rad/s） |
| --- | --- | --- | --- |
| 0～19999 | ±0.5 | ±0.5 | ±1.0 |
| 20000～49999 | ±1.0 | ±1.0 | ±1.5 |
| 50000～74999 | ±2.0 | ±1.0 | ±2.0 |
| 75000及以后 | ±3.0 | ±1.0 | ±3.14 |

不同地形还有进一步限速，以上是全局指令范围。

| 地形 | 配置比例 |
| --- | --- |
| 波浪 | 5% |
| 上坡 | 10% |
| 下坡 | 10% |
| 粗糙坡地 | 5% |
| 上台阶 | 25% |
| 下台阶 | 10% |
| 障碍物 | 20% |
| 平地 | 15% |
| 踏石、沟壑 | 0% |

地形分为10级，初始最大等级5，根据运动完成情况升降级。固定任务条件下的学生独立评测、动作延迟、轮胎打滑以及速度课程跳变对训练的影响，仍是后续评估重点。

配置来源：[环境配置](source/robot_lab/robot_lab/tasks/legbot/env_cfg.py)、[算法配置](source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py)、[算法实现](source/rsl_rl/rsl_rl/algorithms/moe_cts.py)、[奖励实现](source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py)、[观测对齐](source/robot_lab/robot_lab/tasks/legbot/env/observations.py)、[执行器配置](source/robot_lab/robot_lab/assets/legbot.py)、[地形配置](source/robot_lab/robot_lab/tasks/go2/mdp/terrains.py)。

## 项目结构

```text
scripts/
  rsl_rl/                 train.py、play.py、cli_args.py、rsl_rl_utils.py
  tools/                  URDF/MJCF 转换与清理辅助脚本
source/
  rsl_rl/                 定制版 RSL-RL，包含 MoE-CTS
  robot_lab/              robot_lab 扩展：Legbot 与 Go2 任务/资产
deploy/
  deploy_mujoco/          MuJoCo Sim2Sim 部署（Legbot 与 Go2）
resources/
  go2/                    Go2 模型与场景（公开 Unitree 资产）
  w1w_wf/                 Legbot-WF 模型与场景（本地私有资产，不纳入 Git）
```

MoE-CTS 核心代码位于：

- `source/rsl_rl/rsl_rl/algorithms/moe_cts.py`
- `source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py`
- `source/rsl_rl/rsl_rl/networks/moe.py`
- `source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py`
- `source/rsl_rl/rsl_rl/storage/rollout_storage_cts.py`

## 依赖库

- Python 3.11
- Isaac Lab `2.3.2.post1`
- PyTorch `2.7.0`、torchvision `0.22.0`
- RSL-RL `3.3.0`（定制版，从 `source/rsl_rl` 安装）
- robot_lab `2.3.0`（定制版，从 `source/robot_lab` 安装）
- tensordict、numpy、onnx、onnxscript、GitPython
- MuJoCo 与 pygame（可选，用于 Sim2Sim）

## 安装步骤

1. 安装 Isaac Lab：

```bash
conda create -n wf_cts_moe python=3.11
conda activate wf_cts_moe
pip install --upgrade pip
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

2. 以可编辑模式安装定制版 RSL-RL 与 robot_lab：

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

3. 安装 MuJoCo（可选，用于 Sim2Sim）：

```bash
pip install mujoco pygame
```

## 运行步骤

### 训练

默认任务 `RobotLab-Legbot-v0` 和显式任务 `RobotLab-Legbot-Simplified-v0` 均读取 `resources/w1w_wf/urdf/w1w_wf.urdf`，默认 2048 环境。原始网格模型仅由 `RobotLab-Legbot-Original-v0` 加载，日志单独存放。右前轮统一为 `fr_foot_joint`。此前原始模型的短测试只验证了程序能运行，不能证明策略有效；2048 环境也曾出现 CUDA 700，减少环境数量并非已验证的根治方法。

```bash
python scripts/rsl_rl/train.py \
  --task=RobotLab-Legbot-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 300000 \
  --run_name v1
```

检查点保存在 `logs/rsl_rl/w1w_wf_moe_cts/<run>/model_<iter>.pt`。

### 播放与导出

```bash
python scripts/rsl_rl/play.py \
  --task=RobotLab-Legbot-v0 \
  --num_envs 1 \
  --checkpoint=/绝对路径/model_xxx.pt
```

`play.py` 同时会把学生策略导出为 `<run>/exported/policy.pt` 与 `policy.onnx`。训练侧接口已与 `w1w_deploy` 对齐；旧检查点必须按新配置重新训练，导出的模型还需经过仿真与受支撑实机验证。

### MuJoCo Sim2Sim

```bash
# 仅校验模型与配置，不打开窗口
python deploy/deploy_mujoco/deploy_legbot.py --validate

# 在 MuJoCo 中运行导出的策略
python deploy/deploy_mujoco/deploy_legbot.py \
  --policy=/绝对路径/exported/policy.pt

# 选择地形：flat（默认）、stairs、rough、mixed
python deploy/deploy_mujoco/deploy_legbot.py \
  --terrain=mixed \
  --policy=/绝对路径/exported/policy.pt
```

追加 `--headless --duration 5` 可进行无渲染的短时冒烟测试。

### Go2 参考任务

```bash
python scripts/rsl_rl/train.py --task=RobotLab-Go2-v0 --headless
python scripts/rsl_rl/play.py --task=RobotLab-Go2-v0
```

## 配置说明

任务配置位于两处：

- 环境配置：`source/robot_lab/robot_lab/tasks/legbot/env_cfg.py`
- 算法配置：`source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py`

任务在 `source/robot_lab/robot_lab/tasks/legbot/__init__.py` 中注册为 `RobotLab-Legbot-v0`。

命令行可覆盖的参数包括：`--num_envs`、`--max_iterations`、`--run_name`、`--experiment_name`、`--checkpoint`、`--seed`、`--logger`。

## 模型与权重说明

Legbot-WF 模型为私有资产，本仓库**不包含**：

- `resources/w1w_wf/`（URDF、网格、场景）已通过 `.gitignore` 排除。
- 训练检查点与导出策略（`.pt`、`.onnx`、`.pth`、`.ckpt`）均已排除。

如需使用 Legbot 任务，请将你自己的机器人模型放到 `resources/w1w_wf/` 后从零训练。W1W 部署源码已纳入 Git，但虚拟环境、构建产物和策略权重均不纳入。`resources/go2/` 下的 Go2 模型为公开 Unitree 资产，已包含在仓库中。

## 许可证

本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE)。

仓库中包含了派生自开源项目的代码，其许可证保留在对应源文件与 `source/rsl_rl/licenses` 目录中。

## 致谢

本项目基于以下开源项目构建：

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym)

算法参考文献：

- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/abs/2405.10830)
