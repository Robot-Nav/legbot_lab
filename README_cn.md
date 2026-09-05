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
- 单帧策略观测为 53 维，带 10 帧历史。
- 每步输出 16 维动作（12 个腿部位置 + 4 个轮速）。

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
  legbot_wf/              Legbot-WF 模型与场景（未包含，见下文说明）
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

```bash
python scripts/rsl_rl/train.py \
  --task=RobotLab-Legbot-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --run_name v1
```

检查点保存在 `logs/rsl_rl/legbot_wf_moe_cts/<run>/model_<iter>.pt`。

### 播放与导出

```bash
python scripts/rsl_rl/play.py \
  --task=RobotLab-Legbot-v0 \
  --checkpoint=/绝对路径/model_xxx.pt
```

`play.py` 同时会把学生策略导出为 `<run>/exported/policy.pt` 与 `policy.onnx`。

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

- `resources/legbot_wf/`（URDF、网格、场景）已通过 `.gitignore` 排除。
- 训练检查点与导出策略（`.pt`、`.onnx`、`.pth`、`.ckpt`）均已排除。

如需使用 Legbot 任务，请将你自己的机器人模型放到 `resources/legbot_wf/` 后从零训练。`resources/go2/` 下的 Go2 模型为公开 Unitree 资产，已包含在仓库中。

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
