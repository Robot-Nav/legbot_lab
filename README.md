<div align="center">

# WF-CTS-MOE

[English](README.md) | [中文](README_cn.md)

Reinforcement learning for a 16-DOF wheel-foot quadruped, built on Isaac Lab and RSL-RL.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-3.3.0-orange.svg)](https://github.com/leggedrobotics/rsl_rl)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.4.0-lightgrey.svg)](https://mujoco.org/)

</div>

## Overview

WF-CTS-MOE trains a 16-DOF wheel-foot quadruped, **Legbot-WF**, to move over rough terrain with velocity commands. The learner is **MoE-CTS**: a Mixture-of-Experts version of Concurrent Teacher-Student (CTS) reinforcement learning, running on top of Isaac Lab and a modified RSL-RL.

The robot has:

- 12 leg joints (hip / thigh / calf × 4 legs), controlled by position.
- 4 active wheels (one per foot), controlled by velocity.
- A single-frame policy observation of 53 values, with a 10-frame history.
- 16 actions per step (12 leg positions + 4 wheel velocities).

A Go2 task is also included as a reference environment. The core algorithm is shared between the two robots.

## Why MoE-CTS

CTS splits training into a teacher and a student that share the same actor and critic:

- The teacher reads a complete, privileged observation and produces a latent representation.
- The student reads only onboard observation (with history) and is trained to reproduce the teacher latent.

At deployment time only the student path is used. The teacher never runs on the robot; it only supervises during training.

MoE-CTS replaces the student's plain MLP encoder with a Mixture-of-Experts encoder. Different experts can specialize in different locomotion modes (stance, swing, rolling on wheels, climbing), and a gating network selects among them. A load-balancing loss keeps the experts from collapsing to a single specialist.

## Algorithm

### Notation

| Symbol | Meaning |
| --- | --- |
| `o_priv` | privileged observation (critic group) |
| `o_on` | onboard observation with history (policy group) |
| `o_single` | single-frame onboard observation |
| `f_theta` | teacher encoder: `o_priv -> z_t` |
| `g_phi` | student MoE encoder: `o_on -> z_s` |
| `pi` | shared actor: `(z, o_single) -> a` |
| `V` | shared critic: `(z, o_priv) -> value` |
| `rho` | teacher environment ratio (0.75) |

### Teacher-student split

A rollout of `N` environments is divided into `rho * N` teacher environments and `(1 - rho) * N` student environments. Both sets use the same actor and critic, but compute the latent differently:

- teacher: `z_t = f_theta(o_priv)`
- student: `z_s = g_phi(o_on)`

### PPO objective

Both teacher and student samples are optimized with a clipped PPO objective.

Surrogate loss:

```math
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
```

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
```

Value loss:

```math
L^{VF} = \mathbb{E}_t\left[(V_\theta(s_t) - R_t)^2\right]
```

Total PPO loss:

```math
L^{PPO} = L^{CLIP} + c_1 L^{VF} - c_2\, H[\pi_\theta]
```

Advantages are computed with Generalized Advantage Estimation (GAE):

```math
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

The learning rate follows an adaptive KL schedule around `desired_kl = 0.01`.

### MoE student encoder

The student encoder is a Mixture-of-Experts layer:

```math
z_s = \sum_{i=1}^{N} w_i(x)\, e_i(x), \qquad w(x) = \mathrm{softmax}(g(x))
```

`g` is the gating network and `e_i` are the experts. The default configuration uses 8 experts.

To keep the experts balanced, the gating weights are regularized toward a uniform distribution:

```math
L^{LB} = \sum_{i=1}^{N} \left(\bar{w}_i - \frac{1}{N}\right)^2, \qquad \bar{w}_i = \mathbb{E}_x\left[w_i(x)\right]
```

The student encoder is trained with a separate optimizer and a distillation loss:

```math
L^{student} = \left\| z_t - z_s \right\|_2^2 + \alpha\, L^{LB}
```

In the implementation, `alpha` is `load_balance_coef = 0.01`.

### Default hyperparameters

| Parameter | Value |
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

## Project layout

```text
scripts/
  rsl_rl/                 train.py, play.py, cli_args.py, rsl_rl_utils.py
  tools/                  URDF/MJCF conversion and cleanup helpers
source/
  rsl_rl/                 modified RSL-RL with MoE-CTS
  robot_lab/              robot_lab extension: Legbot and Go2 tasks/assets
deploy/
  deploy_mujoco/          MuJoCo Sim2Sim deployment (Legbot and Go2)
resources/
  go2/                    Go2 model and scenes (public Unitree assets)
  legbot_wf/              Legbot-WF model and scenes (not included, see below)
```

The MoE-CTS code lives in:

- `source/rsl_rl/rsl_rl/algorithms/moe_cts.py`
- `source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py`
- `source/rsl_rl/rsl_rl/networks/moe.py`
- `source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py`
- `source/rsl_rl/rsl_rl/storage/rollout_storage_cts.py`

## Dependencies

- Python 3.11
- Isaac Lab `2.3.2.post1`
- PyTorch `2.7.0`, torchvision `0.22.0`
- RSL-RL `3.3.0` (customized, installed from `source/rsl_rl`)
- robot_lab `2.3.0` (customized, installed from `source/robot_lab`)
- tensordict, numpy, onnx, onnxscript, GitPython
- MuJoCo and pygame (optional, for Sim2Sim)

## Installation

1. Install Isaac Lab:

```bash
conda create -n wf_cts_moe python=3.11
conda activate wf_cts_moe
pip install --upgrade pip
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

2. Install the customized RSL-RL and robot_lab in editable mode:

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

3. Install MuJoCo (optional, for Sim2Sim):

```bash
pip install mujoco pygame
```

## Usage

### Train

```bash
python scripts/rsl_rl/train.py \
  --task=RobotLab-Legbot-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --run_name v1
```

Checkpoints are written to `logs/rsl_rl/legbot_wf_moe_cts/<run>/model_<iter>.pt`.

### Play and export

```bash
python scripts/rsl_rl/play.py \
  --task=RobotLab-Legbot-v0 \
  --checkpoint=/absolute/path/to/model_xxx.pt
```

`play.py` also exports the student policy to `<run>/exported/policy.pt` and `policy.onnx`.

### MuJoCo Sim2Sim

```bash
# Validate the model and config without opening a window
python deploy/deploy_mujoco/deploy_legbot.py --validate

# Run the exported policy in MuJoCo
python deploy/deploy_mujoco/deploy_legbot.py \
  --policy=/absolute/path/to/exported/policy.pt

# Choose a terrain: flat (default), stairs, rough, mixed
python deploy/deploy_mujoco/deploy_legbot.py \
  --terrain=mixed \
  --policy=/absolute/path/to/exported/policy.pt
```

Add `--headless --duration 5` for a short smoke test without rendering.

### Go2 reference task

```bash
python scripts/rsl_rl/train.py --task=RobotLab-Go2-v0 --headless
python scripts/rsl_rl/play.py --task=RobotLab-Go2-v0
```

## Configuration

Task settings live in two places:

- Environment: `source/robot_lab/robot_lab/tasks/legbot/env_cfg.py`
- Algorithm: `source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py`

The task is registered as `RobotLab-Legbot-v0` in `source/robot_lab/robot_lab/tasks/legbot/__init__.py`.

Runtime overrides are available on the command line: `--num_envs`, `--max_iterations`, `--run_name`, `--experiment_name`, `--checkpoint`, `--seed`, `--logger`.

## Model and weights notice

The Legbot-WF model is proprietary and is **not** included in this repository:

- `resources/legbot_wf/` (URDF, meshes, scenes) is excluded by `.gitignore`.
- Trained checkpoints and exported policies (`.pt`, `.onnx`, `.pth`, `.ckpt`) are excluded.

To use the Legbot task, place your own robot model at `resources/legbot_wf/` and train from scratch. The Go2 model under `resources/go2/` is included and uses public Unitree assets.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

The repository contains code derived from open-source projects; their licenses remain in the corresponding source files and under `source/rsl_rl/licenses`.

## Acknowledgement

This project builds on:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym)

Algorithm reference:

- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/abs/2405.10830)
