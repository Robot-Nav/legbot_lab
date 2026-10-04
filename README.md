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
- A single-frame policy observation of 57 values with a 4-frame history (228 inputs), matching the W1W controller.
- 16 actions per step in hip, thigh, calf, wheel order for each leg (leg position offsets and wheel velocities).

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

## Wheel-foot training configuration tables

This section documents the code reviewed on 2026-10-04 for `RobotLab-Legbot-Simplified-v0` and the default `RobotLab-Legbot-v0`, using `resources/w1w_wf/urdf/w1w_wf.urdf`. `RobotLab-Legbot-Original-v0` uses the original mesh model. For command-line overrides, consult the run's `params/env.yaml` and `params/agent.yaml`.

### Algorithm and training parameters

The algorithm combines **PPO, concurrent teacher-student training, and a student MoE encoder**. The teacher learns a latent from privileged information; the student estimates it from onboard observation history. Both share the actor. Deployment uses the student branch.

| Item | Configuration |
| --- | --- |
| Algorithm / runner / policy | `MoECTS` / `OnPolicyRunnerCTS` / `ActorCriticMoECTS` |
| Teacher encoder | 295 privileged inputs; hidden layers `[512,256]`; 32 latent outputs |
| Student encoder | 228 history inputs; 8 experts; configured hidden dimensions `[512,256,256]` |
| Expert combination | Softmax gating and weighted sum; L2-normalized latent |
| Actor | 32 latent + 57 current observations = 89 inputs; hidden layers `[512,256,128]`; 16 action means |
| Critic | 295 privileged observations + 32 latent = 327 inputs; hidden layers `[512,256,128]`; one state value |
| Activation | ELU |
| Teacher / student environments | 75% / 25%; 1536 / 512 at 2048 environments |
| PPO updates | Teacher encoder, shared actor, critic, action standard deviation |
| Student encoder updates | Match teacher latents on student-environment states; detached teacher targets and separate optimizer |
| Student loss | Latent MSE + 0.01 × MSE between mean expert gating weights and uniform usage |
| Training / inference actions | Gaussian samples during training, initial standard deviation 1.0; student action means at inference |
| Default environment count | 2048 |
| Rollout per iteration | 24 steps per environment; 49152 samples |
| PPO epochs / mini-batches | 5 / 4 |
| PPO clipping / value loss | Clip parameter 0.2; value coefficient 1.0; clipped value loss enabled |
| Discount γ / GAE λ | 0.99 / 0.95 |
| PPO / student encoder learning rates | Both initially 0.001; PPO adapts to KL with target 0.01 |
| Entropy coefficient / gradient norm limit | 0.01 / 1.0 |
| Optimizer | Adam, β=`(0.9,0.999)`, weight decay=0 |
| Physics / control periods | 0.005 s (200 Hz) / 0.02 s (50 Hz), decimation=4 |
| Maximum episode duration | 25 s, or 1250 control steps |
| Save interval / default training limit | Every 50 iterations / 300000 iterations; the earlier run overrides the limit to 100000 |

### Reward terms: 14

Each control step receives `r_t = 0.02 × Σ(w_i × f_i)`. The expressions below are evaluated before multiplying by the weight and time step. `q₀` denotes the default joint pose; `a` denotes the action manager's action.

| Name | Quantity | Weight | Purpose / scope |
| --- | --- | --- | --- |
| `track_lin_vel_xy_exp` | `exp(−squared horizontal velocity error/0.25)` | +2.0 | Track body-frame vx and vy commands |
| `track_ang_vel_z_exp` | `exp(−squared yaw-rate error/0.25)` | +1.0 | Track yaw-rate command |
| `lin_vel_z_l2` | Squared body-frame vertical velocity | −2→−1 | Suppress vertical motion; linear schedule over the first 1500 iterations |
| `ang_vel_xy_l2` | Sum of squared roll and pitch rates | −0.05 | Suppress body oscillation |
| `joint_acc_l2` | Sum of squared accelerations of all 16 joints | −1e−7 | Penalize large joint accelerations |
| `joint_power` | `Σabs(τ×q̇)` | −2e−5 | Absolute mechanical power across all 16 joints |
| `joint_torques_l2` | Sum of squared torques of all 16 joints | −1e−4 | Penalize large torques |
| `base_height_l2` | `(height above ground−0.49)²` | −1→−10 | Maintain body height; linear schedule over the first 5000 iterations |
| `action_rate_l2` | `Σ(aₜ−aₜ₋₁)²` | −0.01 | Penalize action changes |
| `action_smoothness_l2` | `Σ(aₜ−2aₜ₋₁+aₜ₋₂)²` | −0.01 | Penalize second differences; active when episode step count exceeds 2 |
| `undesired_contacts` | Number of thigh/calf bodies with contact force above 5 N | −1.0 | Uses the maximum over contact history |
| `joint_pos_limits` | Sum of distances beyond soft position limits | −2.0 | Only the 12 leg joints; excludes wheels |
| `hip_pos_penalty_l1` | Sum of absolute deviations from q₀ for 4 hip joints | −0.05 | Limit hip displacement |
| `joint_pos_penalty_l1` | Sum of absolute deviations from q₀ for 8 thigh/calf joints | −0.01 | Limit displacement from the default leg pose |

- Height above ground is base world height minus mean ground height from the small scanner.
- Both pose-deviation terms use `stand_still_scale=1`; stationary and moving branches therefore have equal multipliers.
- Thigh/calf contacts incur penalties; base contacts can terminate the episode.
- No separate wheel-slip, gait-phase, foot-clearance, or bilateral-symmetry reward is enabled.
- `Episode return` is total episode reward. TensorBoard `Episode_Reward/*` divides each accumulated episode term by the maximum episode duration of 25 s; these are different statistics.

### Domain and initial-state randomization

Ranges use uniform sampling. Startup randomization is assigned during environment initialization and is not resampled every episode.

| Item | Timing | Range / operation | Scope / notes |
| --- | --- | --- | --- |
| Base mass | Startup | Add ±1 kg to nominal mass | Base; recompute inertia |
| Other body masses | Startup | Multiply nominal masses by 0.9–1.1 | Non-base bodies; recompute inertia |
| Base center of mass | Startup | Offset each of x, y, z by ±0.03 m | Base |
| Static friction | Startup | 0–2 | All robot collision bodies |
| Dynamic friction | Startup | 0–2, constrained not to exceed static friction | `make_consistent=True` |
| Restitution | Startup | 0–0.5 | 64 material buckets shared with friction sampling |
| Initial joint positions | Reset | Multiply default angles by 0.9–1.1 | Zero default angles remain zero |
| Initial joint velocities | Reset | 0 | No initial joint-velocity perturbation |
| Kp | Reset | Multiply nominal gains by 0.9–1.1 | Wheel Kp remains zero |
| Kd | Reset | Multiply nominal gains by 0.9–1.1 | All joints |
| Motor zero offsets | Reset | ±0.035 rad | Position-target offsets for the 12 leg joints |
| Initial horizontal position | Reset | x and y each ±0.5 m | Relative to default position and environment origin |
| Initial height | Reset | 0.53 m + 0–0.05 m | Also add environment-origin height |
| Initial orientation | Reset | Yaw ±3.14 rad | No random roll/pitch angle offsets |
| Initial linear velocity | Reset | Each axis ±0.5 m/s | Root velocity |
| Initial angular velocity | Reset | Each axis ±0.5 rad/s | Distinct from initial orientation |
| Horizontal pushes | Every 4 s | Add ±0.4 m/s to world-frame vx and vy | Zero z-velocity increment |
| Angular pushes | Every 4 s | Add ±0.6 rad/s on each world-frame angular axis | Implemented as a velocity jump |
| Observation noise | When student observations are generated | See observation table | No privileged-observation noise |

Random action delay, packet loss, independent motor-strength randomization, gravity randomization, and independent inertia randomization are not enabled. Inertia follows mass changes. The locally used `push_by_setting_velocity` implementation adds perturbations to current velocity.

### Student observations: 57 values × 4 frames = 228

Processing order: raw values → noise → clipping → scaling → history concatenation. All terms below are clipped to `[-100,100]` before scaling.

| Term | Single-frame size | Meaning | Scale | Uniform noise before scaling |
| --- | --- | --- | --- | --- |
| `base_ang_vel` | 3 | Body-frame angular velocity | 0.25 | ±0.2 rad/s |
| `projected_gravity` | 3 | Gravity direction in body frame | 1 | ±0.05 |
| `velocity_commands` | 3 | vx, vy, yaw-rate commands | `(2,2,0.25)` | None |
| `joint_pos` | 16 | Positions relative to default pose | 1 | Leg joints ±0.03 rad; wheel columns are zeroed afterward |
| `joint_vel` | 16 | Velocities relative to default velocities | 0.05 | ±2 rad/s |
| `actions` | 16 | Previous action | 1 | None |
| Total | 57 | Four history frames | 228 inputs | — |

| Group / rule | Meaning |
| --- | --- |
| `policy` | 228 history values consumed by the student MoE encoder |
| `single_obs` | 57 values matching the newest policy-history frame, consumed by the shared actor |
| History layout | Four frames per observation term, then concatenate terms; not a simple sequence of full frames |
| Wheel positions | Four columns are retained but held at zero, including removal of their noise |
| Student information | No direct true base linear velocity, torque, contact-force, or terrain-scan inputs |

### Privileged observations: 295 values

Used by the teacher encoder and critic, without observation noise or four-frame stacking of the whole group.

| Term | Size | Scale | Notes |
| --- | --- | --- | --- |
| Base linear velocity | 3 | 2.0 | Body frame |
| Base angular velocity | 3 | 0.25 | Body frame |
| Projected gravity | 3 | 1.0 | Gravity direction |
| Velocity commands | 3 | 1.0 | Different scale from student commands |
| Relative joint positions | 16 | 1.0 | Wheel positions zeroed |
| Relative joint velocities | 16 | 0.05 | All joints |
| Previous action | 16 | 1.0 | Motor order |
| Joint accelerations | 16 | 0.0001 | All joints |
| Applied torques | 16 | 0.01 | All joints |
| Wheel contact forces | 16 | 0.001 | 4 wheels × (3 force-magnitude history frames + historical maximum) |
| Terrain height scan | 187 | 2.5 | 1.6×1.0 m area, 0.1 m grid |
| Total | 295 | — | Cross-checked against the training startup log |

Height scans are clipped to `[-1,1]` before scaling; other terms are clipped to `[-100,100]` before scaling.

### Actions and actuators

There are 16 continuous actions. The training wrapper first clips network actions to `[-10,10]`.

| Action type | Size | Target | Control |
| --- | --- | --- | --- |
| Hip | 4 | `q_target = default_angle + zero_offset + 0.25a` | Position PD |
| Thigh | 4 | Same as above | Position PD |
| Calf | 4 | Same as above | Position PD |
| Foot/wheel | 4 | `ω_target = clip(20a,−150,150)` rad/s | Velocity servo |

Leg action terms additionally clip position targets to `[-100,100]` rad. This is distinct from the physical URDF joint limits.

| Action indices (zero-based) | Joint order |
| --- | --- |
| 0–3 | Front left: hip, thigh, calf, foot |
| 4–7 | Front right: hip, thigh, calf, foot |
| 8–11 | Rear left: hip, thigh, calf, foot |
| 12–15 | Rear right: hip, thigh, calf, foot |

| Joint | Nominal Kp | Nominal Kd | Torque limit | Simulation velocity limit |
| --- | --- | --- | --- | --- |
| Hip | 100 | 2 | 120 N·m | 15 rad/s |
| Thigh | 100 | 2 | 120 N·m | 15 rad/s |
| Calf | 100 | 2 | 175.38 N·m | 10.26 rad/s |
| Foot/wheel | 0 | 1 | 28.68 N·m | 104.72 rad/s |

The ±150 rad/s wheel target clip and the 104.72 rad/s simulation velocity limit are different settings.

| Default pose joint | Left legs | Right legs |
| --- | --- | --- |
| Hip | 0 | 0 |
| Thigh | −0.68 rad | +0.68 rad |
| Calf | +1.40 rad | −1.40 rad |
| Foot | 0 | 0 |

Left/right signs follow the URDF joint-axis directions.

### Termination conditions: 2

| Condition | Criterion | Type | Result |
| --- | --- | --- | --- |
| `time_out` | 1250 steps, or 25 s | Time truncation | Reset; learner handles timeout value bootstrapping |
| `illegal_contact` | Base net-contact-force magnitude exceeds 10 N in the three-frame history | Failure termination | Reset environment |

| Other condition | Current handling |
| --- | --- |
| Thigh/calf contact | Penalty above 5 N; no direct termination |
| Joint limits | Soft-limit penalty and physical limits |
| Excessive tilt | No separate orientation termination |
| Low base height | Height penalty; no separate height termination |
| Poor velocity tracking | Lower tracking reward |
| Wheel slip | No separate termination |

Base contact detection uses net force without distinguishing the contacting object.

### Command curriculum and terrain

| Training iterations | vx (m/s) | vy (m/s) | Yaw rate (rad/s) |
| --- | --- | --- | --- |
| 0–19999 | ±0.5 | ±0.5 | ±1.0 |
| 20000–49999 | ±1.0 | ±1.0 | ±1.5 |
| 50000–74999 | ±2.0 | ±1.0 | ±2.0 |
| 75000 onward | ±3.0 | ±1.0 | ±3.14 |

These are global command ranges; individual terrains impose additional limits.

| Terrain | Configured proportion |
| --- | --- |
| Waves | 5% |
| Uphill slopes | 10% |
| Downhill slopes | 10% |
| Rough slopes | 5% |
| Ascending stairs | 25% |
| Descending stairs | 10% |
| Obstacles | 20% |
| Flat | 15% |
| Stepping stones, gaps | 0% |

There are 10 terrain levels, with initial levels capped at 5. Levels change according to locomotion progress. Further evaluation should cover student performance under fixed test conditions, action delay, wheel slip, and the effect of abrupt command-curriculum transitions.

Sources: [environment](source/robot_lab/robot_lab/tasks/legbot/env_cfg.py), [algorithm configuration](source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py), [algorithm implementation](source/rsl_rl/rsl_rl/algorithms/moe_cts.py), [rewards](source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py), [observation alignment](source/robot_lab/robot_lab/tasks/legbot/env/observations.py), [actuators](source/robot_lab/robot_lab/assets/legbot.py), [terrain](source/robot_lab/robot_lab/tasks/go2/mdp/terrains.py).

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
  w1w_wf/                 Legbot-WF model and scenes (local private assets, excluded from Git)
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
  --num_envs 2048 \
  --max_iterations 300000 \
  --run_name v1
```

Checkpoints are written to `logs/rsl_rl/w1w_wf_moe_cts/<run>/model_<iter>.pt`.

### Play and export

```bash
python scripts/rsl_rl/play.py \
  --task=RobotLab-Legbot-v0 \
  --num_envs 1 \
  --checkpoint=/absolute/path/to/model_xxx.pt
```

`play.py` also exports the student policy to `<run>/exported/policy.pt` and `policy.onnx`. The training interface now matches `w1w_deploy`; older checkpoints need retraining, and new exports still require simulation and supported-robot validation.

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

- `resources/w1w_wf/` (URDF, meshes, scenes) is excluded by `.gitignore`.
- Trained checkpoints and exported policies (`.pt`, `.onnx`, `.pth`, `.ckpt`) are excluded.

To use the Legbot task, place your own robot model at `resources/w1w_wf/` and train from scratch. W1W deployment source is tracked; its virtual environment, build outputs, and policy weights are excluded. The Go2 model under `resources/go2/` is included and uses public Unitree assets.

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

`RobotLab-Legbot-v0` and `RobotLab-Legbot-Simplified-v0` both use `resources/w1w_wf/urdf/w1w_wf.urdf`, with 2048 environments by default. `RobotLab-Legbot-Original-v0` retains the original mesh model and a separate log directory. The earlier original-model smoke test verified execution only, not policy quality. CUDA 700 has also occurred with 2048 environments; reducing concurrency is not a verified cure.
