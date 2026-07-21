<h1 align="center">Legbot Lab</h1>

<p align="center">
  <a href="https://www.linux.org/"><img src="https://img.shields.io/badge/Platform-Linux-orange" alt="Platform"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img src="https://img.shields.io/badge/Isaac_Lab-2.2.0-green" alt="Isaac Lab"></a>
  <a href="https://arxiv.org/abs/1707.06347"><img src="https://img.shields.io/badge/RL-PPO-red" alt="PPO"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
</p>

<p align="center">
  English | <a href="README.md">简体中文</a>
</p>

---

> A reinforcement learning training and Sim2Real deployment project for the Legbot quadruped robot, built on NVIDIA Isaac Lab.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Algorithm](#algorithm)
  - [PPO Overview](#ppo-overview)
  - [Mathematical Formulation](#mathematical-formulation)
  - [Network Architecture](#network-architecture)
- [Project Structure](#project-structure)
- [Robot Parameters](#robot-parameters)
- [Quick Start](#quick-start)
  - [1. Install Isaac Lab](#1-install-isaac-lab)
  - [2. Install This Project](#2-install-this-project)
  - [3. Start Training](#3-start-training)
  - [4. Policy Inference and Export](#4-policy-inference-and-export)
- [Sim2Real Deployment](#sim2real-deployment)
  - [System Architecture](#system-architecture)
  - [Build and Run](#build-and-run)
- [MDP Definition](#mdp-definition)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Privileged Critic Observations](#privileged-critic-observations)
  - [Termination Conditions](#termination-conditions)
- [Reward Design](#reward-design)
- [Domain Randomization](#domain-randomization)
- [Training Parameters](#training-parameters)
- [Safety Mechanisms](#safety-mechanisms)
- [Technology Stack](#technology-stack)
- [Project Team](#project-team)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Legbot Lab is a complete reinforcement learning (RL) training and deployment framework for quadruped robots. Its goal is to train high-performance locomotion policies in simulation and deploy them seamlessly on a real robot through Sim2Real transfer.

**Key features:**

- **Complete Sim2Real workflow**: training → ONNX export → C++ controller deployment → real-robot execution
- **PPO reinforcement learning**: Proximal Policy Optimization implemented with RSL-RL
- **DDS communication architecture**: decouples the controller from the hardware and allows simulation and the real robot to share the same C++ control code
- **Extensive domain randomization**: randomization of mass, inertia, friction, PD gains, external disturbances, and other parameters
- **Safety mechanisms**: multi-stage clipping, limit protection, and a safety finite-state machine (FSM)
- **4,096 parallel environments**: GPU-accelerated training that fully utilizes the parallel simulation capability of Isaac Sim

---

## Algorithm

### PPO Overview

This project uses **Proximal Policy Optimization (PPO)** to train the locomotion policy of the Legbot quadruped robot. PPO was introduced by OpenAI in 2017 and is an Actor-Critic reinforcement learning algorithm based on policy gradients.

The core idea of PPO is to **clip the policy update magnitude**, thereby constraining the difference between the old and new policies. This improves training stability while maintaining good sample efficiency.

### Mathematical Formulation

#### 1. Clipped Surrogate Objective

PPO uses an importance-sampling ratio to measure the probability change assigned by the old and new policies to the same action:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

The clipped surrogate objective is:

$$
L^{\mathrm{CLIP}}(\theta)=\hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\mathrm{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t\right)\right]
$$

where:

- $r_t(\theta)$ is the importance-sampling ratio;
- $\hat{A}_t$ is the estimated advantage;
- $\epsilon$ is the clipping parameter, set to **0.2** in this project.

#### 2. Generalized Advantage Estimation (GAE)

To avoid confusion with the importance-sampling ratio $r_t(\theta)$, $R_t$ denotes the immediate reward at time step $t$ in the following equations.

The temporal-difference residual is:

$$
\delta_t=R_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

The generalized advantage estimate is:

$$
\hat{A}_t^{\mathrm{GAE}}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

where:

- $\delta_t$ is the TD residual;
- $\gamma$ is the discount factor, set to **0.99**;
- $\lambda$ is the GAE parameter, set to **0.95**.

#### 3. Value Function Loss

The value network is supervised using a target value:

$$
\hat{V}_t=\hat{A}_t+V_\phi(s_t)
$$

The value function loss is:

$$
L^{\mathrm{VF}}(\phi)=\hat{\mathbb{E}}_t\left[\frac{1}{2}\left(V_\phi(s_t)-\hat{V}_t\right)^2\right]
$$

#### 4. PPO Objective and Training Loss

From the perspective of objective maximization, the combined PPO objective can be written as:

$$
J(\theta,\phi)=L^{\mathrm{CLIP}}(\theta)-c_1L^{\mathrm{VF}}(\phi)+c_2H\left[\pi_\theta(\cdot\mid s_t)\right]
$$

In practice, gradient descent is usually used, so the total loss to be minimized is:

$$
L_{\mathrm{total}}(\theta,\phi)=-L^{\mathrm{CLIP}}(\theta)+c_1L^{\mathrm{VF}}(\phi)-c_2H\left[\pi_\theta(\cdot\mid s_t)\right]
$$

The policy entropy term encourages exploration. The coefficients used in this project are:

- Value loss coefficient: c₁ = 1.0;
- Entropy coefficient: c₂ = 0.01.

#### 5. Adaptive Learning Rate Based on KL Divergence

PPO dynamically adjusts the learning rate according to the relationship between the measured KL divergence and the target KL divergence.

When the measured KL divergence is too large, the learning rate is reduced:

$$
D_{\mathrm{KL}}\gt 2D_{\mathrm{KL}}^{\mathrm{target}},\qquad
\alpha\leftarrow\max\left(\alpha_{\min},\frac{\alpha}{1.5}\right)
$$

When the measured KL divergence is too small, the learning rate is increased:

$$
0\lt D_{\mathrm{KL}}\lt 0.5D_{\mathrm{KL}}^{\mathrm{target}},\qquad
\alpha\leftarrow\min\left(\alpha_{\max},1.5\alpha\right)
$$

Otherwise, the learning rate remains unchanged:

$$
\alpha\leftarrow\alpha
$$

The target KL divergence is set to **0.01**. Here, $\alpha_{\min}$ and $\alpha_{\max}$ denote the lower and upper learning-rate bounds, respectively.

#### 6. PD Controller

The policy outputs desired joint positions, which are converted into joint torques by a PD controller:

$$
\tau=K_p(q_{\mathrm{des}}-q)+K_d(\dot{q}_{\mathrm{des}}-\dot{q})
$$

The actuator gains used during training are:

- $K_p=50$ N·m/rad;
- $K_d=3.0$ N·m·s/rad.

The FixStand state used during real-robot deployment applies:

- $K_p=60$ N·m/rad;
- $K_d=4.0$ N·m·s/rad.

PD gains for different runtime states should follow the corresponding training and deployment configuration files.

#### 7. Action Mapping

The raw Gaussian-policy output is first clipped element-wise:

$$
\widetilde{\mathbf{a}}=\min\left(\max\left(\mathbf{a},-100\right),100\right)
$$

The clipped action is then mapped to desired joint positions:

$$
\mathbf{q}_{\mathrm{des}}=\mathbf{q}_{\mathrm{default}}+0.25\widetilde{\mathbf{a}}
$$

Here, $\mathbf{a}\in\mathbb{R}^{12}$ is the raw output of the Gaussian policy network. Each action dimension is clipped to $[-100,100]$, multiplied by 0.25 rad, and added to the default joint position. Therefore, the policy output is not hard-bounded to $[-1,1]$ by a tanh function.

The default standing pose is:

- Hip joint: 0.0 rad;
- Thigh joint: 0.9 rad;
- Calf joint: -1.8 rad.

### Network Architecture

| Component | Architecture |
|---|---|
| Actor policy network | MLP: `[actor_obs_dim] → 512 → 256 → 128 → 12`, with ELU activations |
| Critic value network | MLP: `[critic_obs_dim] → 512 → 256 → 128 → 1`, with ELU activations |
| Policy distribution | Gaussian policy with an initial action-noise standard deviation of 1.0 |

A single policy observation contains **45 dimensions**. With `history_length=10`, `concatenate_terms=True`, and `flatten_history_dim=True`, ten observation frames are flattened and concatenated, resulting in an actual Actor input dimension of $45\times10=450$.

---

## Project Structure

```text
legbot_lab/
├── README.md                          # Chinese documentation
├── README_EN.md                       # English documentation
├── env_cfg.py                         # Main environment configuration for PPO training with 4,096 parallel environments
├── legbot_rl_lab/                     # Core RL training and deployment system
│   ├── source/unitree_rl_lab/         # Python training source code based on Isaac Lab
│   │   └── unitree_rl_lab/
│   │       ├── assets/robots/         # Legbot robot configuration
│   │       │   ├── unitree.py         # UNITREE_LEGBOT_CFG
│   │       │   └── unitree_actuators.py # Actuator torque-speed curve model
│   │       ├── tasks/locomotion/
│   │       │   ├── agents/            # PPO algorithm configuration
│   │       │   ├── mdp/               # MDP components: rewards, observations, randomization, and commands
│   │       │   └── robots/legbot/     # Legbot environment configuration
│   │       └── utils/                 # Utility functions, including deployment-configuration export
│   ├── scripts/                       # Training, inference, and testing scripts
│   │   └── rsl_rl/
│   │       ├── train.py               # Training entry point with headless support
│   │       ├── play.py                # Policy inference and ONNX export
│   │       └── cli_args.py            # Command-line arguments
│   ├── deploy/                        # C++ deployment code for Sim2Real
│   │   ├── include/
│   │   │   ├── FSM/                   # Finite-state machine
│   │   │   │   ├── CtrlFSM.h          # 1 kHz main state machine
│   │   │   │   ├── State_FixStand.h   # Fixed-standing state
│   │   │   │   ├── State_Passive.h    # Passive damping state
│   │   │   │   └── State_RLBase.h     # RL policy execution state
│   │   │   ├── deploy_safety.h        # Safety protection for torque, temperature, and attitude limits
│   │   │   ├── deploy_csv_logger.h    # 50 Hz CSV diagnostic logger
│   │   │   └── param.h                # Command-line argument parser
│   │   └── robots/legbot/
│   │       ├── config/config.yaml     # FSM configuration and safety parameters
│   │       ├── include/Types.h        # DDS interface definitions
│   │       ├── main.cpp               # Controller entry point
│   │       └── src/State_RLBase.cpp   # ONNX Runtime inference
│   ├── unitree_ros/robots/legbot_description/ # URDF and MJCF models
│   └── logs/rsl_rl/unitree_legbot_velocity/   # Training logs and models
├── simulate/                          # MuJoCo DDS simulator
│   ├── src/
│   │   ├── main.cc                    # Main simulation loop
│   │   ├── legbot_bridge.h            # DDS-to-MuJoCo bridge
│   │   └── physics_joystick.h         # Gamepad and keyboard input
│   └── config.yaml                    # Simulation configuration
├── serial_dds_gateway/                # Serial-to-DDS hardware gateway
│   ├── src/legbot_rt_gait_pd.cpp      # Main gateway program at 500 Hz
│   ├── include/                       # IMU frame parsing and motor protocol
│   └── start_gateway.sh               # One-click launch script
├── legbot/                            # NumPy training environment for an independent navigation task
│   ├── cfg.py                         # Scenario configuration
│   ├── legbot_section001_np.py        # Training entry point
│   └── xmls/
│       ├── legbot.xml                 # Legbot MJCF model
│       ├── scene_stairs.xml           # Stair scenario
│       └── scene_world.xml            # Complete track scenario
├── terrain_tool/                      # Terrain-generation tools
└── unitree_sdk2/                      # Unitree DDS communication library
```

---

## Robot Parameters

The Legbot URDF model is located at [legbot_rl_lab/unitree_ros/robots/legbot_description/urdf/legbot_description.urdf](legbot_rl_lab/unitree_ros/robots/legbot_description/urdf/legbot_description.urdf).

### Mass Distribution

| Component | Mass (kg) | Quantity | Total Mass (kg) |
|---|---:|---:|---:|
| Base | 6.584 | 1 | 6.584 |
| Hip joint | 0.080 | 4 | 0.319 |
| Thigh | 1.550 | 4 | 6.202 |
| Calf | 0.184 | 4 | 0.736 |
| Foot | 0.040 | 4 | 0.160 |
| **Total** |  |  | **~14.0** |

### Robot Specifications

| Parameter | Value |
|---|---|
| Robot name | Legbot |
| Motor model | RobStride RS02, 12 motors |
| Degrees of freedom | 12, with hip/thigh/calf joints on each leg |
| Standing base height | 0.28 m, with a center-of-mass height of 0.277 m |
| Total mass | ~14.0 kg |
| Base mass | 6.584 kg |
| Thigh length | 0.1985 m |
| Calf length | 0.214 m |
| Foot radius | 0.021 m |

### Motor Parameters

| Joint | Peak Torque | Peak Speed | Gear Ratio | Notes |
|---|---:|---:|---:|---|
| Hip | ±16 N·m | 30 rad/s | 1:1 | |
| Thigh | ±16 N·m | 30 rad/s | 1:1 | |
| Calf | ±32 N·m | 15.7 rad/s | 1:2 | With a 1:2 reduction ratio, model-space angles must be multiplied by 2 when converted to motor-space angles |

### Joint Limits

| Joint | Lower Limit (rad) | Upper Limit (rad) |
|---|---:|---:|
| FR/FL Hip | -0.733 | 0.733 |
| FR/FL Thigh | -1.559 | 3.130 |
| RR/RL Hip | -0.733 | 0.733 |
| RR/RL Thigh | -0.512 | 4.177 |
| All Calf joints | -2.639 | -0.785 |

### Hardware Configuration

| Parameter | Value |
|---|---|
| Onboard computer | Orange Pi 6, aarch64 |
| Sensors | Custom serial IMU with accelerometer and gyroscope |
| Communication bus | 2 × USB-CAN plus USB-to-serial for the IMU |
| Control frequency | 1 kHz controller / 500 Hz gateway |
| Middleware | CycloneDDS over the local `lo` loopback interface |

---

## Quick Start

### 1. Install Isaac Lab

Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### 2. Install This Project

```bash
# Clone the repository
git clone https://github.com/Robot-Nav/legbot_lab.git
cd legbot_lab

# Activate the Isaac Lab Conda environment
conda activate env_isaaclab

# Install this project in editable mode
pip install -e legbot_rl_lab/source/unitree_rl_lab
```

### 3. Start Training

#### 3.1 Main PPO Training with 4,096 Parallel Environments

```bash
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --headless \
    --num_envs 4096 \
    --max_iterations 50000
```

> The default maximum number of iterations in the PPO configuration file is **100000**. The command above overrides it with `--max_iterations 50000` for the current training run.

**Command-line arguments:**

| Argument | Description | Default |
|---|---|---|
| `--task` | Training task name | Unitree-Legbot-Velocity |
| `--headless` | Disable rendering to accelerate training | False |
| `--num_envs` | Number of parallel environments | 4096 |
| `--max_iterations` | Maximum number of training iterations | 100000 in the configuration; overridden to 50000 in the example |
| `--seed` | Random seed | Random |
| `--resume` | Resume from a checkpoint | False |
| `--load_run` | Run directory to load when resuming | - |
| `--checkpoint` | Checkpoint filename to load | - |

#### 3.2 Resume Interrupted Training

```bash
# Automatically resume from the latest checkpoint
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --resume \
    --headless

# Manually specify a run directory and checkpoint
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --resume \
    --load_run 2026-06-25_17-11-16 \
    --checkpoint model_108.pt \
    --headless
```

#### 3.3 Train the NumPy Navigation Environment

```bash
python legbot/legbot_section001_np.py
```

### 4. Policy Inference and Export

```bash
# Run inference and automatically export the policy to ONNX
python legbot_rl_lab/scripts/rsl_rl/play.py --task Unitree-Legbot-Velocity
```

The following files are automatically exported to `logs/rsl_rl/unitree_legbot_velocity/<run_name>/exported/`:

- `policy.pt` — TorchScript model for Python inference
- `policy.onnx` — ONNX model for C++ deployment
- `deploy.yaml` — Deployment configuration, including joint mapping and scaling factors

---

## Sim2Real Deployment

### System Architecture

```text
┌────────────────────┐  DDS (rt/lowcmd, rt/lowstate)  ┌──────────────────────┐  Serial   ┌──────────────┐
│   legbot_ctrl       │◄──────────────────────────────►│  serial_dds_gateway  │◄────────►│ 12 Motors   │
│   RL Controller     │         CycloneDDS             │  DDS/Serial Gateway  │ type1-4  │ and IMU     │
└────────────────────┘                                └──────────────────────┘          └──────────────┘
         │                                                      │
   Runs on Orange Pi                                      Runs on Orange Pi
   1 kHz control loop                                     500 Hz protocol bridge
   └──────────── Shared loopback interface `lo` ────────────┘
```

**Core design principle:** the controller code is identical in simulation and on the real robot. The DDS abstraction layer allows switching between MuJoCo simulation and the `serial_dds_gateway` hardware interface by changing only the communication endpoint.

### Build and Run

#### 1. Build the Hardware Gateway

```bash
cd serial_dds_gateway
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)
```

#### 2. Build the Controller

```bash
cd legbot_rl_lab/deploy/robots/legbot
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)
```

#### 3. Run in Two Terminals

**Terminal 1 — Start the gateway first:**

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

**Terminal 2 — Start the controller after the gateway is ready:**

```bash
cd legbot_rl_lab/deploy/robots/legbot
./build/legbot_ctrl --network lo
```

#### 4. FSM State Transitions Using the Gamepad

```text
Passive Damping ──LT+A──► FixStand ──Start──► Velocity / RL Control
      ▲                       ▲                       │
      │                       │                       │
      └──────── LT+B ─────────┴──────── LT+B ────────┘
```

---

## MDP Definition

### Observation Space

| Observation | Dimension | Scale | Noise |
|---|---:|---:|---|
| Base angular velocity from the IMU gyroscope | 3 | 0.25 | U(-0.2, 0.2) |
| Projected gravity direction | 3 | 1.0 | U(-0.05, 0.05) |
| Velocity command | 3 | 1.0 | - |
| Joint position relative to the default pose | 12 | 1.0 | U(-0.03, 0.03) |
| Joint velocity | 12 | 0.05 | U(-2.0, 2.0) |
| Previous action | 12 | 1.0 | - |
| **Total** | **45** |  |  |

A single policy observation contains **45 dimensions**. Ten frames of observation history are flattened and concatenated, resulting in an Actor input dimension of **450**.

### Action Space

| Parameter | Value |
|---|---|
| Control mode | Joint-position control with PD |
| Raw policy output | $\mathbb{R}^{12}$, Gaussian policy without a hard output bound |
| Environment action clipping | $[-100,100]^{12}$ |
| Action scale | ×0.25 rad |
| Training actuator stiffness $K_p$ | 50 N·m/rad |
| Training actuator damping $K_d$ | 3.0 N·m·s/rad |
| FixStand deployment gains | $K_p=60$ N·m/rad and $K_d=4.0$ N·m·s/rad |
| Default hip angle | 0.0 rad |
| Default thigh angle | 0.9 rad |
| Default calf angle | -1.8 rad |

### Privileged Critic Observations

The Critic network receives information that is unavailable to the policy network, forming an asymmetric Actor-Critic architecture:

| Privileged Observation | Dimension | Description |
|---|---:|---|
| Base linear velocity | 3 | Ground-truth linear velocity |
| Joint acceleration | 12 | Joint acceleration |
| Joint torque | 12 | Ground-truth joint torque |
| Foot contact force | 4 | Normal force for each foot |
| Large-area height scan | 187 | 1.6 × 1.0 m terrain height map |

The Critic uses the current 45-dimensional base observation and an additional 218 dimensions of privileged information, resulting in a total Critic input dimension of **263**. The Critic does not use the Actor's ten-frame history.

### Termination Conditions

| Condition | Threshold |
|---|---|
| Timeout | Episode reaches 25 seconds |
| Fall | Base contact force exceeds 1.0 N |

---

## Reward Design

The total training reward is the weighted sum of the terms listed below. Positive rewards encourage velocity-command tracking, while negative rewards penalize undesirable motion.

### Positive Rewards

| Reward | Weight | Formula | Description |
|---|---:|---|---|
| Linear velocity tracking | +1.0 | $\exp(-((v_x-v_x^{cmd})^2+(v_y-v_y^{cmd})^2)/0.5)$ | Exponential reward for planar linear-velocity tracking |
| Angular velocity tracking | +0.5 | $\exp(-(\omega_z-\omega_z^{cmd})^2/0.5)$ | Exponential reward for yaw-rate tracking |

### Penalties

| Penalty | Weight | Description |
|---|---:|---|
| Vertical linear velocity L2 | -2.0 | Suppresses vertical oscillation |
| Horizontal angular velocity L2 | -0.05 | Suppresses roll and pitch motion |
| Base-height error L2 | -1.0 to -10.0 | Curriculum term that maintains a target height of 0.28 m |
| Joint acceleration L2 | -1e-7 | Encourages smooth motion |
| Joint power | -2e-5 | Reduces energy consumption |
| Joint torque L2 | -1e-4 | Discourages excessive torque |
| Action-rate L2 | -0.01 | Encourages smooth control commands |
| Action smoothness L2 | -0.01 | Penalizes the third derivative of the action |
| Undesired contact | -1.0 | Prevents the thighs and calves from touching the ground, with a 5 N threshold |
| Joint limit | -2.0 | Prevents motion outside the joint range |
| Foot regulation | -0.05 | Constrains foot height and spacing |
| Hip position L1 | -0.05 | Keeps the hip joints near zero while standing |
| Joint position L1 | -0.01 | Maintains the default pose while standing |

### Curriculum Learning

| Curriculum Item | Initial Value | Final Value | Iteration Range |
|---|---:|---:|---:|
| Vertical-velocity penalty weight | -2.0 | 0.0 | 0 → 1,500 |
| Height-error penalty weight | -1.0 | -10.0 | 0 → 5,000 |
| Terrain difficulty | Level 0 | Level 5 | Increases with training progress |

---

## Domain Randomization

To transfer the policy from simulation to the real robot, a wide range of parameters are randomized during training. Randomization is applied at startup, at every episode reset, or at fixed intervals depending on the parameter.

### Mass and Inertia Randomization

> Mode: `startup`, sampled once when training starts and kept fixed during the run.

| Parameter | Distribution | Operation | Description |
|---|---|---|---|
| Base mass | U(-1.0, 1.0) kg | Addition | Simulates payload changes of ±1 kg |
| Non-base link mass | U(0.9, 1.1) | Multiplication | Varies link mass by ±10% |
| Rotational inertia | U(0.9, 1.1) | Multiplication | Varies the inertia of all links by ±10% |
| Center-of-mass position | U(-0.05, 0.05) m | Offset | Shifts the base center of mass by ±5 cm along x, y, and z |

### Friction and Contact Randomization

> Mode: `startup`.

| Parameter | Range | Description |
|---|---|---|
| Static friction coefficient | U(0.0, 2.0) | 64 discrete friction combinations |
| Dynamic friction coefficient | U(0.0, 2.0) | Tied to the static friction setting |
| Restitution coefficient | U(0.0, 0.5) | Contact elasticity |

### Actuator Randomization

> Mode: `reset`, sampled at every episode reset.

| Parameter | Distribution | Description |
|---|---|---|
| PD stiffness $K_p$ | U(0.9, 1.1) | Multiplicative variation that models motor differences |
| PD damping $K_d$ | U(0.9, 1.1) | Multiplicative variation |
| Joint zero-position offset | U(-0.035, 0.035) rad | ±35 mrad, approximately ±2°, to simulate encoder error |

### Initial-State Randomization

> Mode: `reset`.

| Parameter | Distribution | Description |
|---|---|---|
| Initial joint position | U(0.5, 1.5) × default pose | 50% to 150% of the default pose |
| Base position | U(-0.5, 0.5) m in x and y | Random horizontal offset |
| Base height | U(0.0, 0.2) m offset | Random initial height |
| Base yaw | U(-π, π) | Random heading |
| Base linear velocity | U(-0.5, 0.5) m/s | Random initial linear velocity |
| Base angular velocity | U(-0.5, 0.5) rad/s | Random initial angular velocity |

### Periodic Push Disturbances

> Mode: `interval`, triggered every 4 seconds.

| Parameter | Distribution | Description |
|---|---|---|
| Linear-velocity disturbance in x and y | U(-0.4, 0.4) m/s | Simulates external pushes |
| Angular-velocity disturbance in roll, pitch, and yaw | U(-0.6, 0.6) rad/s | Simulates rotational disturbances |

### Observation Noise

> Mode: applied at every step to policy observations.

| Observation | Noise Distribution | Scale |
|---|---|---:|
| Base angular velocity | U(-0.2, 0.2) | 0.25 |
| Projected gravity | U(-0.05, 0.05) | 1.0 |
| Joint position | U(-0.03, 0.03) | 1.0 |
| Joint velocity | U(-2.0, 2.0) | 0.05 |

> Critic observations do not include observation corruption because `enable_corruption=False`, which helps reduce additional randomness in value estimation.

### Terrain Randomization

A procedural terrain generator creates rough terrain and supports curriculum learning. As `train_env_steps` increases, terrain difficulty progresses from Level 0, corresponding to flat ground, to Level 5, corresponding to complex rough terrain.

---

## Training Parameters

### PPO Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO, Proximal Policy Optimization |
| Learning rate | $1.0\times10^{-3}$, adaptive |
| Clipping parameter $\epsilon$ | 0.2 |
| Discount factor $\gamma$ | 0.99 |
| GAE parameter $\lambda$ | 0.95 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 1.0 |
| Maximum gradient norm | 1.0 |
| Steps per environment | 24 |
| Learning epochs | 5 |
| Number of mini-batches | 4 |
| Target KL divergence | 0.01 |
| Maximum iterations | 100,000 by default |

### Training Scale

| Parameter | Value |
|---|---|
| Parallel environments | 4,096 |
| Physics time step | 0.005 s, corresponding to 200 Hz |
| Control decimation | 4, corresponding to a 50 Hz control frequency |
| Episode length | 25 s, corresponding to 1,250 control steps |
| Samples per iteration | 4,096 × 24 = 98,304 |
| Maximum total training steps | 4,096 × 24 × 100,000 ≈ 9.83 × 10⁹ |

### Network Configuration Summary

| Parameter | Value |
|---|---|
| Hidden-layer dimensions | [512, 256, 128] |
| Activation function | ELU |
| Single-frame policy observation dimension | 45 |
| History length | 10 frames |
| Actual Actor input dimension | 450, from 45 × 10 flattened frames |
| Policy output dimension | 12 joint-position offsets |
| Critic observation dimension | 263 |

---

## Safety Mechanisms

### Command-Side Clipping and Limits

| Protection Item | Limit | Description |
|---|---:|---|
| Action clip | ±100 | Clips the raw policy output |
| Joint-angle limit | Configurable | Hard joint-position limit |
| Torque limit | ±40 N·m | Torque clipping |
| Angle-change limit | 0.05 rad/tick | Prevents abrupt neural-network output jumps |
| Velocity-change limit | 1.0 rad/s/tick | Prevents abrupt velocity changes |

### Feedback-Side Protection

When any of the following limits are exceeded, the controller switches to the Passive damping state.

| Monitored Item | Threshold | Action |
|---|---:|---|
| Communication timeout | - | Switch to Passive |
| Joint velocity | 30 rad/s | Switch to Passive |
| Measured torque | 45 N·m | Switch to Passive |
| Motor temperature | 80°C | Switch to Passive |
| IMU roll | ±0.5 rad, approximately ±28° | Switch to Passive |
| IMU pitch | ±0.5 rad, approximately ±28° | Switch to Passive |
| Emergency-stop flag | - | Switch to Passive |

---

## Technology Stack

| Category | Technology |
|---|---|
| Simulation platform | NVIDIA Isaac Sim 5.0.0 / Isaac Lab 2.2.0 |
| Physics engine | PhysX GPU with 4,096 parallel environments |
| RL algorithm | RSL-RL 2.3.1, PPO |
| Deep learning | PyTorch with CUDA acceleration |
| Model export | ONNX Runtime 1.22.0 |
| Deployment language | C++17 |
| Communication middleware | CycloneDDS |
| Robot model | URDF / MJCF |
| Configuration management | Hydra / YAML |
| Simulation validation | MuJoCo with a DDS bridge |
| Onboard computer | Orange Pi 6, aarch64 |
| Motors | RobStride RS02 |

---

## Project Team

This project is developed and maintained by the **Robot-Nav** team.

GitHub: [https://github.com/Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab)

---

## License

MIT License

---

## Citation

```bibtex
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

---

## Acknowledgments

We sincerely thank the **unitree_rl_lab** team for providing the open-source training framework, engineering implementation, and community support. Building on their work, this project adapts the environment and training configuration for the Legbot robot and implements simulation validation and Sim2Real deployment.
