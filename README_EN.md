# <p align="center"> Legbot Lab </p>



<p align="center">
  <a href="https://www.linux.org/"><img src="https://img.shields.io/badge/Platform-Linux-orange" alt="Platform"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img src="https://img.shields.io/badge/Isaac_Lab-2.2.0-green" alt="Isaac Lab"></a>
  <a href="https://arxiv.org/abs/1707.06347"><img src="https://img.shields.io/badge/RL-PPO-red" alt="PPO"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
</p>

<p align="center">
  English|<a href="README.md">简体中文</a>
</p>

---
> Reinforcement Learning Training & Sim2Real Deployment Framework for the Legbot Quadruped Robot Based on NVIDIA Isaac Lab
---

## Table of Contents

- [Introduction](#introduction)
- [Algorithm Principles](#algorithm-principles)
  - [PPO Overview](#ppo-overview)
  - [Algorithm Formulas](#algorithm-formulas)
  - [Network Architecture](#network-architecture)
- [Project Structure](#project-structure)
- [Robot Specifications](#robot-specifications)
- [Quick Start](#quick-start)
  - [1. Install Isaac Lab](#1-install-isaac-lab)
  - [2. Install This Project](#2-install-this-project)
  - [3. Start Training](#3-start-training)
  - [4. Model Inference & Export](#4-model-inference--export)
- [Sim2Real Deployment](#sim2real-deployment)
  - [System Architecture](#system-architecture)
  - [Build & Run](#build--run)
- [MDP Definition](#mdp-definition)
  - [Observation Space](#observation-space-policy-network)
  - [Action Space](#action-space)
  - [Critic Privileged Information](#critic-privileged-information)
  - [Termination Conditions](#termination-conditions)
- [Reward Function Design](#reward-function-design)
- [Domain Randomization](#domain-randomization)
- [Training Parameters](#training-parameters)
- [Safety Mechanisms](#safety-mechanisms)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## Introduction

Legbot Lab is a complete quadruped robot reinforcement learning (RL) training and deployment framework. The goal is to train high-performance locomotion policies entirely in simulation and seamlessly deploy them onto a physical robot (Sim2Real).

**Key Features:**

- **Closed-loop Sim2Real**: Training → ONNX Export → C++ Controllers Deployment → Physical Robot
- **PPO Reinforcement Learning**: Proximal Policy Optimization implemented on top of RSL-RL
- **DDS Communication Architecture**: Controller decoupled from hardware; simulation and real robot share identical C++ code
- **Extensive Domain Randomization**: Mass, inertia, friction, PD gains, external perturbations, and more
- **Safety Mechanisms**: Multi-level clipping, over-limit protection, finite state machine (FSM)
- **4,096 Parallel Environments**: GPU-accelerated training leveraging Isaac Sim's parallel simulation

---

## Algorithm Principles

### PPO Overview

This project uses **Proximal Policy Optimization (PPO)** to train locomotion policies for a quadruped robot. PPO was proposed by OpenAI in 2017 and is a policy-gradient-based Actor-Critic reinforcement learning algorithm.

The core idea of PPO is to **clip the policy update magnitude** to constrain the divergence between old and new policies, thereby stabilizing training while maintaining high sample efficiency.

### Algorithm Formulas

#### 1. Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the importance sampling ratio
- $\hat{A}_t$ is the advantage estimate
- $\epsilon$ is the clipping parameter (set to **0.2** in this project)

#### 2. Generalized Advantage Estimation (GAE)

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual:
- $\gamma$ is the discount factor (set to **0.99**)
- $\lambda$ is the GAE parameter (set to **0.95**)

#### 3. Value Function Loss

$$L^{VF}(\phi) = \hat{\mathbb{E}}_t \left[ \frac{1}{2} \left( V_\phi(s_t) - \hat{R}_t \right)^2 \right]$$

Where $\hat{R}_t = A_t + V_\phi(s_t)$ is the target return.

#### 4. Total Loss

$$L(\theta, \phi) = L^{CLIP}(\theta) - c_1 \cdot L^{VF}(\phi) + c_2 \cdot S[\pi_\theta](s_t)$$

Where $S[\pi_\theta]$ is the policy entropy that encourages exploration. This project uses:
- Value loss coefficient $c_1 = 1.0$
- Entropy coefficient $c_2 = 0.01$

#### 5. Adaptive Learning Rate (KL Divergence Control)

PPO uses an adaptive learning rate that dynamically adjusts based on KL divergence:

$$\text{if } KL > \text{desired\_kl} \times 2: \ \alpha \leftarrow \alpha \times 1.5$$
$$\text{if } KL < \text{desired\_kl} \times 0.5: \ \alpha \leftarrow \alpha \times 0.67$$

The target KL divergence is set to **0.01**.

#### 6. PD Controller

The policy outputs joint position targets, which are converted to torques through a PD controller:

$$\tau = K_p \cdot (q_{des} - q) + K_d \cdot (\dot{q}_{des} - \dot{q})$$

Where $K_p = 60$ N·m/rad and $K_d = 4.0$ N·m·s/rad.

#### 7. Action Mapping

$$q_{des} = q_{default} + 0.25 \cdot a$$

Where $a \in [-1, 1]^{12}$ is the policy network output and $q_{default}$ is the default standing pose:
- Hip joint: $0.0$ rad
- Thigh joint: $0.9$ rad
- Calf joint: $-1.8$ rad

### Network Architecture

| Component | Architecture |
|-----------|-------------|
| Actor (Policy Network) | MLP: `[obs_dim] → 512 → 256 → 128 → 12`, ELU activation |
| Critic (Value Network) | MLP: `[critic_obs_dim] → 512 → 256 → 128 → 1`, ELU activation |
| Initialization | Orthogonal initialization, initial noise std = 1.0 |

The policy network receives **10 frames of observation history** (`history_length=10`) to capture temporal dynamics.

---

## Project Structure

```
legbot_mujoco/
├── README.md                          # Documentation (Chinese)
├── README_EN.md                       # Documentation (English)
├── env_cfg.py                         # Main environment config (4096 parallel envs, PPO)
├── legbot_rl_lab/                     # Core RL training & deployment system
│   ├── source/unitree_rl_lab/         # Python training source (based on Isaac Lab)
│   │   └── unitree_rl_lab/
│   │       ├── assets/robots/         # Legbot robot configuration
│   │       │   ├── unitree.py         # UNITREE_LEGBOT_CFG
│   │       │   └── unitree_actuators.py # Actuator T-N curve model
│   │       ├── tasks/locomotion/
│   │       │   ├── agents/            # PPO algorithm configuration
│   │       │   ├── mdp/               # MDP components (rewards/observations/DR/commands)
│   │       │   └── robots/legbot/     # Legbot environment config
│   │       └── utils/                 # Utilities (deploy config export)
│   ├── scripts/                       # Training/Inference/Test scripts
│   │   └── rsl_rl/
│   │       ├── train.py               # Training entry (headless mode)
│   │       ├── play.py                # Inference / ONNX export
│   │       └── cli_args.py            # CLI arguments
│   ├── deploy/                        # C++ deployment code (Sim2Real)
│   │   ├── include/
│   │   │   ├── FSM/                   # Finite State Machine
│   │   │   │   ├── CtrlFSM.h          # 1kHz main state machine
│   │   │   │   ├── State_FixStand.h   # Standing state
│   │   │   │   ├── State_Passive.h    # Passive damped state
│   │   │   │   └── State_RLBase.h     # RL policy running state
│   │   │   ├── deploy_safety.h        # Safety protection (torque/temp/attitude)
│   │   │   ├── deploy_csv_logger.h    # 50Hz CSV diagnostic logger
│   │   │   └── param.h               # CLI parameter parsing
│   │   └── robots/legbot/
│   │       ├── config/config.yaml     # FSM config & safety parameters
│   │       ├── include/Types.h        # DDS interface definition
│   │       ├── main.cpp               # Controller entry
│   │       └── src/State_RLBase.cpp   # ONNX Runtime inference
│   ├── unitree_ros/robots/legbot_description/ # URDF & MJCF models
│   └── logs/rsl_rl/unitree_legbot_velocity/   # Training logs & models
├── simulate/                          # MuJoCo DDS simulator
│   ├── src/
│   │   ├── main.cc                    # Simulation main loop
│   │   ├── legbot_bridge.h            # DDS↔MuJoCo bridge
│   │   └── physics_joystick.h         # Gamepad/keyboard driver
│   └── config.yaml                    # Simulation config
├── serial_dds_gateway/                # Serial↔DDS hardware gateway
│   ├── src/legbot_rt_gait_pd.cpp      # Gateway main (500Hz)
│   ├── include/                       # IMU frame parsing / motor protocol
│   └── start_gateway.sh               # One-click startup script
├── legbot/                            # NumPy training env (independent navigation)
│   ├── cfg.py                         # Scene configuration
│   ├── legbot_section001_np.py        # Training entry
│   └── xmls/
│       ├── legbot.xml                 # Legbot MJCF definition
│       ├── scene_stairs.xml           # Stairs scene
│       └── scene_world.xml            # Full track scene
├── terrain_tool/                      # Terrain generation tool
└── unitree_sdk2/                      # Unitree DDS communication library
```

---

## Robot Specifications

The Legbot URDF model definition is located at [legbot_rl_lab/unitree_ros/robots/legbot_description/urdf/legbot_description.urdf](legbot_rl_lab/unitree_ros/robots/legbot_description/urdf/legbot_description.urdf).

### Mass Distribution

| Component | Mass (kg) | Count | Total Mass (kg) |
|-----------|-----------|-------|-----------------|
| Base | 6.584 | 1 | 6.584 |
| Hip | 0.080 | 4 | 0.319 |
| Thigh | 1.550 | 4 | 6.202 |
| Calf | 0.184 | 4 | 0.736 |
| Foot | 0.040 | 4 | 0.160 |
| **Total** | | | **~14.0** |

### Robot Parameters

| Parameter | Value |
|-----------|-------|
| Robot Name | Legbot |
| Motor Model | RobStride RS02 (12 motors) |
| Degrees of Freedom | 12 (4 legs × hip/thigh/calf) |
| Standing Height (CoM) | 0.28 m (CoM height: 0.277 m) |
| Total Mass | ~14.0 kg |
| Base Mass | 6.584 kg |
| Thigh Length | 0.1985 m |
| Calf Length | 0.214 m |
| Foot Radius | 0.021 m |

### Motor Specifications

| Joint | Peak Torque | Peak Velocity | Gear Ratio | Note |
|-------|------------|---------------|------------|------|
| Hip | ±16 N·m | 30 rad/s | 1:1 | |
| Thigh | ±16 N·m | 30 rad/s | 1:1 | |
| Calf | ±32 N·m | 15.7 rad/s | 1:2 | Gear ratio 1:2. Model-space angle → motor-space angle: multiply by 2 |

### Joint Limits

| Joint | Lower (rad) | Upper (rad) |
|-------|-------------|-------------|
| FR/FL Hip | -0.733 | 0.733 |
| FR/FL Thigh | -1.559 | 3.130 |
| RR/RL Hip | -0.733 | 0.733 |
| RR/RL Thigh | -0.512 | 4.177 |
| All Calf | -2.639 | -0.785 |

### Hardware Configuration

| Parameter | Value |
|-----------|-------|
| Onboard Computer | Orange Pi 6 (aarch64) |
| Sensor | Custom serial IMU (accelerometer + gyroscope) |
| Communication Bus | USB-CAN × 2 + USB-Serial (IMU) |
| Control Frequency | 1 kHz (controller) / 500 Hz (gateway) |
| Middleware | CycloneDDS (localhost loopback) |

---

## Quick Start

### 1. Install Isaac Lab

Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) carefully.

### 2. Install This Project

```bash
# Clone the repository
git clone https://github.com/Robot-Nav/legbot_lab.git
cd legbot_lab

# Activate Isaac Lab conda environment
conda activate env_isaaclab

# Install in editable mode
pip install -e legbot_rl_lab/source/unitree_rl_lab
```

### 3. Start Training

#### 3.1 Main Environment Training (4096 Parallel Envs, PPO)

```bash
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --headless \
    --num_envs 4096 \
    --max_iterations 50000
```

**CLI Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Training task name | Unitree-Legbot-Velocity |
| `--headless` | Headless mode (faster training) | False |
| `--num_envs` | Number of parallel environments | 4096 |
| `--max_iterations` | Maximum training iterations | 50000 |
| `--seed` | Random seed | Random |
| `--resume` | Resume from checkpoint | False |
| `--load_run` | Run directory for resuming | - |
| `--checkpoint` | Model file for resuming | - |

#### 3.2 Resume Interrupted Training

```bash
# Auto-find latest checkpoint
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --resume \
    --headless

# Specify run directory and checkpoint manually
python legbot_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Legbot-Velocity \
    --resume \
    --load_run 2026-06-25_17-11-16 \
    --checkpoint model_108.pt \
    --headless
```

#### 3.3 NumPy Navigation Environment Training

```bash
python legbot/legbot_section001_np.py
```

### 4. Model Inference & Export

```bash
# Run inference and auto-export ONNX
python legbot_rl_lab/scripts/rsl_rl/play.py --task Unitree-Legbot-Velocity
```

Running inference automatically exports the following files to `logs/rsl_rl/unitree_legbot_velocity/<run_name>/exported/`:
- `policy.pt` — TorchScript model (Python inference)
- `policy.onnx` — ONNX model (C++ deployment)
- `deploy.yaml` — Deployment configuration (joint mapping, scale factors, etc.)

---

## Sim2Real Deployment

### System Architecture

```
┌────────────────────┐  DDS (rt/lowcmd, rt/lowstate)  ┌──────────────────────┐  Serial   ┌──────────────┐
│   legbot_ctrl       │◄──────────────────────────────►│  serial_dds_gateway  │◄─────────►│ 12 Motors +  │
│   (RL Controller)  │         CycloneDDS              │  (DDS↔Serial Bridge) │ type1-4   │ IMU          │
└────────────────────┘                                └──────────────────────┘           └──────────────┘
         │                                                       │
  Runs on Orange Pi                                        Runs on Orange Pi
  1kHz control loop                                       500Hz protocol translation
   └──────────── Shared lo interface ────────────┘
```

**Core Design Philosophy: The controller code is identical in simulation and on the real robot.** The DDS abstraction layer enables switching by simply changing the communication peer (MuJoCo simulator or serial_dds_gateway hardware bridge).

### Build & Run

#### 1. Build Gateway (Hardware Bridge)

```bash
cd serial_dds_gateway
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)
```

#### 2. Build Controller

```bash
cd legbot_rl_lab/deploy/robots/legbot
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)
```

#### 3. Run (Two-Terminal Setup)

**Terminal 1 — Start Gateway (must start first):**

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

**Terminal 2 — Start Controller (after gateway is ready):**

```bash
cd legbot_rl_lab/deploy/robots/legbot
./build/legbot_ctrl --network lo
```

#### 4. FSM State Transitions (Gamepad Controls)

```
Passive (damped)──LT+A──► FixStand (standing)──Start──► Velocity (RL active)
      ▲                        ▲                              │
      │                        │                              │
      └──────── LT+B ──────────┴────────── LT+B ─────────────┘
```

---

## MDP Definition

### Observation Space (Policy Network)

| Observation | Dims | Scale | Noise |
|-------------|------|-------|-------|
| Base angular velocity (IMU gyro) | 3 | 0.25 | U(-0.2, 0.2) |
| Projected gravity | 3 | 1.0 | U(-0.05, 0.05) |
| Velocity commands | 3 | 1.0 | - |
| Joint positions (relative to default) | 12 | 1.0 | U(-0.03, 0.03) |
| Joint velocities | 12 | 0.05 | U(-2.0, 2.0) |
| Last action | 12 | 1.0 | - |
| **Total** | **45** | | |

The policy network also receives **10 frames of observation history** (`history_length=10`) for temporal modeling.

### Action Space

| Parameter | Value |
|-----------|-------|
| Control Mode | Joint position control (PD) |
| Action Range | $[-1, 1]^{12}$ |
| Action Scale | ×0.25 rad |
| PD Stiffness $K_p$ | 60 N·m/rad |
| PD Damping $K_d$ | 4.0 N·m·s/rad |
| Default Hip Angle | 0.0 rad |
| Default Thigh Angle | 0.9 rad |
| Default Calf Angle | -1.8 rad |

### Critic Privileged Information

The Critic network has access to privileged information unavailable to the policy (Asymmetric Actor-Critic):

| Privileged Info | Dims | Description |
|-----------------|------|-------------|
| Base linear velocity | 3 | Ground-truth velocity |
| Joint accelerations | 12 | Acceleration values |
| Joint torques | 12 | Ground-truth torques |
| Foot contact forces | 4 | Normal forces for four feet |
| Height scan (wide) | 187 | 1.6×1.0m terrain height map |

### Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| Timeout | Episode reaches 25 seconds |
| Fallen | Base contact force > 1.0 N |

---

## Reward Function Design

The total reward is a weighted sum of the following terms. Positive rewards encourage tracking velocity commands; negative rewards penalize undesirable motion patterns.

### Positive Rewards (Tracking)

| Reward Term | Weight | Formula | Description |
|-------------|--------|---------|-------------|
| Linear velocity tracking | +1.0 | $\exp(-\|v_{xy} - v_{cmd}\|^2 / 0.5)$ | Exponential kernel, σ=0.5 |
| Angular velocity tracking | +0.5 | $\exp(-\|\omega_z - \omega_{cmd}\|^2 / 0.5)$ | Exponential kernel, σ=0.5 |

### Negative Rewards (Penalties)

| Penalty Term | Weight | Description |
|--------------|--------|-------------|
| Vertical linear velocity L2 | -2.0 | Suppress vertical oscillation |
| Horizontal angular velocity L2 | -0.05 | Suppress roll/pitch |
| Base height deviation L2 | -1.0 ~ -10.0 | Curriculum learning, maintain 0.28m target |
| Joint acceleration L2 | -1e-7 | Encourage smooth motion |
| Joint power | -2e-5 | Minimize energy consumption |
| Joint torque L2 | -1e-4 | Avoid excessive torque |
| Action rate L2 | -0.01 | Encourage smooth control |
| Action smoothness L2 | -0.01 | Third-derivative penalty |
| Undesired contacts | -1.0 | Thigh/calf should not touch ground (threshold 5N) |
| Joint position limits | -2.0 | Avoid exceeding joint range |
| Feet regulation | -0.05 | Foot height and spacing constraints |
| Hip position L1 | -0.05 | Keep hip near 0 at rest |
| Joint position L1 | -0.01 | Maintain default pose at rest |

### Curriculum Learning

| Curriculum Item | Start | End | Iteration Range |
|-----------------|-------|-----|-----------------|
| Vertical velocity penalty weight | -2.0 | 0.0 | 0 → 1,500 |
| Height penalty weight | -1.0 | -10.0 | 0 → 5,000 |
| Terrain difficulty | Level 0 | Level 5 | Progressive |

---

## Domain Randomization

To bridge the reality gap and enable Sim2Real transfer, extensive domain randomization is applied during training. Parameters are randomized at startup, on each episode reset, or at fixed intervals.

### Mass & Inertia Randomization

> Mode: `startup` (randomized once at training start, held constant throughout)

| Parameter | Distribution | Operation | Description |
|-----------|-------------|-----------|-------------|
| Base mass | U(-1.0, 1.0) kg | Additive | Simulate payload variation (±1kg) |
| Non-base link mass | U(0.9, 1.1) | Multiplicative | Link mass ±10% |
| Moment of inertia | U(0.9, 1.1) | Multiplicative | All bodies ±10% |
| Center of mass | U(-0.05, 0.05) m | Offset | Base CoM ±5cm (x/y/z) |

### Friction & Contact Randomization

> Mode: `startup`

| Parameter | Range | Description |
|-----------|-------|-------------|
| Static friction coefficient | U(0.0, 2.0) | 64 discrete friction pairs |
| Dynamic friction coefficient | U(0.0, 2.0) | Tied to static friction |
| Restitution coefficient | U(0.0, 0.5) | Collision elasticity |

### Actuator Randomization

> Mode: `reset` (randomized at each episode reset)

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| PD stiffness Kp | U(0.9, 1.1) | Multiplicative, simulates motor variance |
| PD damping Kd | U(0.9, 1.1) | Multiplicative |
| Joint zero offset | U(-0.035, 0.035) rad | ±35 mrad ≈ ±2°, simulates encoder errors |

### Initial State Randomization

> Mode: `reset`

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| Initial joint positions | U(0.5, 1.5) × default | 50%~150% of default pose |
| Base position | U(-0.5, 0.5) m (x/y) | Random horizontal offset |
| Base height | U(0.0, 0.2) m offset | Random initial height |
| Base yaw | U(-π, π) | Omnidirectional random orientation |
| Base linear velocity | U(-0.5, 0.5) m/s | Random initial velocity |
| Base angular velocity | U(-0.5, 0.5) rad/s | Random initial angular velocity |

### External Perturbations

> Mode: `interval`, triggered every 4 seconds

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| Linear velocity push (x/y) | U(-0.4, 0.4) m/s | Simulate external force push |
| Angular velocity push (roll/pitch/yaw) | U(-0.6, 0.6) rad/s | Simulate rotational disturbance |

### Observation Noise

> Mode: applied every step (policy observation only)

| Observation | Noise Distribution | Scale |
|-------------|-------------------|-------|
| Base angular velocity | U(-0.2, 0.2) | 0.25 |
| Projected gravity | U(-0.05, 0.05) | 1.0 |
| Joint positions | U(-0.03, 0.03) | 1.0 |
| Joint velocities | U(-2.0, 2.0) | 0.05 |

> Note: Critic observations are noise-free (`enable_corruption=False`) to ensure accurate value estimates.

### Terrain Randomization

Terrains are procedurally generated to create rough terrain with curriculum learning support. As training progresses (`train_env_steps`), terrain difficulty increases from Level 0 (flat) to Level 5 (complex rugged).

---

## Training Parameters

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Learning rate | $1.0 \times 10^{-3}$ (adaptive) |
| Clipping parameter $\epsilon$ | 0.2 |
| Discount factor $\gamma$ | 0.99 |
| GAE parameter $\lambda$ | 0.95 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 1.0 |
| Max gradient norm | 1.0 |
| Steps per environment | 24 |
| Learning epochs | 5 |
| Mini-batches | 4 |
| Target KL divergence | 0.01 |
| Max iterations | 50,000 |

### Training Scale

| Parameter | Value |
|-----------|-------|
| Parallel environments | 4,096 |
| Physics timestep | 0.005 s (200 Hz) |
| Decimation | 4 (control = 50 Hz) |
| Episode length | 25 s (1,250 control steps) |
| Samples per iteration | 4,096 × 24 = 98,304 |
| Total training steps | ~4.9 × 10⁹ max |

### Network Architecture

| Parameter | Value |
|-----------|-------|
| Hidden layer dimensions | [512, 256, 128] |
| Activation function | ELU |
| Policy observation dim | 45 (with 10-frame history) |
| Policy output dim | 12 (joint position offset) |
| Critic observation dim | ~263 (including 187-dim height scan) |

---

## Safety Mechanisms

### Command-Side Clipping (Non-intrusive)

| Protection | Limit | Description |
|------------|-------|-------------|
| Action Clip | ±100 | Raw policy output clipping |
| Joint angle limits | Configurable | Hard position limits |
| Torque clipping | ±40 Nm | Torque saturation |
| Angular change clipping | 0.05 rad/tick | Prevent neural network spikes |
| Velocity change clipping | 1.0 rad/s/tick | Prevent velocity surges |

### Feedback-Side Overlimit Protection (Transition to Passive)

| Monitor | Threshold | Action |
|---------|-----------|--------|
| Communication timeout | - | → Passive |
| Joint velocity | 30 rad/s | → Passive |
| Feedback torque | 45 Nm | → Passive |
| Motor temperature | 80°C | → Passive |
| IMU Roll | ±0.5 rad (~28°) | → Passive |
| IMU Pitch | ±0.5 rad (~28°) | → Passive |
| Emergency stop flag | - | → Passive |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Simulation Platform | NVIDIA Isaac Sim 5.0.0 / Isaac Lab 2.2.0 |
| Physics Engine | PhysX GPU (4,096 parallel) |
| RL Algorithm | RSL-RL 2.3.1 (PPO) |
| Deep Learning | PyTorch (CUDA accelerated) |
| Model Export | ONNX Runtime 1.22.0 |
| Deployment Language | C++17 |
| Middleware | CycloneDDS |
| Robot Model | URDF / MJCF |
| Configuration | Hydra / YAML |
| Simulation Verification | MuJoCo + DDS bridge |
| Onboard Computer | Orange Pi 6 (aarch64) |
| Motors | RobStride RS02 |

---

## Team

This project is developed and maintained by the **Robot-Nav** team.

GitHub: [https://github.com/Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab)

---

## License

MIT License

---

## Citation

```
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
