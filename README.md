# <div align="center">LegBot RL RobotLab</div>

<div align="center">

**MoE-CTS: Mixture of Experts Concurrent Teacher-Student Reinforcement Learning for Quadruped Locomotion**

[中文文档 (Chinese README)](README_cn.md) | [技术文档 (Technical Doc)](TECHNICAL_DOC_zh.md)

</div>

---

## Overview

**LegBot RL RobotLab** is an NVIDIA IsaacLab-based reinforcement learning training and deployment framework for the LegBot quadruped robot. It implements the **MoE-CTS (Mixture of Experts – Concurrent Teacher-Student)** algorithm — the IsaacLab implementation of the [CTS paper](https://arxiv.org/pdf/2405.10830) and a reimplementation of [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym).

LegBot is a custom quadruped robot sharing the same 12-joint kinematic structure as Unitree Go2 (4 legs × 3 joints: hip, thigh, calf), enabling direct reuse of MDP modules.

**Core pipeline:**

<p align="center">
  <b>IsaacLab Training → MuJoCo Sim2Sim Validation → Real LegBot Deployment</b>
</p>

| 名称 | 视频演示 |
| :--- | :---: |
| **legbot爬楼梯** | <video src="https://github.com/user-attachments/assets/99bb00bd-92cd-465d-ac19-94da37ae8810" width="100%" controls muted autoplay loop></video> |
| **legbot碎石堆** | <video src="https://github.com/user-attachments/assets/35efd286-cab9-4b0f-a7f3-bb45e754a128" width="100%" controls muted autoplay loop></video> |
| **实物sim2real** | <video src="https://github.com/user-attachments/assets/eef8a52f-630c-465d-b445-44af1b34fdec" width="100%" controls muted autoplay loop></video><br><video src="https://github.com/user-attachments/assets/ac39a0d0-0bbb-45e5-ba66-a616a95ac10d" width="100%" controls muted autoplay loop></video><br><video src="" width="100%" controls muted autoplay loop></video> |



### Problem & Contribution

Legged locomotion faces three key challenges:

1. **Incomplete perception**: Real robots lack full state information (exact linear velocity, terrain height maps), yet training without privileged information hurts performance.
2. **Sim2Real gap**: Dynamics mismatch between simulation and reality (motor characteristics, friction, mass distribution).
3. **Terrain generalization**: Robust walking across diverse terrains (stairs, slopes, rough ground).

This project contributes:

- **Concurrent Teacher-Student (CTS)**: Teacher (privileged) and Student (deployable) networks train concurrently; the student distills latent representations from the teacher. At deployment, only the student is used.
- **Mixture of Experts (MoE)**: The student encoder uses 8 expert networks with gating for dynamic composition, improving modeling of complex terrain patterns.
- **Realistic motor model**: Uses the official Unitree torque-speed-curve motor model (Go2 HV) instead of simple PD controllers, reducing the Sim2Real gap.

---

## Algorithm: MoE-CTS

### Core Idea

MoE-CTS combines **PPO** with **Concurrent Teacher-Student distillation** and introduces a **Mixture of Experts (MoE)** mechanism in the student encoder.

During training, environments are split into two groups:
- **Teacher environments (75%)**: Use the teacher encoder with privileged observations (critic obs: linear velocity, height scans, joint torques, etc.)
- **Student environments (25%)**: Use the student MoE encoder with deployable observations only (actor obs: angular velocity, gravity direction, joint states, commands)

Both groups share the same Actor and Critic networks; only the encoders differ. The student learns to mimic the teacher's latent representations via distillation loss. At deployment time, only the student branch is used — no privileged information required.

### Algorithm Flow

```
For each training iteration:

1. Rollout (num_steps_per_env=24 steps):
   - Teacher envs: obs → TeacherEncoder → latent → Actor → action
   - Student envs: obs → StudentMoEEncoder → latent → Actor → action
   Environment executes actions, stores transitions

2. Compute returns (GAE):
   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
   A_t = δ_t + γ·λ·A_{t+1}
   R_t = A_t + V(s_t)

3. PPO Update (5 epochs × 4 mini-batches):
   - Policy Loss:   L_policy = -E[min(r·A, clip(r)·A)]
   - Value Loss:    L_value  = E[(V - R)²]
   - Entropy Bonus: L_entropy = -β·H[π]

4. Student Encoder Distillation:
   - Latent Loss:      L_latent  = ||StudentEncoder(o_a) - TeacherEncoder(o_c).detach()||²
   - Load Balance Loss: L_balance = ||mean(gates) - 1/N||²
   - Student Total:    L_student = L_latent + α·L_balance, α=0.01
```

### Network Architecture

#### Teacher Encoder

$$z_t = \text{L2Norm}(\text{MLP}_{\text{teacher}}(o_c))$$

- Input: critic observations (privileged)
- Structure: MLP[512, 256] → L2Norm
- Output: `latent_dim = 32`

#### Student MoE Encoder

$$g = \text{Softmax}(\text{MLP}_{\text{gate}}(o_a))$$
$$e_i = \text{Expert}_i(\text{Backbone}(o_a)), \quad i=1,\dots,N$$
$$z_s = \text{L2Norm}\left(\sum_{i=1}^{N} g_i \cdot e_i\right)$$

- Number of experts: N = 8
- Shared backbone: MLP[512, 256, 256]
- Each expert: Conv1d(groups=8)
- Gating network: MLP[512, 256, 256] → Softmax
- Output: `latent_dim = 32`

#### Actor (Policy Network)

$$a \sim \mathcal{N}(\mu, \sigma^2), \quad \mu = \text{MLP}_{\text{actor}}([z, o_{\text{single}}])$$

- Input: [latent(32), single_obs(45)] = 77 dims
- Structure: MLP[512, 256, 128] → 12 (action dim)
- Standard deviation: learnable scalar parameter

#### Critic (Value Network)

$$V(s) = \text{MLP}_{\text{critic}}([z.\text{detach}(), o_c])$$

- Input: [latent(32), critic_obs]
- Structure: MLP[512, 256, 128] → 1

### PPO Objective

$$L_{\text{PPO}} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right] + c_v \cdot L_{\text{value}} - c_e \cdot H[\pi]$$

where:

$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, \quad \epsilon = 0.2$$

### L2Norm & SimNorm

Latent vector normalization for stable distillation:

$$\text{L2Norm}(x) = \frac{x}{\|x\|_2}$$

$$\text{SimNorm}(x) = \text{Softmax}(x_{\text{reshape}[-1, 8]}) \quad \text{(Simplicial Normalization)}$$

### CatELU Activation (optional)

Inspired by [Concat ReLU](https://arxiv.org/pdf/2303.07507), CatELU doubles feature dimension:

$$\text{CatELU}(x) = [\text{ELU}(x), \text{ELU}(-x)]$$

---

## Project Structure

```
legbot_lab/
├── scripts/rsl_rl/              # Training & evaluation entry points
│   ├── train.py                 # Training entry
│   ├── play.py                  # Evaluation + policy export
│   ├── cli_args.py              # CLI arguments
│   └── rsl_rl_utils.py          # Logging & export utilities
├── source/
│   ├── robot_lab/               # Environment & task definitions
│   │   └── robot_lab/
│   │       ├── assets/          # Robot assets, actuator configs
│   │       │   ├── legbot.py    # LegBot robot configuration
│   │       │   └── unitree_actuator.py  # Unitree motor model
│   │       └── tasks/
│   │           ├── go2/         # Go2 MDP modules (reused by LegBot)
│   │           │   ├── mdp/     # Rewards, observations, commands, events, curriculums
│   │           │   └── manager/ # Action manager
│   │           └── legbot/      # LegBot-specific configs
│   │               ├── env_cfg.py      # Environment configuration
│   │               ├── rsl_rl_cfg.py   # RL algorithm configuration
│   │               └── env/legbot_env.py  # Environment class
│   └── rsl_rl/                  # Custom RSL-RL library
│       └── rsl_rl/
│           ├── algorithms/moe_cts.py        # MoE-CTS algorithm core
│           ├── modules/actor_critic_moe_cts.py  # Teacher-student network
│           ├── networks/moe.py             # MoE network implementation
│           ├── runners/on_policy_runner_cts.py  # CTS training loop
│           ├── storage/rollout_storage_cts.py   # CTS rollout storage
│           └── utils/exporter_cts.py       # Policy exporter
├── deploy/deploy_mujoco/        # MuJoCo Sim2Sim deployment
│   ├── deploy_legbot.py         # Deployment script
│   ├── utils.py                 # Deployment utilities
│   └── configs/legbot.yaml      # Deployment configuration
├── resources/legbot/            # MuJoCo scenes & URDF
│   ├── urdf/legbot.urdf         # LegBot URDF model
│   ├── meshes/                  # STL mesh files
│   ├── flat.xml / stairs.xml / boxes.xml  # Terrain scenes
│   └── legbot.xml               # LegBot MuJoCo model
├── src/                         # C++ simulation & bridge
│   ├── legbot_bridge.h          # DDS bridge for MuJoCo
│   ├── param.h                  # Simulation config
│   └── main.cc                  # MuJoCo simulation main
├── TECHNICAL_DOC_zh.md          # Detailed technical documentation (Chinese)
└── README.md                    # This file
```

---

## MDP Definition

### Observation Space

Defined in [env_cfg.py](source/robot_lab/robot_lab/tasks/legbot/env_cfg.py):

| Group | Purpose | History | Noisy | Components |
|-------|---------|---------|-------|------------|
| **policy** (actor obs) | Student encoder input | 10 | Yes | base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action |
| **critic** (critic obs) | Teacher/Critic input | 1 | No | All above + base_lin_vel, joint_acc, joint_torque, contact_force, height_scan |
| **single_obs** | Actor concatenation | 1 | Yes | Same as policy, current frame only |

**Joint order:**

```python
["FL_hip", "FL_thigh", "FL_calf",
 "FR_hip", "FR_thigh", "FR_calf",
 "RL_hip", "RL_thigh", "RL_calf",
 "RR_hip", "RR_thigh", "RR_calf"]
```

### Action Space

- Type: Joint position control (`JointPositionActionCfg`)
- Dimensions: 12
- Scale: 0.25 (action × 0.25 + default joint angle → target joint angle)
- Clip range: [-100, 100]

### Command Space

- Dimensions: 3 — `[lin_vel_x, lin_vel_y, ang_vel_yaw]`
- Resample interval: 5s
- Dynamic resampling: adjusts lower velocity bound based on remaining distance and episode time
- Terrain-dependent ranges: different speed limits for different terrain types
- Command curriculum: expands velocity ranges at 20k and 50k iterations
- Zero-command curriculum: gradually increases zero-command probability from 0 to 0.1

### Termination

- Timeout: 25s per episode
- Illegal contact: base link contact force > 1.0N

---

## Reward Design

Defined in [rewards.py](source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py), weighted sum of multiple terms.

### Tracking Rewards (positive)

**Linear velocity tracking** (exponential kernel):

$$r_{\text{lin}} = \exp\left(-\frac{\|v_{\text{cmd}}^{xy} - v_{\text{base}}^{xy}\|^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=1.0$$

**Angular velocity tracking**:

$$r_{\text{ang}} = \exp\left(-\frac{(\omega_{\text{cmd}}^{z} - \omega_{\text{base}}^{z})^2}{\sigma^2}\right), \quad \sigma=0.5, \quad w=0.5$$

### Penalty Terms (negative)

| Term | Formula | Weight | Description |
|------|---------|--------|-------------|
| lin_vel_z_l2 | $\|v_z\|^2$ | -2.0 → 0 (curriculum) | Vertical velocity penalty |
| ang_vel_xy_l2 | $\|\omega_{xy}\|^2$ | -0.05 | Roll/pitch angular velocity |
| joint_acc_l2 | $\sum \ddot{q}_i^2$ | -1e-7 | Joint acceleration (physics-step-level, very small in Lab) |
| joint_power | $\sum \|\dot{q}_i \cdot \tau_i\|$ | -2e-5 | Joint power consumption |
| joint_torques_l2 | $\sum \tau_i^2$ | -1e-4 | Joint torque penalty |
| base_height_l2 | $(h - 0.28)^2$ | -1.0 → -10.0 (curriculum) | Base height (uses height scanner for ground estimation) |
| action_rate_l2 | $\|a_t - a_{t-1}\|^2$ | -0.01 | Action rate of change |
| action_smoothness_l2 | $\|a_t - 2a_{t-1} + a_{t-2}\|^2$ | -0.01 | Second-order action smoothness |
| undesired_contacts | $\sum \mathbb{1}(\|F\| > 5)$ | -1.0 | Thigh/calf contact |
| joint_pos_limits | Joint limit violation | -2.0 | Joint position limits |
| feet_regulation | $\sum v_{\text{foot}}^{xy\,2} \cdot e^{-h_{\text{foot}}/(0.025 \cdot h_{\text{target}})}$ | -0.05 | Near-ground foot sliding |
| hip_pos_penalty_l1 | $\sum \|q_{\text{hip}} - q_{\text{default}}\|_1$ | -0.05 | Hip deviation from default |
| joint_pos_penalty_l1 | $\sum \|q_{\text{thigh,calf}} - q_{\text{default}}\|_1$ | -0.01 | Thigh/calf deviation from default |

### Base Height Estimation

`base_height_l2` and `feet_regulation` use a height scanner to estimate ground height rather than world z-coordinates:

```python
base_height = base_z - mean(ray_hits_z)  # minus estimated ground height
```

---

## Domain Randomization & Motor Model

### Domain Randomization

| Parameter | Mode | Range | Description |
|-----------|------|-------|-------------|
| Base mass | startup | ±1 kg | Payload variation |
| Other body mass | startup | ×[0.9, 1.1] | Mass distribution |
| Center of mass | startup | ±0.03 m | COM offset |
| Joint reset position | reset | ×[0.5, 1.5] | Random initial pose |
| Actuator gains (kp/kd) | reset | ×[0.9, 1.1] | PD parameter perturbation |
| Motor zero offset | reset | ±0.035 rad | Encoder offset |
| Push perturbation | interval (4s) | ±0.4 m/s, ±0.6 rad/s | Random external force |
| Friction coefficient | startup | [0, 2.0] | Variable ground friction |
| Base initial state | reset | pos ±0.5m, yaw ±π | Random initial pose |

### Unitree Motor Model

The project uses the official Unitree motor torque-speed curve model (Go2 HV parameters):

| Parameter | Value | Description |
|-----------|-------|-------------|
| X1 | 13.5 rad/s | Max speed at full torque (T-N curve knee) |
| X2 | 30 rad/s | No-load speed |
| Y1 | 20.2 N·m | Peak torque (same direction) |
| Y2 | 23.4 N·m | Peak torque (opposite direction) |

**Friction model:**

$$\tau_{\text{applied}} = \tau_{\text{PD}} - F_s \cdot \tanh\left(\frac{\dot{q}}{V_a}\right) - F_d \cdot \dot{q}$$

**Torque clipping:**
- $|\dot{q}| < X1$: clamped to Y1 (same direction) or Y2 (opposite direction)
- $|\dot{q}| \geq X1$: linearly decays to 0 at X2

**Motor delay**: `min_delay=0, max_delay=4` steps (motor-level, not action-level)

---

## Curriculum Learning

### Terrain Curriculum (terrain_levels_vel_gym)

Dynamically adjusts terrain difficulty based on robot traversal distance:
- `move_up`: max distance > terrain length/2 → increase difficulty
- `move_down`: max distance < target distance × 0.5 → decrease difficulty

### Reward Weight Curriculum (gradual_reward_weight_modification)

Linear interpolation of reward weights:
- `lin_vel_z_l2`: -2.0 → 0.0 (0→1500 iterations)
- `base_height_l2`: -1.0 → -10.0 (0→5000 iterations)

### Command Range Curriculum (command_range_curriculum)

Expands velocity command ranges at specified iterations:
```python
# 20000 iterations: lin_vel_x [-1,1], lin_vel_y [-1,1], ang_vel [-1.5,1.5]
# 50000 iterations: lin_vel_x [-2,2], lin_vel_y [-1,1], ang_vel [-2,2]
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_envs | 4096 | Parallel environments |
| num_steps_per_env | 24 | Steps per iteration |
| max_iterations | 300000 | Maximum training iterations |
| save_interval | 500 | Checkpoint save interval |
| num_learning_epochs | 5 | PPO epochs |
| num_mini_batches | 4 | Mini-batches per epoch |
| learning_rate | 1e-3 | Adaptive learning rate |
| student_encoder_lr | 1e-3 | Student encoder LR |
| gamma | 0.99 | Discount factor |
| lam | 0.95 | GAE lambda |
| clip_param | 0.2 | PPO clip range |
| entropy_coef | 0.01 | Entropy coefficient |
| value_loss_coef | 1.0 | Value loss coefficient |
| load_balance_coef | 0.01 | MoE load balance coefficient |
| teacher_env_ratio | 0.75 | Teacher environment ratio |
| desired_kl | 0.01 | Target KL divergence |
| max_grad_norm | 1.0 | Gradient clipping |
| expert_num | 8 | Number of MoE experts |
| latent_dim | 32 | Latent vector dimension |
| history_length | 10 | Observation history length |
| sim_dt | 0.005 s | Physics timestep |
| control_decimation | 4 | 50 Hz control frequency |
| episode_length_s | 25.0 | Max episode duration |

---

## Installation

### 1. Install IsaacLab

Follow the [official guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/isaaclab_pip_installation.html):

```bash
conda create -n legbot_lab python=3.11
conda activate legbot_lab
pip install --upgrade pip
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install Custom RSL-RL and RobotLab

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

### 3. Install MuJoCo (optional, for Sim2Sim)

```bash
pip install mujoco pygame
```

---

## Training & Evaluation

```bash
# Train
python scripts/rsl_rl/train.py --task=RobotLab-Legbot-v0 --headless

# Evaluate
python scripts/rsl_rl/play.py --task=RobotLab-Legbot-v0
```

### CLI Flags

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

### Training Loop (OnPolicyRunnerCTS)

```python
for it in range(max_iterations):
    # 1. Rollout: collect num_steps_per_env=24 steps
    for _ in range(num_steps_per_env):
        actions = alg.act(obs)              # Teacher/student inference
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)

    # 2. Compute GAE returns
    alg.compute_returns(obs)

    # 3. PPO + distillation update
    loss_dict = alg.update()

    # 4. Save checkpoint (every save_interval=500 steps)
    if it % save_interval == 0:
        runner.save(f"model_{it}.pt")
```

---

## MuJoCo Sim2Sim Deployment

### Export Policy

Running `play.py` automatically exports policies in both TorchScript (`.pt`) and ONNX (`.onnx`) formats. The export wraps the student branch (StudentMoEEncoder + Actor) with normalizers into a single-input model:

```bash
python scripts/rsl_rl/play.py \
    --task=RobotLab-Legbot-v0 \
    --checkpoint logs/rsl_rl/legbot_moe_cts/<run_name>/model_<iter>.pt
```

**Key feature**: the exported policy internally maintains observation history, so deployment only requires current-frame input.

### Deploy in MuJoCo

Configure `deploy/deploy_mujoco/configs/legbot.yaml`:

```yaml
policy_path: "{ROOT_DIR}/logs/rsl_rl/legbot_moe_cts/<timestamp>/exported/policy.pt"
```

Run:

```bash
python deploy/deploy_mujoco/deploy_legbot.py
```

**Deployment loop:**

```python
while viewer.is_running():
    # 1. PD control → compute torques
    data.ctrl[:] = pd_control(target_pos, qpos, kps, target_vel, qvel, kds)
    # 2. MuJoCo physics step
    mujoco.mj_step(model, data)
    # 3. Query policy every decimation steps
    if counter % decimation == 0:
        features = build_features(data, action, cmd, cfg)
        single_obs = build_single_obs(features, layout)
        action = policy(single_obs)
        target_pos = default_angles + action * 0.25
```

### Switch Scenes

Modify `xml_path` in `legbot.yaml`:

```yaml
# Flat terrain
xml_path: "{ROOT_DIR}/resources/legbot/flat.xml"
# Stairs
xml_path: "{ROOT_DIR}/resources/legbot/stairs.xml"
# Box obstacles
xml_path: "{ROOT_DIR}/resources/legbot/boxes.xml"
# Stairs and slope
xml_path: "{ROOT_DIR}/resources/legbot/stairs_and_slope.xml"
```

### Controller Mapping

| Input | Function |
|-------|----------|
| LX / LY | Forward / Lateral velocity |
| RX | Angular velocity (yaw) |

- Auto-detects joystick connection; falls back to config `cmd_init: [1.0, 0.0, 0.0]` when no joystick is connected.

---

## Acknowledgments

This project is based on the following open-source works:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — Unified robot learning framework on NVIDIA Isaac Sim
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — Reinforcement learning algorithm library
- [robot_lab](https://github.com/fan-ziqi/robot_lab) — IsaacLab-based robot RL extension
- [MuJoCo](https://github.com/google-deepmind/mujoco) — High-performance physics simulator
- [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) — IsaacGym-based Go2 RL training (original implementation),本项目基于https://robogauge.github.io/complete项目进行适配与研究，需要详细的讲解可以参照上述链接。【点赞】

Relevant paper:

- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)

---

## License

This project is released under the Apache 2.0 License. See individual source files for specific licensing details.
