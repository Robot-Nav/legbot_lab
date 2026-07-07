# 第一阶段：香橙派6 单机 DDS 实机（Sim to Real）

> **全部进程**在香橙派上运行，DDS 使用本机回环 `lo`，不经过 PC。

```
┌──────────────────────────────────────────────────────┐
│                    香橙派 6                            │
│  终端 1: dds_to_serial_gateway ─→ myttyCAN0/1 + IMU │
│  终端 2: fatu_ctrl               ─→ rt/lowcmd/lowstate│
└──────────────────────────────────────────────────────┘
        ↕ rt/lowcmd / rt/lowstate (DDS)
┌──────────────────────────────────────────────────────┐
│              dds_to_serial_gateway                    │
│         串口 A/B (电机) ─ 串口 IMU                    │
├──────────────────────────────────────────────────────┤
│                灵足电机 ×12 + IMU                    │
└──────────────────────────────────────────────────────┘
```

| 组件 | 职责 |
|------|------|
| `fatu_ctrl` | FSM、策略推理、生成 `lowcmd` |
| `dds_to_serial_gateway` | 协议转换、串口收发、IMU/关节预处理、发布 `lowstate` |

> sim2sim（MuJoCo）可不启 gateway；香橙派实机**必须**启 gateway 才能驱动电机和读 IMU。

---

# 目录

- [1. 架构说明](#1-架构说明)
- [2. 编译](#2-编译)
- [3. 运行方式](#3-运行方式)
- [4. 启动标定说明](#4-启动标定说明)
- [5. 操作顺序（键盘）](#5-操作顺序键盘)
- [6. 推荐实机流程](#6-推荐实机流程)
- [7. One-Euro Filter 调参指南](#7-one-euro-filter-调参指南)
- [8. 2→3 无指令自旋排查](#8-23-无指令自旋排查)
- [9. 上线前检查清单](#9-上线前检查清单)
- [10. Gateway 重要说明](#10-gateway-重要说明)
- [11. 硬件拓扑](#11-硬件拓扑)
- [12. Gateway 主循环（500 Hz）](#12-gateway-主循环500-hz)
- [13. 协议层与工具](#13-协议层与工具)
- [14. Gateway 参数速查](#14-gateway-参数速查)

---

## 1. 架构说明

### 1.1 平台兼容性

| 环境 | CPU | 能否直接跑对方二进制 |
|------|-----|----------------------|
| 开发 PC | x86_64 | 否 |
| 香橙派 6 | aarch64 (arm64) | 否 |

> **不能在 PC 上编好的 `fatu_ctrl` / `dds_to_serial_gateway` 拷到香橙派运行**，必须在对应机器上**本地编译**（或做交叉编译）。

### 1.2 CMake 自动识别

| 组件 | 自动识别 |
|------|----------|
| `unitree_sdk2` | 已有 `lib/x86_64` 与 `lib/aarch64` |
| `fatu_ctrl` ONNX Runtime | 按 CPU 选 `onnxruntime-linux-x64` 或 `onnxruntime-linux-aarch64` |
| `serial_dds_gateway` | 源码通用，链接本机 DDS / unitree_sdk2 |

#### 下载 ONNX Runtime

香橙派首次编译前，在仓库根目录下载 **aarch64** 版：

```bash
cd /path/to/fatuDog
./scripts/fetch_onnxruntime.sh
```

PC 上若缺少 x64 包也可运行同一脚本。`cmake` 配置时会打印：

```text
Fatu deploy: CPU=aarch64, ONNX Runtime=onnxruntime-linux-aarch64-1.22.0
```

---

## 2. 编译

### 2.1 香橙派（aarch64，实机推荐）

```bash
cd ~/workspace/fatuDog   # 或你的克隆路径

./scripts/fetch_onnxruntime.sh

# 安装 unitree_sdk2 + CycloneDDS（若尚未安装）
cd unitree_sdk2 && cmake -S . -B build && cmake --build build -j && sudo cmake --install build

cd ../serial_dds_gateway
cmake -S . -B build && cmake --build build -j --target dds_to_serial_gateway

cd ../unitree_rl_lab/deploy/robots/fatu
cmake -S . -B build && cmake --build build -j
```

### 2.2 开发 PC（x86_64，sim2sim）

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway
cmake -S . -B build
cmake --build build -j --target dds_to_serial_gateway

cd /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/robots/fatu
cmake -S . -B build
cmake --build build -j
```

可选：复制控制器到统一输出目录：

```bash
mkdir -p /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/build
cp /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/robots/fatu/build/fatu_ctrl \
   /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/build/
```

---

## 3. 运行方式

### 3.1 终端 1：Gateway

> **上电后保持机器人趴下静止**（IMU 陀螺偏置标定 + 关节偏置标定期间不要碰）。

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway

./build/dds_to_serial_gateway \
  --serial-port-a /dev/myttyCAN0 \
  --serial-port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --imu-port /dev/myttyIMU \
  --imu-baudrate 921600 \
  --network lo \
  --channel 0x00 \
  --master-id 0x00FD \
  --tick-hz 500 \
  --send-disable-on-exit \
  --imu-gyro-calib-seconds 4 \
  --imu-gyro-deadzone 0.03 \
  --joint-bias-calib-seconds 2
```

首次标定用上面命令；已有 `config/joint_prone_bias.fatu.txt` 时见下文 [方式 B](#422-方式-b固定-bias-配置文件推荐重复开机)（`--joint-bias-load-file` + `--no-joint-bias-calib`）。

#### 正常启动日志

```text
========== Fatu Phase-1: DDS Serial Gateway ==========
[PHASE1] joint map: sign only (FL/RL thigh+calf sign=-1), no gear ratio
[PHASE1] joint bias: power-on prone calib 2s (motor q -> model q)
[INFO] IMU gyro bias calibrated over 4s ...
[INFO] joint motor bias calibration over 2s ...
[INFO] motor raw q: [...]              # 电机编码器原始角
[INFO] joint q after sign: [...]       # sign 换算后（无减速比）
[INFO] joint bias (joint-model space): [...]
[PHASE1] publishing rt/lowstate now (not waiting for rt/lowcmd)
[PHASE1] main loop running at 500 Hz
[STAT] type2_a=... type2_b=...
```

加载固定 bias 时：

```text
[PHASE1] joint bias: load from config/joint_prone_bias.fatu.txt
[INFO] joint motor bias loaded from file
```

### 3.2 终端 2：Controller

> **先启动终端 1，再启动终端 2。** SSH 远程请加 `-t` 以读取键盘。

```bash
cd /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/robots/fatu/build
./fatu_ctrl --network lo --csv-log
```

或使用统一目录：

```bash
cd /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/build
./fatu_ctrl --network lo --csv-log
```

#### 正常日志

```text
========== Fatu Phase-1: Controller ==========
[PHASE1] rt/lowstate connected — gateway is publishing motor/IMU feedback
[KEYBOARD] [1] Passive  [2] FixStand  [3] Velocity  [4] LieDown
[PHASE1] FSM started — current state: Passive
```

#### `config/config.yaml` 要点

| 状态 | 实现 | 说明 |
|------|------|------|
| FixStand | `type: FatuFixStand` | 专用站立状态；`kp/kd` 见 `FixStand` 段 |
| Velocity | `type: RLBase` | ONNX 策略；刚度见 `deploy.yaml` 的 `stiffness/damping` |
| 前馈力矩 | — | FixStand / Velocity 均为 **`tau = 0`** |

---

## 4. 启动标定说明

### 4.1 IMU 陀螺偏置

- Gateway 启动后静止 **4 s** 标定陀螺偏置，并施加死区（默认 `0.03 rad/s`）
- `fatu_ctrl` 进 Velocity **[3] 不再减 bias**，直接使用 gateway 处理后的角速度

### 4.2 关节映射（Gateway 内完成）

电机顺序：**FR, FL, RR, RL**（各 hip / thigh / calf），与 `config.yaml` 中 `qs` 一致。

> **当前实机策略：软件不做减速比换算**（编码器读数按 1:1 当作关节角），仅做符号翻转与偏置。

| 关节 | sign | 说明 |
|------|------|------|
| FR / RR：hip、thigh、calf | +1 | 与 URDF 同向 |
| FL / RL：hip | +1 | |
| FL / RL：**thigh、calf** | **−1** | 左腿编码器与模型相反 |

**完整映射公式：**

```text
q_model = sign × q_motor - bias
q_motor_cmd = (q_model_cmd + bias) / sign
```

kp / kd / tau：gateway 与模型空间 **1:1**（不对 calf 额外缩放）。

**默认参考趴姿**（与 LieDown / `config.yaml` 一致）：

```text
[-0.02, 1.08, -2.64, 0.03, 1.08, -2.64, -0.05, 1.08, -2.64, 0.06, 1.08, -2.64]
```

> **重要：** 更换 `motor_sign`、参考趴姿或机械结构后须 **重新标定 bias**（不要用旧 bias 文件）。

- `lowstate.q`：映射后的模型角（供 `fatu_ctrl` / 策略使用）
- `lowstate.q_raw`：电机侧原始角（调试用）

**配置文件目录**：`serial_dds_gateway/config/`（须在 `serial_dds_gateway/` 目录下启动 gateway，路径相对该目录）

| 文件 | 用途 |
|------|------|
| `joint_prone_reference.example.txt` | 标定用参考趴姿（模型空间，一行 12 个数） |
| `joint_prone_bias.fatu.txt` | **实机 bias 文件**（含标定会话注释 + 一行 12 个浮点数） |
| `joint_prone_bias.example.txt` | 占位模板（全 0） |
| `joint_prone_bias.calibrated.example.txt` | 数值演算示例（**勿直接当真机用**） |

### 4.3 关节偏置标定流程

两种方式二选一：**每次上电在线标定**（默认），或 **固定 bias 文件**（量产 / 重复开机更快）。

#### 4.3.1 方式 A：在线标定（默认）

**前提：** 上电后机器人已是自然趴姿（与 LieDown 终点接近），全程静止。

关节标定在 **主循环内后台进行**：gateway 启动后**立即发布** `rt/lowstate`，可先启 `fatu_ctrl`；电机需 Passive 使能后才有 12 路反馈，标定才会在约 2 s 内完成。

**操作步骤：**

1. 扶稳机器人，启动 gateway（保留 `--joint-bias-calib-seconds 2`）
2. 启动 `fatu_ctrl` 并进入 Passive（`1`），等待电机反馈
3. 约 4 s（IMU）+ 12 路齐后 2 s（关节）标定完成；若 60 s 仍无 12 路反馈会 **超时用 zero bias** 并打 `[WARN]`
4. 在终端 1 日志中找到关键输出：

```text
[INFO] motor raw q: [ ... ]              # 电机编码器原始角（上电趴姿平均）
[INFO] joint q after sign: [ ... ]      # sign 换算后；应与参考趴姿接近
[INFO] joint bias (joint-model space): [ ... ] # 本次 bias，可保存到 joint_prone_bias.fatu.txt
```

5. 启动 `fatu_ctrl`，按 `1` 进入 Passive，按 **`P`** 打印关节角；`qs` 应接近参考趴姿：

```text
[Passive JOINT] qs: [-0.02, 1.08, -2.64, 0.03, 1.08, -2.64, ...]
```

6. 若 `qs` 与参考差 > 0.05 rad：检查是否碰过机器人、FL/RL 符号、参考文件；标定样本应 **几百个以上**（若只有几十个说明 CAN 反馈不稳）

**bias 计算公式（与 gateway 一致）：**

```text
bias[i] = sign[i] × q_motor_avg[i] - q_reference[i]
```

**合格标定：** 各关节 `|bias[i]|` 一般 **< 0.35**；calf 若出现 **> 1.0** 多为样本太少或参考与真趴姿不符。

#### 4.3.2 方式 B：固定 bias 配置文件（推荐重复开机）

仓库已提供 `config/joint_prone_bias.fatu.txt`（Fatu 趴姿初值，无减速比映射下标定）。香橙派需同步该文件到 `serial_dds_gateway/config/`。

**操作步骤：**

1. 若需更新：按方式 A 完成**一次**标定，从日志复制 `joint bias` 那一行的 12 个数
2. 写入 `serial_dds_gateway/config/joint_prone_bias.fatu.txt`（一行 12 个浮点数，逗号分隔）：

```text
# 示例（Fatu 实测，sign only，无 gear）：
0.1724, 0.1589, 0.3106, -0.1813, 0.0875, 0.3140, -0.0737, 0.1137, 0.2193, 0.0736, 0.0676, 0.2774
```

3. 在 **`serial_dds_gateway/` 目录下** 启动 gateway，跳过在线关节标定：

```bash
./build/dds_to_serial_gateway \
  --serial-port-a /dev/myttyCAN0 \
  --serial-port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --imu-port /dev/myttyIMU \
  --imu-baudrate 921600 \
  --network lo \
  --channel 0x00 \
  --master-id 0x00FD \
  --tick-hz 500 \
  --send-disable-on-exit \
  --imu-gyro-calib-seconds 4 \
  --imu-gyro-deadzone 0.03 \
  --joint-bias-load-file config/joint_prone_bias.fatu.txt \
  --no-joint-bias-calib
```

**说明：**

- `--joint-bias-load-file` 与 `--no-joint-bias-calib` 成对使用；IMU 陀螺标定仍默认进行
- 路径相对 **当前工作目录**；请在 `serial_dds_gateway/` 下执行，或使用绝对路径
- 若报 `cannot open joint bias file`：检查是否在正确目录、文件是否已 scp 到香橙派
- 更换参考趴姿时须同步更新 `joint_prone_reference` 并**重新标定** bias
- 机械拆装、换电机、趴姿改变后应重新标定或更新 `joint_prone_bias.fatu.txt`

#### 4.3.3 方式 C：仅改参考角（bias 仍在线算）

若仿真 / `config.yaml` 更新了趴姿目标，可指定参考文件而不改代码默认值：

```bash
  --joint-bias-reference-file config/joint_prone_reference.example.txt \
  --joint-bias-calib-seconds 2
```

#### 4.3.4 偏置相关参数速查

| 参数 | 说明 |
|------|------|
| `--no-joint-bias-calib` | 关闭在线偏置标定（配合 `--joint-bias-load-file`） |
| `--joint-bias-calib-seconds 2` | 在线标定时长（12 路齐后计时） |
| `--joint-bias-calib-timeout 60` | 等不到 12 路反馈时的超时（秒） |
| `--joint-bias-reference-file PATH` | 自定义参考趴姿（一行 12 个数） |
| `--joint-bias-load-file PATH` | 加载已保存 bias |

---

## 5. 操作顺序（键盘）

| 按键 | 功能 |
|------|------|
| `1` | Passive（阻尼，跟随当前关节角） |
| `2` | FixStand（约 2 s 插值到站立 `[0, 0.9, -1.8] × 4`） |
| `3` | Velocity（RL 速度控制） |
| `4` | LieDown（约 2.5 s 插值到趴姿） |
| `P` | Passive 下打印当前 12 关节角（标定/核对用） |
| `W` / `S` | 前进 / 后退（锁存，训练范围内） |
| `A` / `D` | 左移 / 右移 |
| `Q` / `E` | 左转 / 右转 |
| `Space` | 速度清零 |

**Velocity 默认速度**（与训练范围对齐）：

| 按键 | 速度 | 单位 |
|------|------|------|
| `W` | +0.4 | m/s |
| `S` | -0.5 | m/s |
| `A` / `D` | ±0.4 | m/s |
| `Q` / `E` | ±0.5 | rad/s |

> **力矩前馈：** FixStand 与 Velocity 当前均为 **`motor_cmd.tau = 0`**（纯 PD，无重力前馈）。站立发软时可先加大 `config.yaml` 中 FixStand 的 `kp/kd`，或 Velocity 对应 `deploy.yaml` 的 `stiffness/damping`。

---

## 6. 推荐实机流程

```text
[准备] ──→ [上电] ──→ [终端1: Gateway] ──→ [终端2: fatu_ctrl]
                                                      │
                                                      ↓
                                            [2] FixStand ──→ [3] Velocity
                                                                  │
                                                          [WASDQE] 行走测试
                                                                  │
                                                     [Space] 清零 ──→ [1] Passive
                                                                          │
                                                                 [Ctrl+C] 关闭
```

**详细步骤：**

| 步骤 | 操作 | 注意 |
|------|------|------|
| 1 | 扶稳或架空，确认串口设备存在 | `ls -l /dev/mytty*` |
| 2 | **终端 1** 启动 gateway | 在线标定或加载 bias 文件，等 IMU 标定完成 |
| 3 | **终端 2** 启动 `fatu_ctrl` | `--csv-log` 记录诊断日志 |
| 4 | 按 `2` → FixStand | 等待站立稳定 |
| 5 | 按 `3` → Velocity | **站稳约 1 s 不按键**，观察 `grav≈0,0,-1` |
| 6 | `WASDQE` 行走测试；`Space` 清零 | 先轻按，确认响应再推进 |
| 7 | 急停：按 `1` → Passive | |
| 8 | **终端 1** `Ctrl+C` | 自动向 12 路发 type4 失能 |

---

## 7. One-Euro Filter 调参指南

Velocity 模式下采用 One-Euro 自适应低通滤波器对策略输出动作进行平滑，原理：

```
fc       = fc_min + beta × |speed|       # 自适应截止频率
alpha    = 1 / (1 + 1/(2π·fc·te))         # 平滑系数
filtered = alpha × raw + (1-alpha) × prev_filtered
```

### 7.1 参数含义

| 参数 | 当前值 | 含义 | 调参方向 |
|------|--------|------|---------|
| `kEuroTe` | **0.02 s** | 采样周期 = 1/策略频率(50Hz) | **固定值，勿改** |
| `kEuroFcMin` | **1.0 Hz** | 静止时最小截止频率，越低越平滑 | ↓ 更强抑抖 ↑ 起步更快 |
| `kEuroBeta` | **0.05** | 速度→截止频率增益，越高运动时越跟手 | ↓ 运动中也平滑 ↑ 运动滞后小 |
| `kEuroFcD` | **1.0 Hz** | 速度估计平滑截止频率，一般与 fc_min 一致 | ↓ 速度更稳 ↑ fc 响应更快 |

### 7.2 问题速查表

| 你遇到的问题 | 改什么 | 怎么改 |
|------------|-------|-------|
| 站立时 hip 还抖 | ↓ `kEuroFcMin` | 1.0 → **0.5** |
| 站立很稳但起步慢半拍 | ↑ `kEuroFcMin` | 1.0 → **1.5** |
| 前进/转向滞后 | ↑ `kEuroBeta` | 0.05 → **0.08~0.10** |
| 运动时也在晃 | ↓ `kEuroBeta` + ↓ `kEuroFcMin` | 0.05→0.03, 1.0→0.5 |
| 会抖但想保持起步响应 | ↑ `kEuroFcD` | 1.0 → **2.0** |

### 7.3 修改位置

`unitree_rl_lab (实物部署)/deploy/include/FSM/State_RLBase.h` 中 4 个 `static constexpr` 常量。

---

## 8. 2→3 无指令自旋排查

| 步骤 | 检查项 | 解决 |
|------|--------|------|
| 1 | Gateway 是否已打印 `IMU gyro bias calibrated` | 若未打印，等标定完成 |
| 2 | `[2]` 站稳 → `[3]`，观察 `ang_vel` z 是否偏大 | 加大 `--imu-gyro-deadzone` |
| 3 | `grav` 是否 `0,0,-1` | 若不是，查 IMU 轴序/欧拉角定义 |
| 4 | 站立后关节是否不对称 | 核对 gateway 关节映射与 `deploy.yaml` 中 `joint_ids_map` |

---

## 9. 上线前检查清单

| # | 检查项 | 验证方法 |
|---|--------|---------|
| 1 | 串口设备存在 | `ls -l /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU` |
| 2 | 12 电机串口测试通过 | `twelve_motor_serial`（见 `serial_dds_gateway/README.md`） |
| 3 | ONNX 策略已导出 | `unitree_rl_lab/logs/rsl_rl/unitree_fatu_velocity/.../exported/policy.onnx` |
| 4 | Gateway 通信正常 | `[STAT]` 中 `type2_a`、`type2_b` 持续增加 |
| 5 | 偏置标定正确 | 上电趴姿与参考角接近；Passive 按 `P` 打印的 `qs` 应接近参考趴姿 |

---

## 10. Gateway 重要说明

| # | 规则 | 说明 |
|---|------|------|
| 1 | **不要**加 `--send-enable-on-start` | 电机由 FSM 在 `motor_cmd.mode != 0` 时使能 |
| 2 | **不要**加 `--wait-lowcmd` | 实机会与 `fatu_ctrl` 等待 `lowstate` **死锁** |
| 3 | PC sim2sim 例外 | 若需先启控制器，网关可加 `--wait-lowcmd` |
| 4 | `Ctrl+C` + `--send-disable-on-exit` | 退出时 12 路 type4 失能 |
| 5 | 每秒 `[STAT]` | 收发包计数、解码错误、IMU 帧数等监控 |

---

## 11. 硬件拓扑

### 11.1 串口分配

| 串口 | 设备名 | 负责电机 |
|------|--------|----------|
| A | `/dev/myttyCAN0` | FR(11,21,31) + RR(13,23,33) |
| B | `/dev/myttyCAN1` | FL(12,22,32) + RL(14,24,34) |
| IMU | `/dev/myttyIMU` | 姿态 / 角速度 |

也支持单串口模式（`--serial-port`）。

### 11.2 USB Hub 断线问题

> **实机已确认（Orange Pi 6 Plus）：** CAN 适配器接在 **无独立供电的 USB Hub**（`usb 10-1`）上。Velocity 按 W 行走时内核日志为：

```text
usb 10-1: USB disconnect, device number 2          # 整个 Hub 掉线
usb 10-1.1: USB disconnect, device number 3        # Hub 下设备 1（cdc_acm）
cdc_acm 10-1.1:1.1: acm_start_wb ... failed: -19   # gateway 正在 write，ENODEV
usb 10-1.2: USB disconnect, device number 4        # Hub 下设备 2
usb 10-1: new high-speed USB device number 5 ...   # Hub 重新枚举
cdc_acm 10-1.1:1.0: ttyACM3: USB ACM device        # 设备名可能变化
cdc_acm 10-1.2:1.0: ttyACM4: USB ACM device
```

**根因：**
- **不是** CAN 总线物理故障，也**不是** `fatu_ctrl` / DDS 逻辑 bug
- **是** USB Hub（或供电）在行走大电流/振动下复位，导致 `serial write EIO`
- `dds_to_serial_gateway` 里已打开的 fd 失效，**Space 停走无法自动恢复**，需重启 gateway（必要时重插 USB）
- Passive / FixStand / Velocity（不按 W）正常，与「只有行进指令才掉 Hub」一致

**硬件建议（优先）：**

1. 换 **带外接电源（≥2A）** 的 USB Hub，或把 CAN/IMU 分到板载 **不同 root port**
2. **电机主电源与 USB 5V 完全分离**；固定 USB 线，避免行走拽线
3. 配置 udev 固定名（见 `serial_dds_gateway/udev/99-fatu-serial.rules.example`），避免重枚举后 `ttyACM` 号变化
4. 断线后执行：`dmesg \| tail -30`、`ls -l /dev/myttyCAN0 /dev/myttyCAN1`；若无 symlink 需重插或 `sudo udevadm trigger`

**软件缓解（不能替代供电）：**

- gateway：`--tick-hz 50`～`100`（Velocity 实机）
- `fatu_ctrl`：vx 渐变、降低 `deploy.yaml` 的 `stiffness/damping`、限制关节动作幅度

**对照实验：**

| 实验 | 结果 |
|------|------|
| `[3]` 不按 W，扶稳 2–3 min | 通信正常 |
| 按 W 后立刻 Space | 仍断且不恢复 |
| `dmesg` | `usb disconnect` + `cdc_acm ... -19` |

---

## 12. Gateway 主循环（500 Hz）

### 12.1 订阅 `rt/lowcmd` → 串口电机命令

| DDS `motor_cmd` | 串口帧 | 说明 |
|-----------------|--------|------|
| `mode` 0→非 0 | **type3** 使能 | 由 FSM 控制何时上电 |
| `mode` 非 0→0 | **type4** 失能 | 退出控制时关电机 |
| `mode≠0` 时每 tick | **type1** PD 命令 | `q, dq, kp, kd, tau`（经关节映射换算到电机侧） |

电机固件阻抗：`τ = τ_ff + kp·Δq + kd·Δdq`。

### 12.2 收 type2 反馈 → 发布 `rt/lowstate`

- 解析 `q / dq / tau / 温度`
- `motor_state[0..11]` 顺序：**FR, FL, RR, RL**（hip/thigh/calf）
- 经 **sign + bias** 映射后写入 `q`；原始值保留在 `q_raw`
- 未收到反馈的电机 `lost=1`

### 12.3 收 IMU 串口 → `imu_state`

- 帧头：`EB 90 A5 FF` + yaw/pitch/roll + 陀螺仪 + CRC
- 欧拉角 → 四元数（`w,x,y,z`）
- 陀螺仪：串口 `gz,gy,gx` → DDS `gx,gy,gz`
- 静止偏置标定 + 死区

> 无 IMU 时仍可运行，发布单位四元数占位。

---

## 13. 协议层与工具

| 模块 / 程序 | 功能 |
|-------------|------|
| `joint_motor_bias` | sign 翻转 + 趴姿偏置（**无减速比**）；`q_model = sign×q_motor - bias` |
| `imu_gyro_filter` | 陀螺静止偏置 + 死区 |
| `motor_map` | 12 关节 ↔ CAN ID ↔ A/B 总线 |
| `State_FatuFixStand` | Fatu 专用 FixStand（`config.yaml` 中 `type: FatuFixStand`） |
| `lingzu_frame_verify` | 串口帧回归 |
| `imu_frame_verify` / `imu_serial_monitor` | IMU 测试 |
| `twelve_motor_serial` | 双串口 12 电机联调（支持 CSV） |

更多细节见 [serial_dds_gateway/README.md](../serial_dds_gateway/README.md)。

---

## 14. Gateway 参数速查

### 14.1 启动命令速查

#### 方式 A — 在线标定

```bash
cd /path/to/fatuDog/serial_dds_gateway

./build/dds_to_serial_gateway \
  --serial-port-a /dev/myttyCAN0 \
  --serial-port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --imu-port /dev/myttyIMU \
  --imu-baudrate 921600 \
  --network lo \
  --channel 0x00 \
  --master-id 0x00FD \
  --tick-hz 500 \
  --send-disable-on-exit \
  --imu-gyro-calib-seconds 4 \
  --imu-gyro-deadzone 0.03 \
  --joint-bias-calib-seconds 2
```

#### 方式 B — 加载 bias 文件

```bash
cd /path/to/fatuDog/serial_dds_gateway

./build/dds_to_serial_gateway \
  --serial-port-a /dev/myttyCAN0 \
  --serial-port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --imu-port /dev/myttyIMU \
  --imu-baudrate 921600 \
  --network lo \
  --channel 0x00 \
  --master-id 0x00FD \
  --tick-hz 500 \
  --send-disable-on-exit \
  --imu-gyro-calib-seconds 4 \
  --imu-gyro-deadzone 0.03 \
  --joint-bias-load-file config/joint_prone_bias.fatu.txt \
  --no-joint-bias-calib
```

### 14.2 参数说明表

| 参数 | 默认 | 说明 |
|------|------|------|
| `--no-imu-gyro-calib` | 关 | 禁用 IMU 陀螺偏置标定 |
| `--imu-gyro-calib-seconds` | 2 | IMU 标定时长（实机建议 4） |
| `--imu-gyro-deadzone` | 0.03 | 角速度死区 (rad/s) |
| `--no-joint-bias-calib` | 关 | 禁用关节偏置标定 |
| `--joint-bias-calib-seconds` | 2 | 关节偏置标定时长 |
| `--joint-bias-reference-file` | — | 参考趴姿文件 |
| `--joint-bias-load-file` | — | 加载已保存 bias |
| `--send-enable-on-start` | 关 | 启动即 type3 使能（一般不用） |
| `--wait-lowcmd` | 关 | 等 controller 再发 lowstate（实机勿用） |
