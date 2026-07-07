# Legbot Sim2Real 通信部署指南

> 本文档描述 Legbot 四足机器人从仿真到实机的完整通信链路、SSH 连接香橙派、编译与运行的详细步骤。

---

## 一、系统架构与通信原理

### 1.1 整体通信链路

```
┌──────────────────┐   DDS (rt/lowcmd, rt/lowstate)   ┌────────────────────────┐  串口   ┌─────────────┐
│  legbot_ctrl       │◄────────────────────────────────►│  serial_dds_gateway    │◄──────►│ 12电机 + IMU │
│  (RL策略控制器)  │           CycloneDDS             │  (DDS↔串口协议网关)    │ type1-4 │ (凌足USB-CAN)│
└──────────────────┘                                  └────────────────────────┘         └─────────────┘
        │                                                          │
        │ 运行在香橙派上                                            │ 运行在香橙派上
        │ 1kHz 控制循环                                            │ 500Hz 协议转换
        └────────────────── 共享 lo 网卡 ──────────────────────────┘
```

### 1.2 三个核心组件

| 组件 | 路径 | 作用 |
|------|------|------|
| **legbot_ctrl** | `legbot_rl_lab/deploy/robots/legbot/` | 加载 ONNX 策略模型，1kHz 运行 RL 控制，发布 `rt/lowcmd` |
| **serial_dds_gateway** | `serial_dds_gateway/` | DDS↔串口协议网关，500Hz，订阅 `rt/lowcmd` 发布 `rt/lowstate` |
| **unitree_sdk2** | `unitree_sdk2/` | 宇树 DDS 中间件（CycloneDDS 封装），提供 `LowCmd`/`LowState` 消息定义 |

### 1.3 DDS 数据流

**下行（控制器 → 网关）**：控制器发布 `rt/lowcmd`，每条消息含 12 个电机的：
- `mode`：电机使能标志（0=失能，1=使能）
- `q`：目标关节角度（rad，模型空间）
- `dq`：目标关节速度（rad/s）
- `kp`：位置刚度
- `kd`：阻尼
- `tau`：前馈力矩（Nm）

**上行（网关 → 控制器）**：网关发布 `rt/lowstate`，含：
- 12 个电机的反馈：`q`（当前位置）、`dq`（当前速度）、`tau_est`（估计力矩）、`temperature`（温度）、`lost`（是否掉线）
- IMU 状态：`quaternion`（四元数 w,x,y,z）、`gyroscope`（陀螺仪 body 系）

### 1.4 网关协议转换原理

网关内部完成 **4 道加工工序**：

1. **mode 边沿检测使能**：检测 `mode` 从 0→1 发 type3 使能帧，1→0 发 type4 失能帧
2. **16 位量化**：浮点命令（q/dq/kp/kd/tau）量化为 16 位整数发串口（type1 帧）
3. **joint bias 模型/电机空间转换**：
   - 小腿 2:1 减速比：`q_motor = 2 × q_model`
   - 左侧腿符号翻转：FL/RL 的 thigh + calf 取反
   - 趴姿偏置标定：上电时编码器非零，需标定偏置
4. **IMU 欧拉角 → 四元数 + 陀螺滤波**：
   - 欧拉角 ZYX 顺序转四元数：`q = R_z(yaw)·R_y(pitch)·R_x(roll)`
   - 陀螺仪静止偏置标定（2 秒）+ 死区滤波（0.1 rad/s）

### 1.5 串口总线分组

| 串口设备 | 电机分组 | CAN ID |
|---------|---------|---------|
| `/dev/myttyCAN0` (Port A) | FR + RR 两腿 | 11,21,31, 13,23,33 |
| `/dev/myttyCAN1` (Port B) | FL + RL 两腿 | 12,22,32, 14,24,34 |
| `/dev/myttyIMU` | IMU | - |

CAN ID 编码规则：十位=关节位（hip=1x, thigh=2x, calf=3x），个位=腿（FR=1, FL=2, RR=3, RL=4）。

---

## 二、SSH 连接香橙派

### 2.1 硬件前提

- 香橙派（Orange Pi 5/5B/5Plus）已通过 USB-CAN 适配器连接 12 个电机、串口连接 IMU
- 香橙派与控制电脑在同一局域网（或有网线直连）
- 香橙派已开启 SSH 服务

### 2.2 获取香橙派 IP

方法一：路由器后台查看香橙派的 IP 地址。

方法二：香橙派连接显示器键盘，登录后执行：
```bash
ip addr show | grep inet
```

### 2.3 SSH 连接

```bash
# 在控制电脑上执行（替换为你的香橙派 IP 和用户名）
ssh fatu@192.168.x.x

# 首次连接会提示指纹，输入 yes
# 输入密码登录
```

### 2.4 配置 SSH 免密登录（推荐）

避免每次编译运行都输密码：
```bash
# 在控制电脑上生成密钥（已生成可跳过）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 将公钥拷贝到香橙派
ssh-copy-id fatu@192.168.x.x

# 之后直接 ssh fatu@192.168.x.x 免密登录
```

### 2.5 配置 SSH 别名（推荐）

编辑 `~/.ssh/config`：
```
Host legbot
    HostName 192.168.x.x
    User fatu
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

之后直接：
```bash
ssh legbot
scp file legbot:~/path/   # 也用别名
```

---

## 三、环境依赖确认

SSH 登录香橙派后，先确认开发环境已就绪。

### 3.1 系统依赖检查

```bash
# 编译器
g++ --version          # 需要 g++ 9+ (支持 C++17)
cmake --version        # 需要 3.12+

# DDS 中间件 (CycloneDDS)
ls /usr/local/lib/libddscxx.so    # 应存在
ls /usr/local/lib/libddsc.so      # 应存在

# 宇树 SDK
ls ~/.local/lib/libunitree_sdk2.a  # 应存在
ls ~/.local/include/unitree/robot/ # 应有头文件

# 第三方库
dpkg -l | grep -E "libboost|yaml-cpp|libfmt"
# 需要: libboost-program_options, yaml-cpp, fmt
```

### 3.2 缺失依赖安装

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential cmake \
    libboost-all-dev libyaml-cpp-dev libfmt-dev \
    libssl-dev

# 如果 unitree_sdk2 未安装到 ~/.local，参考 unitree_sdk2/README.md 编译安装
```

### 3.3 ONNX Runtime（实机 aarch64 必需）

```bash
# 检查 aarch64 版本 ONNX Runtime 是否就位
ls legbot_rl_lab/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/include/onnxruntime_cxx_api.h
ls legbot_rl_lab/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib/libonnxruntime.so.1.22.0

# 如果缺失，下载 aarch64 版本（香橙派是 ARM64 架构）
cd legbot_rl_lab/deploy/thirdparty/
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.tgz
tar xzf onnxruntime-linux-aarch64-1.22.0.tgz
rm onnxruntime-linux-aarch64-1.22.0.tgz
```

> 注意：x86_64 电脑用 `onnxruntime-linux-x64-1.22.0`，香橙派（aarch64）用 `onnxruntime-linux-aarch64-1.22.0`。CMake 的 `DetectPlatform.cmake` 会自动检测架构选择对应版本。

---

## 四、串口设备配置（udev 规则）

### 4.1 确认 USB-CAN 适配器已被识别

```bash
# 插入 USB-CAN 适配器后查看
ls /dev/ttyUSB*

# 查看设备序列号（用于 udev 规则）
udevadm info -a -n /dev/ttyUSB0 | grep -E 'idVendor|idProduct|serial|KERNELS'
udevadm info -a -n /dev/ttyUSB1 | grep -E 'idVendor|idProduct|serial|KERNELS'
udevadm info -a -n /dev/ttyUSB2 | grep -E 'idVendor|idProduct|serial|KERNELS'
```

### 4.2 配置 udev 固定设备名

```bash
# 复制示例规则
sudo cp legbot_mujoco/serial_dds_gateway/udev/99-fatu-serial.rules.example \
        /etc/udev/rules.d/99-fatu-serial.rules

# 编辑，将 CAN0_SERIAL / CAN1_SERIAL / IMU_SERIAL 替换为实际序列号
sudo nano /etc/udev/rules.d/99-fatu-serial.rules
```

规则文件内容（替换序列号后）：
```
# Motor USB-CAN adapter for bus A: FR=(11,21,31), RR=(13,23,33)
SUBSYSTEM=="tty", ATTRS{serial}=="实际序列号A", SYMLINK+="myttyCAN0", MODE="0666"

# Motor USB-CAN adapter for bus B: FL=(12,22,32), RL=(14,24,34)
SUBSYSTEM=="tty", ATTRS{serial}=="实际序列号B", SYMLINK+="myttyCAN1", MODE="0666"

# IMU serial adapter
SUBSYSTEM=="tty", ATTRS{serial}=="实际序列号C", SYMLINK+="myttyIMU", MODE="0666"
```

### 4.3 生效 udev 规则

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

# 重新插拔 USB 设备，确认软链接
ls -l /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU
# 应显示 -> /dev/ttyUSBx
```

---

## 五、编译程序

### 5.1 编译 serial_dds_gateway（网关）

```bash
cd ~/legbot_mujoco/serial_dds_gateway

# 创建构建目录并编译
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)

# 验证可执行文件
ls -l dds_to_serial_gateway
./dds_to_serial_gateway --help   # 查看参数说明
```

**编译参数说明**（`./dds_to_serial_gateway --help`）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--serial-port-a` | `/dev/myttyCAN0` | Port A 串口（FR+RR） |
| `--serial-port-b` | `/dev/myttyCAN1` | Port B 串口（FL+RL） |
| `--baudrate` | `2000000` | 电机串口波特率 |
| `--imu-port` | `/dev/myttyIMU` | IMU 串口 |
| `--imu-baudrate` | `921600` | IMU 波特率 |
| `--network` | `lo` | DDS 网卡（本机回环用 lo） |
| `--tick-hz` | `500` | 主循环频率 |
| `--joint-bias-calib` | 关闭 | 上电在线标定趴姿偏置 |
| `--joint-bias-load-file` | 空 | 加载已标定偏置文件 |
| `--send-disable-on-exit` | 关闭 | Ctrl+C 时发 type4 失能所有电机 |

### 5.2 编译 legbot_ctrl（控制器）

```bash
cd ~/legbot_mujoco/legbot_rl_lab/deploy/robots/legbot

# 创建构建目录并编译
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j$(nproc)

# 验证可执行文件
ls -l legbot_ctrl
./legbot_ctrl --help
```

**编译参数说明**（`./legbot_ctrl --help`）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--network` | `lo` | DDS 网卡 |
| `--csv-log` | 关闭 | 启用 50Hz CSV 诊断日志 |

### 5.3 一键编译脚本

也可以使用 gateway 自带的脚本：
```bash
cd ~/legbot_mujoco/serial_dds_gateway
./start_gateway.sh   # 自动编译并启动（但不带 IMU 等参数）
```

---

## 六、配置文件准备

### 6.1 控制器配置 config.yaml

路径：`legbot_rl_lab/deploy/robots/legbot/config/config.yaml`

**关键检查项**：

```yaml
# 1. 安全保护参数（按你的机器人调整）
safety:
  enabled: true
  torque_limit: 40.0              # 力矩限幅 Nm
  delta_q_limit_per_tick: 0.05    # 角度变化量限幅 rad/tick
  joint_pos_lower: [-0.5, -1.5, -2.7, ...]   # 关节下限（按实际调整！）
  joint_pos_upper: [ 0.5,  1.5, -0.5, ...]   # 关节上限（按实际调整！）

# 2. 策略模型路径（确认模型文件存在！）
FSM:
  Velocity:
    policy_dir: ../../../logs/rsl_rl/unitree_legbot_velocity/2026-06-25_17-11-16
    # 该目录下必须有 exported/policy.onnx
```

**确认模型文件存在**：
```bash
ls ~/legbot_mujoco/legbot_rl_lab/logs/rsl_rl/unitree_legbot_velocity/2026-06-25_17-11-16/exported/policy.onnx
# 如果不存在，修改 policy_dir 指向实际存在的模型目录
ls ~/legbot_mujoco/legbot_rl_lab/logs/rsl_rl/unitree_legbot_velocity/
```

### 6.2 关节趴姿偏置文件（joint bias）

实机电机编码器零位与模型零位不一致，必须标定。两种方式二选一：

**方式 A：在线标定（首次部署用）**
```bash
# 机器人保持趴姿（四脚朝天或趴在地上，关节自由状态）
# 启动网关时加 --joint-bias-calib，会采集 2 秒均值并保存
./build/dds_to_serial_gateway --joint-bias-calib ...
# 标定结果保存到 config/joint_prone_bias.fatu.txt
```

**方式 B：加载已标定文件（生产用）**
```bash
# 确认偏置文件存在
cat ~/legbot_mujoco/serial_dds_gateway/config/joint_prone_bias.fatu.txt
# 应有 12 行浮点数

# 启动网关时加载
./build/dds_to_serial_gateway --joint-bias-load-file ../config/joint_prone_bias.fatu.txt ...
```

---

## 七、运行步骤

### 7.1 运行前预检

```bash
# 1. 确认串口设备就绪
ls -l /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU

# 2. 确认没有残留进程占用串口或 DDS
cd ~/legbot_mujoco/serial_dds_gateway
./orangepi_legbot_rt_preflight.sh
# 预期：没有 legbot_ctrl / dds_to_serial_gateway / python EX34 等进程

# 3. 确认机器人处于安全状态（趴姿，周围有空间）
# 4. 确认急停按钮就位（如有）
```

### 7.2 启动网关（终端 1，必须先启动）

```bash
cd ~/legbot_mujoco/serial_dds_gateway

# 完整启动命令（推荐）
./build/dds_to_serial_gateway \
    --serial-port-a /dev/myttyCAN0 \
    --serial-port-b /dev/myttyCAN1 \
    --baudrate 2000000 \
    --imu-port /dev/myttyIMU \
    --imu-baudrate 921600 \
    --network lo \
    --tick-hz 500 \
    --joint-bias-load-file config/joint_prone_bias.fatu.txt \
    --send-disable-on-exit
```

**启动后观察日志**，确认以下内容正常：
```
[INFO] Opening serial port A: /dev/myttyCAN0 @ 2000000 baud   ← 串口打开成功
[INFO] Opening serial port B: /dev/myttyCAN1 @ 2000000 baud
[INFO] Opening IMU port: /dev/myttyIMU @ 921600 baud
[INFO] Joint bias loaded from config/joint_prone_bias.fatu.txt
[STAT] rx_frames=... type2_frames=... imu_frames=...           ← 每秒统计
```

**健康指标**：
- `type2_frames` 应持续增长（电机在反馈）
- `imu_frames` 应持续增长（IMU 在反馈）
- `decode_errors` 应接近 0（解析无误）
- `tx_write_errors` 应为 0（USB 未掉线）

如果 `type2_frames` 不增长，说明电机没反馈，检查接线和 CAN ID。

### 7.3 启动控制器（终端 2，网关就绪后）

```bash
cd ~/legbot_mujoco/legbot_rl_lab/deploy/robots/legbot

# 启动（会阻塞等待网关的 rt/lowstate 就绪）
./build/legbot_ctrl --network lo

# 可选：启用 CSV 诊断日志
./build/legbot_ctrl --network lo --csv-log
```

**启动后观察日志**：
```
[INFO] Waiting for DDS connection (rt/lowstate)...   ← 等待网关
[INFO] DDS connected.                                ← 网关就绪
[INFO] Policy loaded: .../policy.onnx                ← 策略模型加载成功
[INFO] Robot fallen into Passive state               ← 初始 Passive
```

### 7.4 手柄操作流程

控制器启动后进入 **Passive** 状态（电机阻尼模式，mode=1, kp=0, kd=3）。

**标准操作流程**：

```
1. Passive 状态
   ├─ 机器人趴着，电机抱阻尼
   └─ 长按 LT + A 键 → 切到 FixStand

2. FixStand 状态
   ├─ 机器人插值站立到默认姿态（1 秒）
   ├─ kp=60, kd=4 位置控制
   └─ 按 start 键 → 切到 Velocity（RL 策略接管）

3. Velocity 状态
   ├─ RL 策略 1kHz 运行
   ├─ 手柄摇杆控制速度方向
   └─ 长按 LT + B 键 → 紧急切回 Passive（阻尼软停机）
```

**手柄键位**：
| 按键 | 状态切换 | 说明 |
|------|---------|------|
| LT(长按) + A | Passive → FixStand | 站起 |
| start | FixStand → Velocity | 策略接管 |
| LT(长按) + B | 任意 → Passive | 紧急软停机 |

### 7.5 安全停机

**正常停机**：
```bash
# 终端2: 长按 LT + B 切回 Passive，机器人趴下
# 终端2: Ctrl+C 退出 legbot_ctrl
# 终端1: Ctrl+C 退出 gateway（--send-disable-on-exit 会发 type4 失能所有电机）
```

**紧急停机**：
- 手柄长按 LT + B 立即切 Passive（阻尼软着陆）
- 或直接 Ctrl+C 两个程序
- 或拍急停按钮（如有硬件急停）

---

## 八、安全保护机制

### 8.1 命令侧限幅（不会停机，只 clip 命令）

每 tick 在 `post_run` 中执行，超限的命令值被削回安全范围，电机继续运行：

| 保护项 | 限值 | 作用 |
|--------|------|------|
| action clip | ±100 | 策略原始输出 clip |
| 关节角度绝对限位 | joint_pos_lower/upper | `q_des` 硬限位 |
| 力矩限幅 | ±40 Nm | `tau` clip |
| 角度变化量限幅 | 0.05 rad/tick | `\|q_des[t] - q_des[t-1]\|` clip，防 NN 突跳 |
| 角速度变化量限幅 | 1.0 rad/s/tick | `\|dq_des[t] - dq_des[t-1]\|` clip |

### 8.2 反馈侧超限保护（切 Passive 阻尼模式，不断电）

触发后切到 Passive 状态（mode=1, kp=0, kd=3），电机主动抱阻尼软着陆：

| 监测项 | 阈值 | 触发动作 |
|--------|------|---------|
| 通信超时 | is_timeout() | → Passive |
| 关节速度 | 30 rad/s | → Passive |
| 反馈力矩 | 45 Nm | → Passive |
| 电机温度 | 80°C | → Passive |
| IMU roll | 0.5 rad (~28°) | → Passive |
| IMU pitch | 0.5 rad (~28°) | → Passive |
| 急停标志 | emergency_stop | → Passive |

> 设计为阻尼模式而非断电（mode=0）：四足摔倒时阻尼模式让腿缓慢落下，避免 freewheel 砸坏关节。

---

## 九、故障排查

### 9.1 网关启动失败

| 现象 | 原因 | 解决 |
|------|------|------|
| `Failed to open /dev/myttyCAN0` | 设备未识别或权限不足 | 检查 udev 规则、`ls -l /dev/mytty*`、`sudo chmod 666` |
| `type2_frames` 不增长 | 电机未上电/接线错/CAN ID 错 | 检查电机电源、CAN 总线、用 `one_motor_serial` 单独测试 |
| `imu_frames` 不增长 | IMU 接线错/波特率错 | 用 `imu_serial_monitor` 单独测试 IMU |
| `decode_errors` 持续增长 | 串口干扰/波特率不匹配 | 检查接地、降低波特率、换屏蔽线 |

### 9.2 控制器启动失败

| 现象 | 原因 | 解决 |
|------|------|------|
| 卡在 `Waiting for DDS connection` | 网关未启动/网卡不对 | 确认网关已运行、两边 `--network` 一致 |
| `Failed to load policy.onnx` | 模型路径错/架构不匹配 | 检查 `policy_dir`、确认是 aarch64 版 onnxruntime |
| `ONNX Runtime` 未找到 | 第三方库缺失 | 见 3.3 节安装 aarch64 版 onnxruntime |

### 9.3 运行中异常

| 现象 | 原因 | 解决 |
|------|------|------|
| 机器人突然趴下（切 Passive） | 触发安全保护 | 查日志看哪个保护触发（temp/velocity/roll...） |
| 关节抖动 | delta 限幅过严/kp 过高 | 调大 `delta_q_limit_per_tick` 或降低 FixStand kp |
| 机器人站不稳 | joint bias 未标定/模型不匹配 | 重新标定趴姿偏置、检查 default_joint_angles |
| DDS 消息延迟 | 网卡负载高 | 实机用独立网卡，避免 lo 上跑其他流量 |

### 9.4 常用调试工具

```bash
# 1. 单电机测试（绕过 gateway，直接串口）
cd ~/legbot_mujoco/serial_dds_gateway/build
./one_motor_serial --port /dev/myttyCAN0 --can-id 11

# 2. IMU 监控（独立串口）
./imu_serial_monitor --port /dev/myttyIMU

# 3. 帧校验工具
./lingzu_frame_verify    # 电机帧
./imu_frame_verify       # IMU 帧

# 4. 查看 DDS 话题（需 cyclonedds 工具）
# 查看是否有 rt/lowcmd / rt/lowstate 流量

# 5. CSV 诊断日志分析
# 启用 --csv-log 后，日志在 legbot_ctrl 运行目录下
# 用 Excel/Python 分析关节轨迹、力矩、温度
```

---

## 十、完整运行示例（一键脚本）

将以下脚本保存为 `~/run_sim2real.sh`（按需修改路径）：

```bash
#!/bin/bash
set -e

GATEWAY_DIR=~/legbot_mujoco/serial_dds_gateway
CTRL_DIR=~/legbot_mujoco/legbot_rl_lab/deploy/robots/legbot

# 终端1: 启动网关
gnome-terminal -- bash -c "
    cd $GATEWAY_DIR
    ./build/dds_to_serial_gateway \
        --serial-port-a /dev/myttyCAN0 \
        --serial-port-b /dev/myttyCAN1 \
        --imu-port /dev/myttyIMU \
        --network lo \
        --tick-hz 500 \
        --joint-bias-load-file config/joint_prone_bias.fatu.txt \
        --send-disable-on-exit
    read -p 'Gateway exited. Press Enter...'
"

sleep 3  # 等网关初始化

# 终端2: 启动控制器
gnome-terminal -- bash -c "
    cd $CTRL_DIR
    ./build/legbot_ctrl --network lo --csv-log
    read -p 'Controller exited. Press Enter...'
"
```

```bash
chmod +x ~/run_sim2real.sh
~/run_sim2real.sh
```

---

## 十一、目录结构速查

```
legbot_mujoco/
├── serial_dds_gateway/          # 网关（DDS↔串口）
│   ├── build/dds_to_serial_gateway   # 网关可执行文件
│   ├── config/joint_prone_bias.fatu.txt  # 趴姿偏置
│   └── start_gateway.sh
├── unitree_sdk2/                # 宇树 DDS 中间件
│   └── lib/aarch64/libunitree_sdk2.a
├── legbot_rl_lab/                 # Legbot 控制器
│   ├── deploy/
│   │   ├── cmake/DetectPlatform.cmake  # 平台检测（x86_64/aarch64）
│   │   ├── include/              # 共享头文件（FSM, safety, param）
│   │   └── robots/legbot/
│   │       ├── build/legbot_ctrl       # 控制器可执行文件
│   │       ├── config/config.yaml    # 配置文件
│   │       └── src/State_RLBase.cpp  # RL 策略状态
│   └── logs/rsl_rl/unitree_legbot_velocity/  # 训练好的模型
│       └── 2026-06-25_17-11-16/exported/policy.onnx
```

---

## 十二、关键注意事项

1. **启动顺序**：必须先启动网关，再启动控制器。控制器启动时会阻塞等待 `rt/lowstate`。
2. **网卡一致**：网关和控制器的 `--network` 参数必须一致（本机都用 `lo`，多机用实际网卡名）。
3. **joint bias 必须标定**：未标定会导致模型空间和电机空间不匹配，机器人站不稳甚至失控。
4. **安全参数按实机调整**：`config.yaml` 中的 `joint_pos_lower/upper` 是示例值，必须按你的 Legbot 实际关节限位修改。
5. **aarch64 onnxruntime**：香橙派是 ARM64 架构，必须用 aarch64 版本的 onnxruntime，x86 版本无法运行。
6. **Ctrl+C 顺序**：先停控制器，再停网关（网关带 `--send-disable-on-exit` 会失能所有电机）。
7. **首次运行建议**：先用 `one_motor_serial` 单独测试每个电机，再用 `imu_serial_monitor` 测 IMU，确认硬件正常后再整体运行。
