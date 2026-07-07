# LegBot Sim2Real 通信架构文档

## 1. 整体架构概览

LegBot 的 sim2real 部署采用 **三层通信架构**，通过 DDS（Data Distribution Service）中间件将策略推理层与硬件驱动层解耦：

```
┌─────────────────────────────────────────────────────────────┐
│                    香橙派 (Orange Pi)                        │
│                                                             │
│  ┌─────────────┐     DDS      ┌──────────────────────┐      │
│  │  fatu_ctrl  │  rt/lowcmd   │  dds_to_serial       │      │
│  │  (策略推理)  │ ──────────→  │  _gateway           │      │
│  │             │              │  (DDS↔Serial桥接)     │      │
│  │  ONNX推理   │  rt/lowstate │                      │      │
│  │  FSM状态机  │ ←──────────  │  type1/2/3/4串口协议  │      │
│  └──────┬──────┘              └──────┬───────┬───────┘      │
│         │                           │       │               │
│         │    unitree_sdk2            │       │               │
│         │    (DDS中间件)             │       │               │
│         │    CycloneDDS              │       │               │
│  ───────┴───────────────────────────┴───────┴──────────────  │
│                                         │       │           │
│                              ┌──────────┘       └─────────┐  │
│                              │                              │  │
│                         /dev/myttyCAN0              /dev/myttyCAN1│
│                         /dev/myttyIMU                        │  │
└─────────────────────────────────┼──────────────────────────────┘
│                    │
              ┌───────┴───────┐
              │  灵足 RS02    │
              │  USB-CAN×2    │
              │  + IMU串口    │
              └───────┬───────┘
                      │
            ┌─────────┴─────────┐
            │   12路电机 + IMU   │
            │   (LegBot物理机器人)  │
            └───────────────────┘
```

### 1.1 三个核心模块

| 模块 | 路径 | 语言 | 职责 |
|------|------|------|------|
| **serial_dds_gateway** | `serial_dds_gateway/` | C++17 | DDS↔串口桥接：将 DDS `rt/lowcmd` 转为灵足串口 type1 命令，将串口 type2 反馈转为 DDS `rt/lowstate` |
| **unitree_rl_lab/deploy** | `unitree_rl_lab/deploy/robots/fatu/` | C++17 | 策略推理控制器：加载 ONNX 策略，运行 FSM 状态机，发布 `rt/lowcmd`，订阅 `rt/lowstate` |
| **unitree_sdk2** | `unitree_sdk2/` | C++ | DDS 通信中间件：基于 CycloneDDS 的 Publisher/Subscriber 模式，提供 Go2 兼容的 LowCmd/LowState 消息接口 |

### 1.2 数据流

```
训练 (IsaacLab)
  │
  ▼
策略导出 (PyTorch .pt → ONNX + deploy.yaml)
  │
  ▼
fatu_ctrl 启动
  │
  ├─→ 加载 policy.onnx + deploy.yaml
  ├─→ 初始化 FSM (Passive → FixStand → Velocity → LieDown)
  │
  │   ┌─── Velocity 状态 ───┐
  │   │ 1. 读取 rt/lowstate  │ → 角速度、重力投影、关节角/速度
  │   │ 2. 构建观测向量       │
  │   │ 3. ONNX 推理         │ → 12维关节位置增量
  │   │ 4. action × scale + offset → 目标关节角
  │   │ 5. 填充 rt/lowcmd    │ → motor_cmd[].q/kp/kd/tau/mode
  │   └─────────────────────┘
  │
  ▼
DDS (rt/lowcmd)
  │
  ▼
dds_to_serial_gateway
  │
  ├─→ 检测 mode 边沿 → 发 type3(使能) / type4(失能)
  ├─→ 每 tick (500Hz) 逐电机发 type1 命令帧
  │     ┌─ joint_bias 映射 (model→motor 空间)
  │     │  q_motor = (q_model + bias) / (sign × gear)
  │     └─ 量化为 16bit 整数，组装串口帧
  │
  ├─→ 串口 A (/dev/myttyCAN0): FR_hip(11), FR_thigh(21), FR_calf(31)
  │                             RR_hip(13), RR_thigh(23), RR_calf(33)
  ├─→ 串口 B (/dev/myttyCAN1): FL_hip(12), FL_thigh(22), FL_calf(32)
  │                             RL_hip(14), RL_thigh(24), RL_calf(34)
  ├─→ IMU串口 (/dev/myttyIMU): 欧拉角 → 四元数 + 陀螺仪滤波
  │
  ▼
电机反馈 (type2) → 解码 → joint_bias 逆映射 (motor→model) → rt/lowstate
IMU 数据 → 四元数 + 陀螺仪 → rt/lowstate
  │
  ▼
DDS (rt/lowstate) → fatu_ctrl 读取 → 下一轮推理
```

---

## 2. 模块详解

### 2.1 unitree_sdk2 — DDS 通信中间件

**核心作用**：提供进程间 DDS 通信，使 `fatu_ctrl` 与 `dds_to_serial_gateway` 可以在同一主机（香橙派）或跨主机通信。

#### 2.1.1 DDS Topic 与消息类型

| Topic | 方向 | 消息类型 | 发布者 | 订阅者 |
|-------|------|----------|--------|--------|
| `rt/lowcmd` | 控制命令 | `unitree_go::msg::dds_::LowCmd_` | fatu_ctrl | dds_to_serial_gateway |
| `rt/lowstate` | 状态反馈 | `unitree_go::msg::dds_::LowState_` | dds_to_serial_gateway | fatu_ctrl |

#### 2.1.2 LowCmd 消息结构

```cpp
// rt/lowcmd — 控制器发布，网关订阅
struct LowCmd_ {
    std::array<uint8_t, 2> head;       // {0xFE, 0xEF}
    uint8_t level_flag;                // 0xFF
    uint16_t reserve;
    uint32_t crc;                      // CRC32 校验
    std::array<MotorCmd_, 12> motor_cmd;  // 12路电机命令
};

struct MotorCmd_ {
    uint8_t  mode;    // 0=失能, 1=使能(位置控制)
    float     q;       // 目标关节角度 [rad] (model空间)
    float     dq;     // 目标关节速度 [rad/s]
    float     tau;    // 前馈力矩 [N·m]
    float     kp;     // 位置刚度 [N·m/rad]
    float     kd;     // 阻尼 [N·m·s/rad]
    uint32_t  reserve;
};
```

#### 2.1.3 LowState 消息结构

```cpp
// rt/lowstate — 网关发布，控制器订阅
struct LowState_ {
    std::array<uint8_t, 2> head;
    uint8_t level_flag;
    uint16_t reserve;
    uint32_t crc;
    IMUState_ imu_state;                    // IMU数据
    std::array<MotorState_, 12> motor_state; // 12路电机反馈
    std::array<uint8_t, 40> wireless_remote;  // 遥控器
    uint32_t tick;
};

struct IMUState_ {
    std::array<float, 4> quaternion;  // [w, x, y, z]
    std::array<float, 3> gyroscope;   // [gx, gy, gz] body frame [rad/s]
    std::array<float, 3> accelerometer;
};

struct MotorState_ {
    uint8_t  mode;
    float    q;          // 关节角度 [rad] (model空间, 已做bias映射)
    float    dq;         // 关节速度 [rad/s]
    float    ddq;
    float    tau_est;    // 估计力矩 [N·m]
    float    q_raw;      // 原始电机编码器角度 (未做bias映射)
    float    dq_raw;
    uint8_t  temperature; // 温度 [°C]
    uint32_t lost;       // 0=正常, 1=丢失
};
```

#### 2.1.4 关键源文件

| 文件 | 作用 |
|------|------|
| [go2_pub.h](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/dds_wrapper/robots/go2/go2_pub.h) | LowCmd/LowState Publisher，含 CRC32 计算 |
| [go2_sub.h](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/dds_wrapper/robots/go2/go2_sub.h) | LowCmd/LowState Subscriber，含遥控器解析 |
| [go2.h](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/dds_wrapper/robots/go2/go2.h) | `shutdown()` 关闭宇树默认控制器 |
| [ChannelFactory](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/robot/channel/channel_factory.hpp) | DDS 域初始化：`Init(domain_id, network_interface)` |

#### 2.1.5 DDS 初始化

```cpp
// 两个进程使用相同的 domain_id(0) 和 network 接口
unitree::robot::ChannelFactory::Instance()->Init(0, "lo");  // 本机回环
// 或
unitree::robot::ChannelFactory::Instance()->Init(0, "eth0"); // 跨主机
```

---

### 2.2 serial_dds_gateway — DDS↔串口桥接

**核心作用**：将 DDS 层的标准 LowCmd/LowState 消息转换为灵足 RS02 USB-CAN 串口协议帧，驱动 12 路电机并采集 IMU 数据。

#### 2.2.1 串口协议帧格式

灵足 USB-CAN 串口帧统一格式：
```
45 54 [channel] [frame_type] [id_field(2B)] [can/master_id] [dlc] [data(0-8B)] 0D 0A
```

| 帧类型 | frame_type | 方向 | 用途 |
|--------|-----------|------|------|
| **type1** | `0x01` | 主机→电机 | 位置控制命令 (q, dq, kp, kd, tau) |
| **type2** | `0x02` | 电机→主机 | 反馈数据 (q, dq, tau, temperature) |
| **type3** | `0x03` | 主机→电机 | 使能电机 |
| **type4** | `0x04` | 主机→电机 | 失能/清故障 |

#### 2.2.2 type1 命令帧 (标准帧)

```
45 54 00 01 [TT TT] 20 08 [Q Q] [V V] [KP KP] [KD KD] [TAU TAU] 0D 0A
      │      │    │  │  └─ 8字节数据: q/dq/kp/kd/tau 各16bit量化
      │      │    │  └─ CAN ID = 0x20 (标准帧)
      │      │    └─ DLC = 8
      │      └─ 目标力矩 (主控ID, 此处为简化)
      └─ channel
```

**量化范围**（[protocol_codec.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/protocol_codec.hpp)）：
| 参数 | 最小值 | 最大值 | 位数 |
|------|--------|--------|------|
| q | -4π | 4π | 16 bit |
| dq | -44 | 44 | 16 bit |
| kp | 0 | 500 | 16 bit |
| kd | 0 | 5 | 16 bit |
| tau | -17 | 17 | 16 bit |

#### 2.2.3 type2 反馈帧 (扩展帧)

```
45 54 01 02 [00 20] [FD] 08 [Q Q] [V V] [T T] [TEMP] 0D 0A
                            │     │     │     │
                            │     │     │     └─ 温度 (16bit, ÷10 = °C)
                            │     │     └─ 力矩 (16bit)
                            │     └─ 速度 (16bit)
                            └─ 位置 (16bit)
```

#### 2.2.4 电机 ID 映射

双串口总线分配（[motor_map.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/motor_map.hpp)）：

| 总线 | 串口设备 | 电机ID | 关节 |
|------|---------|--------|------|
| A | `/dev/myttyCAN0` | 11, 21, 31 | FR_hip, FR_thigh, FR_calf |
| A | `/dev/myttyCAN0` | 13, 23, 33 | RR_hip, RR_thigh, RR_calf |
| B | `/dev/myttyCAN1` | 12, 22, 32 | FL_hip, FL_thigh, FL_calf |
| B | `/dev/myttyCAN1` | 14, 24, 34 | RL_hip, RL_thigh, RL_calf |

#### 2.2.5 Joint Bias 映射 (model↔motor 空间)

**关键**：仿真中的关节角度空间（model）与物理电机编码器空间（motor）之间存在符号、减速比和偏置差异。

映射公式（[joint_motor_bias.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/joint_motor_bias.hpp)）：

```
# motor → model (反馈解码)
q_model = sign[i] × gear_scale[i] × q_motor - bias[i]

# model → motor (命令编码)
q_motor = (q_model + bias[i]) / (sign[i] × gear_scale[i])
```

参数：
| 参数 | 说明 | 值 |
|------|------|-----|
| `sign` | 编码器方向 vs URDF/model | FR/RR = +1, FL/RL thigh+calf = -1 |
| `gear_scale` | 减速比 (motor:joint) | hip/thigh = 1:1, calf = 2:1 |
| `bias` | 趴姿偏置 | 可在线标定或从文件加载 |

#### 2.2.6 IMU 串口解析

IMU 独立串口 `/dev/myttyIMU`，帧格式：
```
EB 90 A5 FF [yaw(4B)] [pitch(4B)] [roll(4B)] [gz(4B)] [gy(4B)] [gx(4B)] [CRC16] 80 7F
```
- 欧拉角为 float32 小端序 (radians)
- 陀螺仪顺序为 gz, gy, gx → DDS 发布为 body frame gx, gy, gz
- 支持静止偏置标定（默认2秒）和死区滤波（默认0.1 rad/s）

#### 2.2.7 网关主循环

[dds_to_serial_gateway.cpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/src/dds_to_serial_gateway.cpp) 主循环 (500Hz)：

1. 读取 `rt/lowcmd` 快照
2. 逐电机检查 `motor_cmd[i].mode` 边沿：
   - 0→1: 发 type3 使能帧
   - 1→0: 发 type4 失能帧
3. 对已使能电机，每 tick 发 type1 位置命令（经 joint_bias 映射）
4. 快照 RX 线程的电机/IMU 缓存
5. 在线 joint bias 标定（若启用）
6. 填充并发布 `rt/lowstate`（经 joint_bias 逆映射）
7. 每秒打印统计信息

#### 2.2.8 线程模型

```
┌─────────────────────────────────────┐
│         dds_to_serial_gateway        │
├─────────────────────────────────────┤
│  主线程 (500Hz tick loop)           │
│    ├─ DDS lowcmd → type1 串口发送    │
│    ├─ 电机/IMU 缓存快照             │
│    └─ DDS lowstate 发布             │
│                                     │
│  RX线程A (1ms轮询)                   │
│    └─ 串口A type2 → motor_cache      │
│                                     │
│  RX线程B (1ms轮询)                   │
│    └─ 串口B type2 → motor_cache      │
│                                     │
│  IMU线程 (1ms轮询)                   │
│    └─ IMU串口 → imu_cache            │
└─────────────────────────────────────┘
```

---

### 2.3 unitree_rl_lab/deploy — 策略推理控制器

**核心作用**：加载训练好的 ONNX 策略模型，通过 FSM 状态机管理机器人行为，读取传感器数据构建观测向量，推理得到关节目标命令。

#### 2.3.1 FSM 状态机

[config.yaml](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/config/config.yaml) 定义了4个状态：

```
Passive (1) ──LT+A──→ FixStand (2) ──start──→ Velocity (3) ──LT+B──→ Passive
                        │                        │
                        └──LT+B──→ Passive ←──LT+B──┘
                                      │
                                      └──LT+B──→ LieDown (4)
```

| 状态 | 说明 | motor mode | 控制方式 |
|------|------|-----------|---------|
| Passive | 趴姿，电机失力 | mode=1, kp=0, kd=3 | 阻尼悬挂 |
| FixStand | 站立到位 | mode=1, kp=60, kd=4 | PD到站立角 |
| Velocity | 策略控制行走 | mode=1, kp/kd=策略输出 | ONNX推理 |
| LieDown | 躺下 | mode=1, kp=40, kd=4 | PD到趴姿 |

#### 2.3.2 策略推理流程 (State_RLBase)

[State_RLBase.cpp](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/src/State_RLBase.cpp) 中 `enter()` 启动策略线程：

```
1. env->reset()  — 初始化环境
2. process_action(0)  — 初始动作为零
3. 循环 (50Hz, step_dt=0.02s):
   a. env->step()
      - 读取 lowstate → robot->update()
      - 构建观测向量 (base_ang_vel, projected_gravity, velocity_cmd, joint_pos_rel, joint_vel_rel, last_action)
      - ONNX 推理 → 12维 action
      - action × scale(0.25) + offset(default_joint_pos) → 目标关节角
   b. 平滑过渡: FixStand kp/kd → RL kp/kd (1.5s)
   c. 填充 lowcmd.motor_cmd[i].q = action[i]
   d. CSV 日志记录 (可选)
```

#### 2.3.3 机器人抽象层

[unitree_articulation.h](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/include/unitree_articulation.h) 将 DDS LowState 转换为策略可用的观测：

```cpp
void update() override {
    // IMU 角速度 → root_ang_vel_b
    for(int i=0; i<3; i++)
        data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];

    // IMU 四元数 → projected_gravity_b
    data.root_quat_w = Quaternionf(quaternion[w,x,y,z]);
    data.projected_gravity_b = quat.conjugate() * GRAVITY_VEC_W;

    // 关节角/速度 (按 joint_ids_map 重排)
    for(int i=0; i<12; i++) {
        data.joint_pos[i] = lowstate->msg_.motor_state()[map[i]].q();
        data.joint_vel[i] = lowstate->msg_.motor_state()[map[i]].dq();
    }
}
```

#### 2.3.4 关节顺序映射

**训练策略的关节顺序**与 **DDS LowState 的电机顺序**不同，需要通过 `joint_ids_map` 重排：

| 策略索引 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---------|---|---|---|---|---|---|---|---|---|---|----|----|
| 策略关节 | FL_hip | FL_thigh | FL_calf | FR_hip | FR_thigh | FR_calf | RL_hip | RL_thigh | RL_calf | RR_hip | RR_thigh | RR_calf |
| DDS索引 | 3 | 0 | 9 | 6 | 4 | 1 | 10 | 7 | 5 | 2 | 11 | 8 |

> **注意**：Fatu 部署使用 FR,FL,RR,RL 的 DDS 电机顺序，而 LegBot 训练使用 FL,FR,RL,RR 的策略关节顺序。

#### 2.3.5 deploy.yaml 配置

[deploy.yaml](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/logs/rsl_rl/unitree_fatu_velocity/2026-06-25_17-11-16/params/deploy.yaml) 是训练导出的部署配置：

```yaml
joint_ids_map: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]  # 策略→DDS关节映射
step_dt: 0.02                                            # 推理周期 50Hz
stiffness: [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
damping: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
default_joint_pos: [0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8]
actions:
  JointPositionAction:
    scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    offset: [0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8]
observations:
  base_ang_vel:     {scale: [0.2, 0.2, 0.2]}
  projected_gravity:{scale: [1, 1, 1]}
  velocity_commands:{scale: [1, 1, 1]}
  joint_pos_rel:    {scale: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
  joint_vel_rel:    {scale: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]}
  last_action:      {scale: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

#### 2.3.6 编译依赖

[CMakeLists.txt](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/CMakeLists.txt)：

```cmake
target_link_libraries(fatu_ctrl
    fatu_controller_lib     # 本地静态库 (State_RLBase.cpp等)
    unitree_sdk2            # DDS 通信
    ddscxx                  # CycloneDDS C++
    ddsc                    # CycloneDDS C
    ${ONNXRUNTIME_LIB}      # ONNX 推理引擎
    boost_program_options   # 命令行解析
    yaml-cpp                # 配置文件解析
    fmt                     # 格式化
    Eigen3                  # 矩阵运算
)
```

---

## 3. 实机部署操作步骤

### 3.1 硬件准备

1. **香橙派** (Orange Pi 5/5Plus) 已安装 Ubuntu 系统
2. **灵足 RS02 USB-CAN** 双通道（已通过 udev 映射为 `/dev/myttyCAN0` 和 `/dev/myttyCAN1`）
3. **IMU 串口**（映射为 `/dev/myttyIMU`）
4. **LegBot 物理机器人**：12 路电机已正确接线

### 3.2 软件依赖安装

```bash
# 系统依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# 编译安装 unitree_sdk2
cd /home/fatu08/go2_rl_robotlab/unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install    # 安装到 /usr/local

# CycloneDDS (unitree_sdk2 的依赖，通常已随 SDK 安装)
```

### 3.3 配置 udev 规则

```bash
# 复制模板并填入实际 USB 设备属性
cp serial_dds_gateway/udev/99-fatu-serial.rules.example /etc/udev/rules.d/99-fatu-serial.rules
# 编辑文件，填入 USB-CAN 和 IMU 适配器的 vendor/product ID
sudo udevadm control --reload-rules
sudo udevadm trigger
```

验证设备节点：
```bash
ls -la /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU
```

### 3.4 编译网关和控制器

```bash
# 编译 serial_dds_gateway
cd /home/fatu08/go2_rl_robotlab/serial_dds_gateway
cmake -S . -B build
cmake --build build -j

# 编译 fatu_ctrl
cd /home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu
mkdir -p build && cd build
cmake .. && make -j
```

### 3.5 导出策略模型

#### 3.5.1 Fatu (unitree_rl_lab) 训练的模型

训练时自动导出（通过 `export_deploy_cfg.py`）：
- `exported/policy.onnx` — ONNX 策略模型
- `params/deploy.yaml` — 部署配置

确认导出：
```bash
ls logs/rsl_rl/unitree_fatu_velocity/<timestamp>/exported/policy.onnx
ls logs/rsl_rl/unitree_fatu_velocity/<timestamp>/params/deploy.yaml
```

#### 3.5.2 LegBot (go2_rl_robotlab / MoE-CTS) 训练的模型

**当前状态**：LegBot 使用 MoE-CTS 算法训练，目前**缺少自动导出 ONNX 和 deploy.yaml 的步骤**。需要手动完成：

1. **导出 student 策略为 ONNX**：

```python
# 需要在训练环境中执行
import torch

# 加载训练 checkpoint
checkpoint = torch.load("logs/rsl_rl/legbot_moe_cts/<timestamp>/model_154500.pt")

# 提取 student actor (非 teacher)
student_actor = runner.alg.student_actor  # 或 runner.model.student_actor
student_actor.eval()

# 导出为 ONNX
dummy_obs = torch.randn(1, obs_dim)  # 根据 env.yaml 的观测维度
dummy_latent = torch.randn(1, latent_dim)  # MoE-CTS 的 latent 输入

torch.onnx.export(
    student_actor,
    (dummy_obs, dummy_latent),
    "logs/rsl_rl/legbot_moe_cts/<timestamp>/exported/policy.onnx",
    input_names=["obs", "latent"],
    output_names=["action"],
    opset_version=14
)
```

2. **手动创建 deploy.yaml**：

根据 [env.yaml](file:///home/fatu08/go2_rl_robotlab/logs/rsl_rl/legbot_moe_cts/2026-06-27_11-08-53/params/env.yaml) 提取关键参数：

```yaml
# LegBot deploy.yaml (需手动创建)
joint_ids_map: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
step_dt: 0.02  # sim.dt(0.005) × decimation(4) = 0.02
stiffness: [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
damping: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
default_joint_pos: [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8]
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-1.0, 1.0]
      lin_vel_y: [-0.4, 0.4]
      ang_vel_z: [-1.0, 1.0]
      heading: null
actions:
  JointPositionAction:
    clip: [[-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100]]
    joint_names: null
    scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    offset: [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8]
    joint_ids: null
observations:
  base_ang_vel:
    params: {}
    clip: [-100, 100]
    scale: [0.25, 0.25, 0.25]
    history_length: 1
  projected_gravity:
    params: {}
    clip: [-100, 100]
    scale: [1.0, 1.0, 1.0]
    history_length: 1
  velocity_commands:
    params: {command_name: base_velocity}
    clip: [-100, 100]
    scale: [1.0, 1.0, 1.0]
    history_length: 1
  joint_pos_rel:
    params: {}
    clip: [-100, 100]
    scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    history_length: 1
  joint_vel_rel:
    params: {}
    clip: [-100, 100]
    scale: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    history_length: 1
  last_action:
    params: {}
    clip: [-100, 100]
    scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    history_length: 1
```

> **注意 LegBot 与 Fatu 的关键差异**：
> - LegBot stiffness=20, damping=0.5 (Fatu: stiffness=40, damping=2.0)
> - LegBot calf effort_limit=34N·m (Fatu: 可能不同)
> - LegBot 观测含 height_scanner (teacher only, 部署不需要)
> - LegBot 策略有 history_length=10 (部署需处理观测堆叠)

### 3.6 配置 config.yaml

编辑 [config.yaml](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/config/config.yaml)：

```yaml
FSM:
  Velocity:
    policy_dir: ../../../logs/rsl_rl/legbot_moe_cts/<timestamp>  # 指向 LegBot 训练目录
```

### 3.7 启动部署

#### 3.7.1 Sim2Sim (MuJoCo 验证)

先用 MuJoCo 仿真验证策略：

```bash
# 终端1: 启动 MuJoCo 仿真 (如果已安装 unitree_mujoco)
cd unitree_mujoco/simulate/build
./unitree_mujoco  # config.yaml 中 robot 设为 "fatu"

# 终端2: 启动控制器
cd unitree_rl_lab/deploy/robots/fatu/build
./fatu_ctrl --network lo
```

#### 3.7.2 Sim2Real (实机部署)

```bash
# 终端1: 启动 DDS-串口网关
cd /home/fatu08/go2_rl_robotlab/serial_dds_gateway
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
  --send-disable-on-exit

# 终端2: 启动策略控制器
cd /home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/build
./fatu_ctrl --network lo
```

或使用快捷脚本：
```bash
./serial_dds_gateway/start_gateway.sh
./serial_dds_gateway/start_ctrl.sh
```

#### 3.7.3 操作流程

1. 网关启动后打印 `[PHASE1]` 日志，确认串口和 IMU 正常
2. `fatu_ctrl` 启动后等待 `rt/lowstate` 连接
3. 键盘操作：
   - `LT + A`：Passive → FixStand（站立）
   - `start`：FixStand → Velocity（策略控制）
   - `W/S`：前进/后退
   - `A/D`：左移/右移
   - `Q/E`：左转/右转
   - `Space`：停止运动
   - `LT + B`：回到 Passive

---

## 4. LegBot Sim2Real 的关键差异与待解决问题

### 4.1 关节顺序差异

| 层面 | 关节顺序 | 说明 |
|------|---------|------|
| LegBot 训练 (env.yaml) | FL, FR, RL, RR | 策略输出和观测的关节顺序 |
| Fatu 部署 (deploy.yaml) | FR, FL, RR, RL | DDS LowState 的电机顺序 |
| 灵足电机 CAN ID | FR=11,21,31 / FL=12,22,32 / RR=13,23,33 / RL=14,24,34 | 物理总线分配 |

`joint_ids_map: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]` 负责策略关节→DDS电机的重排。

### 4.2 MoE-CTS 策略导出

LegBot 使用 MoE-CTS（Mixture of Experts with Cross-Token-Synthesis）算法，采用 teacher-student 架构：

- **Teacher**：使用完整观测（含 height_scanner、base_lin_vel 等特权信息）训练
- **Student**：使用部署可用观测（IMU、关节角/速度、速度指令）推理

部署时**只需要 student 策略**的 ONNX 导出。当前训练日志中没有自动导出步骤，需要：

1. 在训练环境中添加 ONNX 导出脚本
2. 确保 student actor 的输入输出维度与 deploy.yaml 一致
3. 处理 MoE-CTS 的 latent 输入（推理时可能需要固定或从观测推断）

### 4.3 观测历史堆叠

LegBot 训练使用 `history_length: 10` 的观测堆叠，而 Fatu 部署框架默认 `history_length: 1`。有两种解决方案：

**方案A**：修改 deploy.yaml 为 `history_length: 10`（如果部署框架支持）
**方案B**：导出时将 history 维度展平到 ONNX 输入，部署时维护观测历史缓冲区

### 4.4 物理参数对齐

| 参数 | LegBot 仿真 (env.yaml) | LegBot URDF (本次修改后) | 说明 |
|------|---------------------|----------------------|------|
| 总质量 | — | 14.0 kg | 已调整 |
| base 高度 | 0.36 m (init pos) | ~0.28 m (BASE_HEIGHT_TARGET) | 需确认 |
| hip/thigh kp | 20 | — | 仿真 PD 增益 |
| hip/thigh kd | 0.5 | — | 仿真 PD 阻尼 |
| calf kp | 20 | — | calf 减速比 2:1 |
| calf kd | 0.5 | — | |
| calf effort_limit | 34 N·m | — | 物理电机力矩限制 |

### 4.5 Joint Bias 标定

首次部署 LegBot 物理机器人时，需要标定 joint bias：

```bash
# 在线标定模式（趴姿保持静止2秒）
./build/dds_to_serial_gateway \
  --joint-bias-calib \
  --joint-bias-calib-seconds 2.0 \
  --joint-bias-reference "-0.02,1.08,-2.64,0.03,1.08,-2.64,-0.05,1.08,-2.64,0.06,1.08,-2.64" \
  ...其他参数

# 或加载已标定文件
./build/dds_to_serial_gateway \
  --joint-bias-load-file config/joint_prone_bias.fatu.txt \
  ...其他参数
```

---

## 5. 调试与验证

### 5.1 串口帧验证

```bash
# 验证灵足串口编码
./build/lingzu_frame_verify

# 验证 IMU 串口解析
./build/imu_frame_verify

# 监控 IMU 实时数据
./build/imu_serial_monitor --port /dev/myttyIMU --baudrate 921600 --degrees

# 单电机测试
./build/one_motor_serial --port /dev/myttyCAN0 --motor-id 11 --send-enable --q 0.0 --kp 0 --kd 0.5

# 12电机测试
./build/twelve_motor_serial --port-a /dev/myttyCAN0 --port-b /dev/myttyCAN1 --send-enable --disable-on-exit --q 0 --dq 0 --kp 0 --kd 0.5 --tx-hz 50 --rx-seconds 3
```

### 5.2 DDS 通信验证

```bash
# 检查 DDS topic
ddsls lo  # 如果安装了 cyclonedds 工具

# 网关统计 (每秒打印)
# [STAT] rx_frames=... type2_frames=... tx_type1=... imu_frames=...
```

### 5.3 CSV 日志分析

```bash
# 启动时加 --csv-log
./fatu_ctrl --network lo --csv-log --experiment-group A

# 日志位于 <fatuDog>/log/
# run_*.csv: Velocity 状态的逐帧数据
# fixstand_*.csv: FixStand 状态数据
```

CSV 包含字段：timestamp, vx, vy, wz, gravity, angular_velocity, joint_pos_rel, joint_vel, action, q_target, q_actual, q_motor_raw, tau_est, kp, kd

---

## 6. 文件索引

### serial_dds_gateway
| 文件 | 作用 |
|------|------|
| [dds_to_serial_gateway.cpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/src/dds_to_serial_gateway.cpp) | 网关主程序 |
| [protocol_codec.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/protocol_codec.hpp) | type1/type2 编解码 + 量化范围 |
| [lingzu_motor_protocol.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/lingzu_motor_protocol.hpp) | 灵足串口帧封装 |
| [serial_framer.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/serial_framer.hpp) | 串口帧收发 (45 54...0D 0A) |
| [motor_map.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/motor_map.hpp) | 12关节 CAN ID 映射 |
| [joint_motor_bias.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/joint_motor_bias.hpp) | model↔motor 空间映射 (sign/gear/bias) |
| [imu_framer.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/imu_framer.hpp) | IMU 串口帧解析 |
| [imu_gyro_filter.hpp](file:///home/fatu08/go2_rl_robotlab/serial_dds_gateway/include/imu_gyro_filter.hpp) | IMU 陀螺仪偏置标定 + 死区 |

### unitree_rl_lab/deploy
| 文件 | 作用 |
|------|------|
| [main.cpp](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/main.cpp) | fatu_ctrl 入口, DDS 初始化, FSM 启动 |
| [State_RLBase.cpp](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/src/State_RLBase.cpp) | Velocity 状态: ONNX 推理 + 观测构建 |
| [config.yaml](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/robots/fatu/config/config.yaml) | FSM 配置 + policy_dir |
| [unitree_articulation.h](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/include/unitree_articulation.h) | LowState → 策略观测数据转换 |
| [deploy_joint_layout.h](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/include/deploy_joint_layout.h) | 策略→电机关节索引映射 |
| [deploy_experiment.h](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/deploy/include/deploy_experiment.h) | A/B/C 实验组 (obs/action 参考系) |
| [export_deploy_cfg.py](file:///home/fatu08/go2_rl_robotlab/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py) | 训练时自动导出 deploy.yaml |

### unitree_sdk2
| 文件 | 作用 |
|------|------|
| [go2_pub.h](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/dds_wrapper/robots/go2/go2_pub.h) | LowCmd/LowState DDS Publisher |
| [go2_sub.h](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/dds_wrapper/robots/go2/go2_sub.h) | LowCmd/LowState DDS Subscriber |
| [LowCmd_.hpp](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/idl/go2/LowCmd_.hpp) | LowCmd IDL 消息定义 |
| [LowState_.hpp](file:///home/fatu08/go2_rl_robotlab/unitree_sdk2/include/unitree/idl/go2/LowState_.hpp) | LowState IDL 消息定义 |
