# VBot Sim2Real 部署测试说明

本文档说明如何使用测试文件对 VBot 机器人的电机和 IMU 进行 sim2real 部署测试。

---

## 文件结构

```
deploy/robots/vbot/tests/
├── CMakeLists.txt              # C++ 测试构建配置
├── test_utils.h                # 测试工具函数（纯协议逻辑提取）
├── test_motor_protocol.cpp     # 电机 CAN 协议单元测试
└── test_imu_protocol.cpp       # IMU 串口协议单元测试

scripts/
├── test_motor.py               # 电机硬件测试脚本（Python）
├── test_imu.py                 # IMU 硬件测试脚本（Python）
└── build_and_run_tests.sh      # 一键构建和运行脚本
```

---

## 一、C++ 协议单元测试

### 测试内容

#### test_motor_protocol.cpp（~25 个测试用例）
| 测试组 | 测试内容 |
|--------|----------|
| FloatUintConversion | float↔uint 转换（最小值、最大值、中点、钳位、往返一致性、kp/kd 非负范围） |
| CanFrameEncoding | CAN 帧结构（帧头 0x45 0x54、仲裁 ID 大端编码、DLC、帧尾 0x0D 0x0A） |
| MotionCommand | 运动指令编码（仲裁 ID 结构、电机 ID 编码、数据字节布局、位置极值、力矩在仲裁 ID 中的编码） |
| MotorCommands | 使能/失能/归零/设置模式指令帧编码验证 |
| StatusFrameParsing | 状态帧解析（有效帧、拒绝非请求帧、拒绝损坏帧头/帧尾、短 DLC 拒绝、过短输入） |
| ActuatorLimits | 7 种执行器类型的范围限制验证 |
| EdgeCases | NaN/Inf/零跨距/最大电机ID/最大通道等边界情况 |
| ProtocolRoundTrip | 编码→解码往返一致性验证 |

#### test_imu_protocol.cpp（~25 个测试用例）
| 测试组 | 测试内容 |
|--------|----------|
| Crc16Modbus | 标准 CRC-16/MODBUS 校验值（含 "123456789"→0x4B37）、确定性、单位翻转检测 |
| EulerToQuaternion | 欧拉角↔四元数转换（单位四元数、纯轴旋转、往返、万向节死锁、大角度、2π周期性） |
| ImuFrameConstruction | IMU 帧构建（帧头 0xEB 0x90 0xA5 0xFF、帧尾 0x80 0x7F、数据编码、CRC 正确性） |
| ImuParser | 帧解析器（有效帧、损坏帧头/帧尾/CRC 拒绝、多帧、垃圾前缀跳过、部分帧累积、缓冲区溢出修剪） |
| ImuDataIntegrity | RPY 顺序验证、四元数一致性 |
| ImuStress | 1000 帧压力测试 |

### 依赖安装

```bash
# Ubuntu/Debian
sudo apt install libgtest-dev cmake build-essential

# 如果系统 gtest 版本过旧，CMake 会自动通过 FetchContent 下载 v1.14.0
```

### 构建和运行

```bash
# 方法一：使用便捷脚本（推荐）
cd /path/to/vbot_rl_lab
./scripts/build_and_run_tests.sh

# 方法二：手动构建
cd deploy/robots/vbot/tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 分别运行
./test_motor_protocol
./test_imu_protocol

# 或通过 CTest 运行
ctest --output-on-failure
```

### 单独编译（不依赖 CMake）

```bash
cd deploy/robots/vbot/tests
g++ -std=c++17 -Wall -o test_motor_protocol test_motor_protocol.cpp -lgtest -lgtest_main -lpthread
g++ -std=c++17 -Wall -o test_imu_protocol test_imu_protocol.cpp -lgtest -lgtest_main -lpthread
./test_motor_protocol
./test_imu_protocol
```

---

## 二、Python 硬件测试脚本

### test_motor.py — 电机测试

**功能：** 通过串口直接测试 RobStride 电机，验证 CAN 协议通信和电机控制。

**前置条件：**
- 电机通过串口转 CAN 模块连接到主机
- Python 3.6+，安装 `pyserial`：`pip install pyserial`

**使用示例：**

```bash
# 扫描所有已连接的电机
python3 scripts/test_motor.py --port /dev/ttyUSB0 --scan

# 测试单个电机（索引 0，即 motor_id=0x11）
python3 scripts/test_motor.py --port /dev/ttyUSB0 --motor 0

# 仅运行位置扫描测试
python3 scripts/test_motor.py --port /dev/ttyUSB0 --motor 0 --sweep

# 指定执行器类型和 PID 参数
python3 scripts/test_motor.py --port /dev/ttyUSB0 --actuator-type 2 --kp 30 --kd 3

# 使用第二个串口（电机 6-11）
python3 scripts/test_motor.py --port /dev/ttyUSB1 --motor 6
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyUSB0` | 串口设备路径 |
| `--baudrate` | `921600` | 波特率 |
| `--channel` | `0x00` | CAN 通道 |
| `--master-id` | `0xFF` | 主站 ID |
| `--actuator-type` | `2` | 执行器类型 (0-6)，对应 ROBSTRIDE_00 到 ROBSTRIDE_06 |
| `--motor` | 无 | 测试单个电机（索引 0-11），不指定则测试全部 |
| `--scan` | 无 | 扫描模式：仅检测已连接电机 |
| `--sweep` | 无 | 位置扫描模式：仅运行位置扫描测试 |
| `--kp` | `20.0` | 位置增益 |
| `--kd` | `2.0` | 速度增益 |

**执行器类型对照表：**

| 类型 | 名称 | 位置范围 | 速度范围 | 力矩范围 | kp 上限 | kd 上限 |
|------|------|----------|----------|----------|---------|---------|
| 0 | ROBSTRIDE_00 | ±4π | ±50 | ±17 | 500 | 5 |
| 1 | ROBSTRIDE_01 | ±4π | ±44 | ±17 | 500 | 5 |
| 2 | ROBSTRIDE_02 | ±4π | ±44 | ±17 | 500 | 5 |
| 3 | ROBSTRIDE_03 | ±4π | ±50 | ±60 | 5000 | 100 |
| 4 | ROBSTRIDE_04 | ±4π | ±15 | ±120 | 5000 | 100 |
| 5 | ROBSTRIDE_05 | ±4π | ±33 | ±17 | 500 | 5 |
| 6 | ROBSTRIDE_06 | ±4π | ±20 | ±36 | 5000 | 100 |

**电机 ID 映射：**

| 索引 | 电机 ID | 串口 | 说明 |
|------|---------|------|------|
| 0 | 0x11 | ttyUSB0 | 右前髋 |
| 1 | 0x21 | ttyUSB0 | 右前大腿 |
| 2 | 0x31 | ttyUSB0 | 右前小腿 |
| 3 | 0x12 | ttyUSB0 | 左前髋 |
| 4 | 0x22 | ttyUSB0 | 左前大腿 |
| 5 | 0x32 | ttyUSB0 | 左前小腿 |
| 6 | 0x13 | ttyUSB1 | 右后髋 |
| 7 | 0x23 | ttyUSB1 | 右后大腿 |
| 8 | 0x33 | ttyUSB1 | 右后小腿 |
| 9 | 0x14 | ttyUSB1 | 左后髋 |
| 10 | 0x24 | ttyUSB1 | 左后大腿 |
| 11 | 0x34 | ttyUSB1 | 左后小腿 |

**测试流程：**

1. **扫描测试：** 向每个电机 ID 发送使能指令，检查是否有状态帧返回
2. **使能/失能测试：** 验证电机可以正常使能和失能
3. **位置控制测试：** 使能电机，设置运动控制模式，移动到目标位置，读取反馈
4. **位置扫描测试：** 按序列 [-1.0, -0.5, 0.0, 0.5, 1.0, ...] 移动，验证每个目标位置

---

### test_imu.py — IMU 测试

**功能：** 通过串口读取 IMU 数据，验证帧协议解析和传感器数据。

**前置条件：**
- IMU 通过串口连接到主机（默认 `/dev/ttyUSB2`）
- Python 3.6+，安装 `pyserial`：`pip install pyserial`

**使用示例：**

```bash
# 协议验证测试（无需硬件连接）
python3 scripts/test_imu.py --validate-only

# 持续监测模式（Ctrl+C 退出并显示摘要）
python3 scripts/test_imu.py --port /dev/ttyUSB2

# 读取 500 帧后自动退出
python3 scripts/test_imu.py --count 500

# 将数据记录到 CSV 文件
python3 scripts/test_imu.py --output imu_data.csv

# 结合使用
python3 scripts/test_imu.py --port /dev/ttyUSB2 --count 200 --output test_imu.csv
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyUSB2` | IMU 串口设备路径 |
| `--baudrate` | `921600` | 波特率 |
| `--count` | 无 | 读取 N 帧后退出（不指定则持续运行） |
| `--output` | 无 | CSV 输出文件路径 |
| `--validate` | 无 | 运行协议验证测试 |
| `--validate-only` | 无 | 仅运行协议验证测试，跳过硬件连接 |

**IMU 帧格式（32 字节/帧）：**

| 字节位置 | 内容 | 说明 |
|----------|------|------|
| 0-3 | 0xEB 0x90 0xA5 0xFF | 帧头 |
| 4-27 | 6 个 float32 | yaw, pitch, roll, wx, wy, wz |
| 28-29 | uint16 (小端) | CRC16-Modbus（对前 28 字节计算） |
| 30-31 | 0x80 0x7F | 帧尾 |

**协议验证测试内容：**

1. **CRC16-Modbus 验证：** 用已知测试向量验证 CRC 实现正确性
2. **欧拉角↔四元数往返测试：** 10 组角度（含 0、π/2、π、万向节死锁附近），验证往返误差 < 10⁻⁵
3. **帧解析测试：**
   - 有效帧解析
   - 损坏帧头拒绝
   - 损坏帧尾拒绝
   - 损坏 CRC 拒绝
   - 多帧一次解析
   - 垃圾前缀跳过
   - 数据完整性

**监测模式输出摘要：**

退出时会显示：
- 总解析帧数
- CRC 错误次数
- 帧头/帧尾不匹配次数
- 平均帧率
- 异常告警（帧率 < 50Hz、CRC 错误率 > 5%、零帧接收）

---

## 三、便捷脚本 build_and_run_tests.sh

**用法：**

```bash
# 构建并运行 C++ 单元测试
./scripts/build_and_run_tests.sh

# 仅 C++ 测试
./scripts/build_and_run_tests.sh --cpp-only

# 运行电机硬件测试
./scripts/build_and_run_tests.sh --motor

# 运行 IMU 硬件测试
./scripts/build_and_run_tests.sh --imu

# 运行全部（C++ 测试 + 显示硬件测试命令）
./scripts/build_and_run_tests.sh --all
```

---

## 四、典型测试流程

### 初次部署测试

```bash
# 1. 先运行协议验证（无需硬件）
python3 scripts/test_imu.py --validate-only

# 2. 构建并运行 C++ 单元测试
./scripts/build_and_run_tests.sh --cpp-only

# 3. 连接硬件后，扫描电机
python3 scripts/test_motor.py --port /dev/ttyUSB0 --scan
python3 scripts/test_motor.py --port /dev/ttyUSB1 --scan

# 4. 逐个测试电机
python3 scripts/test_motor.py --port /dev/ttyUSB0 --motor 0

# 5. 测试 IMU
python3 scripts/test_imu.py --port /dev/ttyUSB2 --count 500 --output imu_test.csv
```

### 故障排查

```bash
# 电机无响应 → 检查串口和电机 ID
python3 scripts/test_motor.py --port /dev/ttyUSB0 --scan

# 电机抖动 → 降低增益
python3 scripts/test_motor.py --motor 0 --kp 10 --kd 1

# IMU 无数据 → 检查波特率和帧率
python3 scripts/test_imu.py --port /dev/ttyUSB2 --baudrate 921600

# CRC 错误率高 → 检查接线和接地
python3 scripts/test_imu.py --validate  # 先确认软件 CRC 实现正确
```

---

## 五、注意事项

1. **电机测试前确保机器人处于安全状态**（悬空或固定），防止突然运动造成损坏
2. **电机使能后请及时失能**，长时间使能不运动可能导致电机发热
3. **测试脚本中的 kp/kd 默认值较小（20/2）**，适合测试；实际运行时使用 deploy/config.yaml 中的值（40/4）
4. **IMU 帧率通常在 100-200Hz 之间**，如果帧率过低需要检查串口连接和波特率设置
5. **协议单元测试可以在任何平台运行**，不依赖机器人硬件；Python 硬件测试需要在连接了机器人的主机上运行
