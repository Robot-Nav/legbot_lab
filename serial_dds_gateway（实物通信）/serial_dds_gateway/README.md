# serial_dds_gateway（C++）

C++ 实现的 RS02 串口 <-> DDS 网关（以 `灵足02.pdf` 为协议基线）。

## 组件

- `include/protocol_codec.hpp` + `src/protocol_codec.cpp`
  - type1 编码 / type2 解码
- `include/can_id_codec.hpp` + `src/can_id_codec.cpp`
  - 兼容旧工具的 29 位 CAN ID 移位辅助函数
- `include/serial_framer.hpp` + `src/serial_framer.cpp`
  - `45 54 ... 0D 0A` 串口帧封装与解析（灵足 USB-CAN 格式，见 `lingzu_serial.hpp`）
- `include/lingzu_motor_protocol.hpp` + `src/lingzu_motor_protocol.cpp`
  - 灵足 USB-CAN 帧 <-> type1/type2/模式帧语义转换
- `include/imu_framer.hpp` + `src/imu_framer.cpp`
  - 独立 IMU 串口帧解析与 yaw/pitch/roll 转四元数
- `src/lingzu_frame_verify.cpp`
  - golden-frame 回归测试
- `src/imu_frame_verify.cpp`
  - IMU 帧与四元数回归测试
- `include/motor_map.hpp`
  - 12 路关节 CAN ID 映射
- `src/one_motor_demo.cpp`
  - 本地编解码自测
- `src/one_motor_serial.cpp`
  - 单电机串口收发测试
- `src/twelve_motor_serial.cpp`
  - 双串口 12 电机使能 + type1 收发测试
- `src/dds_to_serial_gateway.cpp`
  - 12 电机 DDS `rt/lowcmd` <-> 串口 type1/type2/type3/type4 网关

## 构建

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway
cmake -S . -B build
cmake --build build -j
```

## 串口命名

Fatu 硬件使用固定串口设备名：

- `/dev/myttyCAN0`：电机总线 A，`FR=(11,21,31)` 与 `RR=(13,23,33)`
- `/dev/myttyCAN1`：电机总线 B，`FL=(12,22,32)` 与 `RL=(14,24,34)`
- `/dev/myttyIMU`：IMU 串口

这些名称应为 udev 符号链接。模板见 `udev/99-fatu-serial.rules.example`；复制到 `/etc/udev/rules.d/`，填入真实适配器序列号后重载 udev。

## 运行示例

### 0) 验证灵足串口帧编码

```bash
cmake --build build -j --target lingzu_frame_verify
./build/lingzu_frame_verify
```

Golden frame（CH1，扩展帧，电机 `0x20`/32，master `0xFD`）：

`45 54 01 02 00 20 fd 08 a3 5b 7f ac 7f ff 01 22 0d 0a`

### 0.5) 验证 IMU 串口帧编码

```bash
cmake --build build -j --target imu_frame_verify
./build/imu_frame_verify
```

IMU 线格式：

`EB 90 A5 FF + yaw/pitch/roll/gz/gy/gx float32 little-endian + CRC16-Modbus little-endian + 80 7F`

### 0.6) 监听真实 IMU 串口

直接读取 IMU 串口，无需 DDS 或电机串口。

```bash
cmake --build build -j --target imu_serial_monitor

./build/imu_serial_monitor \
  --port /dev/myttyIMU \
  --baudrate 921600 \
  --degrees
```

使用 `--duration 10` 可在 10 秒后自动停止；不加 `--degrees` 时，yaw/pitch/roll 与陀螺以弧度、rad/s 输出。

### 1) 本地编解码自测

```bash
./build/one_motor_demo
```

### 1.5) Web 串口帧测试页

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway
node web/serial_frame_codec.test.mjs
python3 -m http.server 8765 --bind 127.0.0.1 --directory web
```

浏览器打开 `http://127.0.0.1:8765/serial_frame_tester.html`。

该页面可解析如下反馈帧：

`45 54 01 02 00 20 fd 08 a3 5b 7f ac 7f ff 01 22 0d 0a`

也可打包 DDS->串口 type1 帧，例如：

`45 54 00 01 00 00 20 08 00 00 00 00 00 00 00 00 0d 0a`

同时支持 type3 使能、type4 停止/清故障帧的打包与解析。

### 2) 串口单电机测试

先构建测试程序：

```bash
cmake --build build -j --target one_motor_serial
```

电机 ID 为十进制。例如 `FR_hip=11` 对应 `0x0B`，不是 `0x11`。

发送 type3 使能后下发一帧 type1：

```bash
./build/one_motor_serial \
  --port /dev/myttyCAN0 \
  --baudrate 2000000 \
  --motor-id 11 \
  --master-id 0x00FD \
  --channel 0x00 \
  --send-enable \
  --q 1.0 --dq 0.0 --kp 30 --kd 1.0 --tau 0.0 --rx-seconds 2.0
```

发送 type4 停止：

```bash
./build/one_motor_serial \
  --port /dev/myttyCAN0 \
  --baudrate 2000000 \
  --motor-id 11 \
  --master-id 0x00FD \
  --channel 0x00 \
  --send-disable \
  --rx-seconds 1.0
```

使用 type4 `Byte0=1` 清故障：

```bash
./build/one_motor_serial \
  --port /dev/myttyCAN0 \
  --baudrate 2000000 \
  --motor-id 11 \
  --master-id 0x00FD \
  --channel 0x00 \
  --clear-fault \
  --rx-seconds 1.0
```

单电机测试的双串口划分：

- 总线 A：`FR=(11,21,31)` 与 `RR=(13,23,33)`
- 总线 B：`FL=(12,22,32)` 与 `RL=(14,24,34)`

例如，在总线 B 停止 `FL_hip=12`：

```bash
./build/one_motor_serial \
  --port /dev/myttyCAN1 \
  --baudrate 2000000 \
  --motor-id 12 \
  --master-id 0x00FD \
  --channel 0x00 \
  --send-disable \
  --rx-seconds 1.0
```

注意：`one_motor_serial` 在可选使能/停止/清故障帧之后总会发送一帧 type1。在硬件上测试时请使用保守的 `kp/kd/tau/q`。

### 2.5) 双串口 12 电机测试

仅需两个电机串口（`myttyCAN0` + `myttyCAN1`），无需 IMU 或 DDS。

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway

cmake --build build -j --target twelve_motor_serial

./build/twelve_motor_serial \
  --port-a /dev/myttyCAN0 \
  --port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --master-id 0x00FD \
  --channel 0x00 \
  --send-enable \
  --disable-on-exit \
  --q 0.0 --dq 0.0 --kp 0.0 --kd 0.5 --tau 0.0 \
  --tx-hz 50 \
  --rx-seconds 3.0
```

串口划分与网关一致：

- 总线 A：`FR=(11,21,31)` 与 `RR=(13,23,33)`
- 总线 B：`FL=(12,22,32)` 与 `RL=(14,24,34)`

程序会打印每关节反馈表。退出码 `2` 表示少于 12 路电机返回 type2 反馈。

#### 每电机不同指令

当各关节需要不同 `q/dq/kp/kd/tau` 时，使用 `--commands-file` 指定 CSV 文件。CLI 参数（`--q`、`--dq` 等）作为未列出关节的默认值。

CSV 格式（`joint,q,dq,kp,kd,tau`；`#` 开头为注释）：

```csv
joint,q,dq,kp,kd,tau
FR_hip_joint,0.10,0.0,0.0,0.5,0.0
FR_thigh_joint,0.80,0.0,0.0,0.5,0.0
FR_calf_joint,-1.50,0.0,0.0,0.5,0.0
FL_hip_joint,0.10,0.0,0.0,0.5,0.0
...
```

示例文件：`config/twelve_motor_commands.example.csv`。第一列也可是十进制电机 ID（`11`、`12` 等）。

临时创建 12 路电机不同 `q` 的测试文件：

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway

cat > /tmp/twelve_motor_diff.csv <<'EOF'
joint,q,dq,kp,kd,tau
FR_hip_joint,0.01,0.0,0.0,0.5,0.0
FR_thigh_joint,0.02,0.0,0.0,0.5,0.0
FR_calf_joint,0.03,0.0,0.0,0.5,0.0
FL_hip_joint,0.04,0.0,0.0,0.5,0.0
FL_thigh_joint,0.05,0.0,0.0,0.5,0.0
FL_calf_joint,0.06,0.0,0.0,0.5,0.0
RR_hip_joint,0.07,0.0,0.0,0.5,0.0
RR_thigh_joint,0.08,0.0,0.0,0.5,0.0
RR_calf_joint,0.09,0.0,0.0,0.5,0.0
RL_hip_joint,0.10,0.0,0.0,0.5,0.0
RL_thigh_joint,0.11,0.0,0.0,0.5,0.0
RL_calf_joint,0.12,0.0,0.0,0.5,0.0
EOF

cmake --build build -j --target twelve_motor_serial

./build/twelve_motor_serial \
  --port-a /dev/myttyCAN0 \
  --port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --master-id 0x00FD \
  --channel 0x00 \
  --commands-file /tmp/twelve_motor_diff.csv \
  --send-enable \
  --disable-on-exit \
  --tx-hz 50 \
  --rx-seconds 3.0
```

这是通信/编码测试。`kp=0.0` 时电机不会主动跟踪不同 `q`；仅在机器人安全支撑时才可启用较小 `kp`。

```bash
cp config/twelve_motor_commands.example.csv /tmp/my_motor_test.csv
# 编辑 /tmp/my_motor_test.csv

./build/twelve_motor_serial \
  --port-a /dev/myttyCAN0 \
  --port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --master-id 0x00FD \
  --channel 0x00 \
  --commands-file /tmp/my_motor_test.csv \
  --send-enable \
  --disable-on-exit \
  --tx-hz 50 \
  --rx-seconds 3.0
```

发送前程序打印 `=== commanded type1 ===`；接收后打印 `=== feedback type2 ===`。对比两表以验证每路电机收到并回显了预期指令。

如需逐电机探测不同值，请在正确总线（`myttyCAN0` 或 `myttyCAN1`）上对所有 12 个 ID 重复使用 `one_motor_serial`。

### 3) DDS -> 串口网关

香橙派第一阶段（单机 `lo`，先启 gateway，再启 `fatu_ctrl`）：详见 [docs/PHASE1_ORANGEPI.md](../docs/PHASE1_ORANGEPI.md)

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
  --send-disable-on-exit
```

终端 2：

```bash
cd /home/fatu06/workspace/fatuDog/unitree_rl_lab/deploy/build
./fatu_ctrl --network lo
```

启动后会打印 `[PHASE1]` 日志。不要加 `--send-enable-on-start`（由 FSM 使能电机）；默认不加 `--wait-lowcmd`。

`dds_to_serial_gateway` 使用一个电机反馈 RX 线程、一个可选 IMU 串口 RX 线程，以及一个周期性 DDS/TX 控制输出循环。双电机串口模式下，总线 A 处理 FR/RR 电机，总线 B 处理 FL/RL 电机。遗留单串口模式仍可用 `--serial-port /dev/myttyCAN0`。

说明：

- `dds_to_serial_gateway` 依赖 Unitree SDK2 + CycloneDDS 库。
- 订阅 `rt/lowcmd`，在 mode 边沿发送 type3/type4，mode 非零期间每 tick 发送 type1，解析 type2 反馈，解析可选 IMU 帧，发布 `rt/lowstate`。
- 若省略 `--imu-port`，网关仍运行并以单位四元数发布 IMU，直到接入真实 IMU。
- 双电机串口划分：
  - A：`FR=(11,21,31)` 与 `RR=(13,23,33)`
  - B：`FL=(12,22,32)` 与 `RL=(14,24,34)`
- 使用 legacy `--serial-port`，或同时使用 `--serial-port-a` 与 `--serial-port-b`；混用会被拒绝。
- 关节电机 ID 为十进制：`FR=(11,21,31)`、`FL=(12,22,32)`、`RR=(13,23,33)`、`RL=(14,24,34)`。
- 运行统计包括 `rx_frames`、`type2_frames`、`decode_errors`、`tx_type1`、`tx_enable`、`tx_disable`、`rx_a`、`rx_b`、`type2_a`、`type2_b`、`tx_a`、`tx_b`、`imu_frames`、`imu_errors`。
- 串口线格式：`header(45 54) + channel + frame_type + id_field(2) + can_or_master_id + dlc + data + 0D 0A`。
- 主机 -> 串口 type1 使用标准帧：`45 54 00 01 TT TT 20 08 ... 0D 0A`，其中 `TT TT` 为 -17..17 Nm 扭矩量化值，`20` 为 CAN ID `0x20`。
- 电机反馈使用扩展帧：`45 54 01 02 00 20 FD 08 ... 0D 0A`，数据区依次为 q、dq、tau、temperature（每项 2 字节大端）。
- Type3 使能帧：`45 54 CC 03 00 FD MM 08 00 00 00 00 00 00 00 00 0D 0A`，其中 `CC` 为 channel，`MM` 为目标电机 CAN ID。
- Type4 停止帧：`45 54 CC 04 00 FD MM 08 00 00 00 00 00 00 00 00 0D 0A`；数据区 `Byte0=01` 清故障。
- IMU 帧将 yaw/pitch/roll 以弧度解码，并以 ZYX 顺序（`绕 Z 偏航`、`绕 Y 俯仰`、`绕 X 横滚`）发布 DDS 四元数 `w,x,y,z`。串口陀螺顺序为 `gz,gy,gx`；DDS `imu_state.gyroscope` 以机体坐标系 `gx,gy,gz` 发布。
- **方案 A（sim2real）：** 网关在静止状态下在线标定陀螺偏置（默认 2 秒，保持机器人静止）并应用死区（`--imu-gyro-deadzone`，默认 `0.03` rad/s）。`fatu_ctrl` 进入 Velocity 时不再重复减偏置。
