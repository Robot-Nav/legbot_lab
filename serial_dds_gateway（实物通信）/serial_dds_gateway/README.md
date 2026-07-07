# serial_dds_gateway (C++)

C++ implementation for RS02 serial <-> DDS gateway (`灵足02.pdf` baseline).

## Components

- `include/protocol_codec.hpp` + `src/protocol_codec.cpp`
  - type1 encode / type2 decode
- `include/can_id_codec.hpp` + `src/can_id_codec.cpp`
  - legacy shifted raw-CAN helpers for older tools
- `include/serial_framer.hpp` + `src/serial_framer.cpp`
  - `45 54 ... 0D 0A` framing (Lingzu USB-CAN capture; see `lingzu_serial.hpp`)
- `include/lingzu_motor_protocol.hpp` + `src/lingzu_motor_protocol.cpp`
  - Lingzu USB-CAN frame <-> type1/type2/mode semantic conversion
- `include/imu_framer.hpp` + `src/imu_framer.cpp`
  - independent IMU serial frame parsing and yaw/pitch/roll to quaternion conversion
- `src/lingzu_frame_verify.cpp`
  - golden-frame regression test
- `src/imu_frame_verify.cpp`
  - IMU frame and quaternion regression test
- `include/motor_map.hpp`
  - 12-joint CAN-ID mapping
- `src/one_motor_demo.cpp`
  - local codec self-test
- `src/one_motor_serial.cpp`
  - single-motor serial tx/rx test
- `src/twelve_motor_serial.cpp`
  - dual-port 12-motor enable + type1 tx/rx test
- `src/dds_to_serial_gateway.cpp`
  - 12-motor DDS `rt/lowcmd` <-> serial type1/type2/type3/type4 bridge

## Build

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway
cmake -S . -B build
cmake --build build -j
```

## Serial Port Naming

Use fixed device names for the fatu hardware serial ports:

- `/dev/myttyCAN0`: motor bus A, `FR=(11,21,31)` and `RR=(13,23,33)`
- `/dev/myttyCAN1`: motor bus B, `FL=(12,22,32)` and `RL=(14,24,34)`
- `/dev/myttyIMU`: IMU serial port

These names are expected to be udev symlinks. A template is provided in
`udev/99-fatu-serial.rules.example`; copy it to `/etc/udev/rules.d/`, fill in the USB adapter attributes, then reload
udev.

## Run examples

### 0) Verify Lingzu serial frame encoding

```bash
cmake --build build -j --target lingzu_frame_verify
./build/lingzu_frame_verify
```

Golden frame (CH1, extended CAN frame, motor `0x20`/32, master `0xFD`):

`45 54 01 02 00 20 fd 08 a3 5b 7f ac 7f ff 01 22 0d 0a`

### 0.5) Verify IMU serial frame encoding

```bash
cmake --build build -j --target imu_frame_verify
./build/imu_frame_verify
```

IMU wire format:

`EB 90 A5 FF + yaw/pitch/roll/gz/gy/gx float32 little-endian + CRC16-Modbus little-endian + 80 7F`

### 0.6) Monitor a real IMU serial port

This reads IMU frames directly from a serial port. It does not require DDS or motor serial ports.

```bash
cmake --build build -j --target imu_serial_monitor

./build/imu_serial_monitor \
  --port /dev/myttyIMU \
  --baudrate 921600 \
  --degrees
```

Use `--duration 10` to stop after 10 seconds. Without `--degrees`, yaw/pitch/roll and gyro values are printed in
radians and rad/s.

### 1) Local codec self-test

```bash
./build/one_motor_demo
```

### 1.5) Web serial frame tester

```bash
cd /home/fatu06/workspace/fatuDog/serial_dds_gateway
node web/serial_frame_codec.test.mjs
python3 -m http.server 8765 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8765/serial_frame_tester.html`.

The page parses feedback frames such as:

`45 54 01 02 00 20 fd 08 a3 5b 7f ac 7f ff 01 22 0d 0a`

It also packs DDS-to-serial type1 frames such as:

`45 54 00 01 00 00 20 08 00 00 00 00 00 00 00 00 0d 0a`

It can also pack/decode type3 enable and type4 stop/clear-fault frames.

### 2) Serial single-motor test

Build the tester first:

```bash
cmake --build build -j --target one_motor_serial
```

Motor IDs are decimal. For example, `FR_hip=11` is `0x0B`, not `0x11`.

Send type3 enable, then one type1 command:

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

Send type4 disable/stop:

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

Clear fault with type4 `Byte0=1`:

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

Dual motor serial split for single-motor tests:

- Port A: `FR=(11,21,31)` and `RR=(13,23,33)`
- Port B: `FL=(12,22,32)` and `RL=(14,24,34)`

For example, disable `FL_hip=12` on port B:

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

Note: `one_motor_serial` always sends one type1 command after optional enable/disable/clear-fault frames. Use
conservative `kp/kd/tau/q` values when testing on hardware.

### 2.5) Dual-port 12-motor test

Only two motor serial ports are required (`myttyCAN0` + `myttyCAN1`); no IMU or DDS.

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

Port split matches the gateway:

- Port A: `FR=(11,21,31)` and `RR=(13,23,33)`
- Port B: `FL=(12,22,32)` and `RL=(14,24,34)`

The program prints a per-joint feedback table. Exit code `2` means fewer than 12 motors returned type2 feedback.

#### Per-motor different commands

When each joint needs different `q/dq/kp/kd/tau`, use a CSV file with `--commands-file`. CLI flags
(`--q`, `--dq`, …) remain the default for any joint not listed in the file.

CSV format (`joint,q,dq,kp,kd,tau`; `#` starts a comment line):

```csv
joint,q,dq,kp,kd,tau
FR_hip_joint,0.10,0.0,0.0,0.5,0.0
FR_thigh_joint,0.80,0.0,0.0,0.5,0.0
FR_calf_joint,-1.50,0.0,0.0,0.5,0.0
FL_hip_joint,0.10,0.0,0.0,0.5,0.0
...
```

Example file: `config/twelve_motor_commands.example.csv`. The first column may also be a decimal motor ID
(`11`, `12`, …) instead of a joint name.

Create a temporary test file where all 12 motors receive different `q` values:

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

This is a communication/encoding test. With `kp=0.0`, motors should not actively track the different `q` commands;
use a small `kp` only when the robot is safely supported.

```bash
cp config/twelve_motor_commands.example.csv /tmp/my_motor_test.csv
# edit /tmp/my_motor_test.csv

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

Before TX, the program prints `=== commanded type1 ===`; after RX, `=== feedback type2 ===`. Compare the two
tables to verify each motor received and echoed the expected command.

To probe one motor at a time with different values, use `one_motor_serial` on the correct port (`myttyCAN0` or
`myttyCAN1`) and repeat for all 12 IDs.

### 3) DDS -> serial gateway

香橙派第一阶段（单机 `lo`，先启 gateway，再启 `fatu_ctrl`）：见 [docs/PHASE1_ORANGEPI.md](../docs/PHASE1_ORANGEPI.md)

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

`dds_to_serial_gateway` uses one RX thread for motor feedback, an optional RX thread for a separate IMU serial
port, and a periodic DDS/TX loop for control output. In dual motor serial mode, port A handles FR/RR motors and port
B handles FL/RL motors. The legacy single-port mode is still available with `--serial-port /dev/myttyCAN0`.

Notes:
- `dds_to_serial_gateway` requires Unitree SDK2 + CycloneDDS libraries.
- It subscribes `rt/lowcmd`, sends type3/type4 on mode edge, sends type1 while motor mode is non-zero,
  parses type2 feedback, parses optional IMU frames, and publishes `rt/lowstate`.
- If `--imu-port` is omitted, the gateway still runs and publishes an identity IMU quaternion until real IMU data is
  wired in.
- Dual motor serial split:
  - A: `FR=(11,21,31)` and `RR=(13,23,33)`
  - B: `FL=(12,22,32)` and `RL=(14,24,34)`
- Use either legacy `--serial-port`, or both `--serial-port-a` and `--serial-port-b`; mixing them is rejected.
- Joint motor IDs are decimal:
  `FR=(11,21,31)`, `FL=(12,22,32)`, `RR=(13,23,33)`, `RL=(14,24,34)`.
- Runtime stats include `rx_frames`, `type2_frames`, `decode_errors`, `tx_type1`, `tx_enable`, `tx_disable`,
  `rx_a`, `rx_b`, `type2_a`, `type2_b`, `tx_a`, `tx_b`, `imu_frames`, and `imu_errors`.
- Serial wire: `header(45 54) + channel + frame_type + id_field(2) + can_or_master_id + dlc + data + 0D 0A`.
- Host -> serial type1 uses standard frame: `45 54 00 01 TT TT 20 08 ... 0D 0A`, where `TT TT`
  is torque over -17..17 Nm and `20` is CAN ID `0x20`.
- Motor feedback uses extended frame: `45 54 01 02 00 20 FD 08 ... 0D 0A`, where data bytes are
  q, dq, tau, temperature (two bytes each, big-endian).
- Type3 enable frame: `45 54 CC 03 00 FD MM 08 00 00 00 00 00 00 00 00 0D 0A`, where
  `CC` is channel and `MM` is target motor CAN ID.
- Type4 stop frame: `45 54 CC 04 00 FD MM 08 00 00 00 00 00 00 00 00 0D 0A`; set
  `Byte0=01` in the data area to clear faults.
- IMU frames decode yaw/pitch/roll as radians and publish DDS quaternion in `w,x,y,z` order using ZYX
  (`yaw around Z`, `pitch around Y`, `roll around X`). Serial gyro order is `gz,gy,gx`; DDS `imu_state.gyroscope` is
  published as body-frame `gx,gy,gz`.
- **Scheme A (sim2real):** gateway applies stationary gyro bias calibration (default 2 s, keep robot still) and
  deadzone (`--imu-gyro-deadzone`, default `0.03` rad/s). `fatu_ctrl` does not subtract bias again on Velocity enter.
