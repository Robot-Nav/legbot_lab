#!/usr/bin/env python3
"""
VBot Sim2Real 部署电机测试脚本。

通过串口逐个测试 RobStride 电机：
  - 电机使能/关闭
  - 位置控制（移动到目标角度）
  - 速度反馈读取
  - 力矩反馈读取
  - CAN 帧通信验证

用法：
  python test_motor.py                          # 使用默认参数测试所有电机
  python test_motor.py --motor 0                # 测试单个电机（索引 0-11）
  python test_motor.py --port /dev/ttyUSB0      # 指定串口
  python test_motor.py --scan                   # 扫描已连接的电机
  python test_motor.py --sweep                  # 位置扫描测试
"""

import argparse
import struct
import serial
import time
import sys

# ─── 执行器类型配置（匹配 robstride_motor.h）──────────────────────────────

ACTUATOR_TYPES = {
    0: {"name": "ROBSTRIDE_00", "position": 4 * 3.14159, "velocity": 50, "torque": 17, "kp_max": 500, "kd_max": 5},
    1: {"name": "ROBSTRIDE_01", "position": 4 * 3.14159, "velocity": 44, "torque": 17, "kp_max": 500, "kd_max": 5},
    2: {"name": "ROBSTRIDE_02", "position": 4 * 3.14159, "velocity": 44, "torque": 17, "kp_max": 500, "kd_max": 5},
    3: {"name": "ROBSTRIDE_03", "position": 4 * 3.14159, "velocity": 50, "torque": 60, "kp_max": 5000, "kd_max": 100},
    4: {"name": "ROBSTRIDE_04", "position": 4 * 3.14159, "velocity": 15, "torque": 120, "kp_max": 5000, "kd_max": 100},
    5: {"name": "ROBSTRIDE_05", "position": 4 * 3.14159, "velocity": 33, "torque": 17, "kp_max": 500, "kd_max": 5},
    6: {"name": "ROBSTRIDE_06", "position": 4 * 3.14159, "velocity": 20, "torque": 36, "kp_max": 5000, "kd_max": 100},
}

# CAN 通信类型常量
COMM_MOTION_CONTROL = 0x01
COMM_MOTOR_REQUEST = 0x02
COMM_MOTOR_ENABLE = 0x03
COMM_MOTOR_STOP = 0x04
COMM_SET_POS_ZERO = 0x06
COMM_SET_PARAM = 0x12

# ─── 协议辅助函数 ──────────────────────────────────────────────────────────

def float_to_uint(x, x_min, x_max, bits=16):
    """将浮点数转换为无符号整数，用于 CAN 帧编码。"""
    x = max(x_min, min(x_max, x))
    span = x_max - x_min
    offset = x - x_min
    return int((offset * ((1 << bits) - 1)) / span)


def uint_to_float(x_int, x_min, x_max, bits=16):
    """将 CAN 帧中的无符号整数还原为浮点数。"""
    span = x_max - x_min
    return (float(x_int) * span / ((1 << bits) - 1)) + x_min


def build_can_frame(channel, arb_id, data, dlc):
    """构建 CAN-over-serial 帧（匹配 robstride_motor.h 格式）。"""
    frame = bytearray()
    frame.append(0x45)  # 帧头字节1
    frame.append(0x54)  # 帧头字节2
    frame.append(channel)
    frame.append((arb_id >> 24) & 0xFF)
    frame.append((arb_id >> 16) & 0xFF)
    frame.append((arb_id >> 8) & 0xFF)
    frame.append(arb_id & 0xFF)
    frame.append(dlc)
    frame.extend(data[:dlc])
    frame.append(0x0D)  # 帧尾字节1
    frame.append(0x0A)  # 帧尾字节2
    return bytes(frame)


def build_motion_command(channel, master_id, motor_id, actuator_type,
                         torque, position, velocity, kp, kd):
    """构建运动控制 CAN 帧。"""
    op = ACTUATOR_TYPES[actuator_type]

    torque_uint = float_to_uint(torque, -op["torque"], op["torque"], 16)
    arb_id = (COMM_MOTION_CONTROL << 24) | (torque_uint << 8) | motor_id

    pos_u = float_to_uint(position, -op["position"], op["position"], 16)
    vel_u = float_to_uint(velocity, -op["velocity"], op["velocity"], 16)
    kp_u = float_to_uint(kp, 0.0, op["kp_max"], 16)
    kd_u = float_to_uint(kd, 0.0, op["kd_max"], 16)

    data = bytearray(8)
    data[0] = (pos_u >> 8) & 0xFF
    data[1] = pos_u & 0xFF
    data[2] = (vel_u >> 8) & 0xFF
    data[3] = vel_u & 0xFF
    data[4] = (kp_u >> 8) & 0xFF
    data[5] = kp_u & 0xFF
    data[6] = (kd_u >> 8) & 0xFF
    data[7] = kd_u & 0xFF

    return build_can_frame(channel, arb_id, data, 8)


def build_enable_command(channel, master_id, motor_id):
    """构建电机使能 CAN 帧。"""
    arb_id = (COMM_MOTOR_ENABLE << 24) | (master_id << 8) | motor_id
    return build_can_frame(channel, arb_id, bytes(8), 8)


def build_disable_command(channel, master_id, motor_id, clear_error=0):
    """构建电机关闭 CAN 帧。"""
    arb_id = (COMM_MOTOR_STOP << 24) | (master_id << 8) | motor_id
    data = bytearray(8)
    data[0] = clear_error
    return build_can_frame(channel, arb_id, bytes(data), 8)


def build_set_mode_command(channel, master_id, motor_id, mode):
    """构建设置模式 CAN 帧（参数索引 0x7005）。"""
    arb_id = (COMM_SET_PARAM << 24) | (master_id << 8) | motor_id
    data = bytearray(8)
    data[0] = 0x05  # 索引低字节
    data[1] = 0x70  # 索引高字节
    data[2] = 0x00
    data[3] = 0x00
    data[4] = mode  # 'j' 模式 - 单字节值
    return build_can_frame(channel, arb_id, bytes(data), 8)


def parse_status_frame(raw_data, actuator_type):
    """解析电机状态响应帧。返回 (motor_id, position, velocity, torque) 或 None。"""
    if len(raw_data) < 10:
        return None

    # 查找帧头
    idx = 0
    while idx <= len(raw_data) - 10:
        if raw_data[idx] == 0x45 and raw_data[idx + 1] == 0x54:
            break
        idx += 1
    else:
        return None

    dlc = raw_data[idx + 7]
    frame_size = 10 + dlc
    if len(raw_data) - idx < frame_size:
        return None

    # 验证帧尾
    if raw_data[idx + frame_size - 2] != 0x0D or raw_data[idx + frame_size - 1] != 0x0A:
        return None

    can_id = ((raw_data[idx + 3] << 24) | (raw_data[idx + 4] << 16) |
              (raw_data[idx + 5] << 8) | raw_data[idx + 6]) & 0x1FFFFFFF

    comm_type = (can_id >> 24) & 0x1F
    resp_motor_id = (can_id >> 8) & 0xFF

    if comm_type != COMM_MOTOR_REQUEST or dlc < 6:
        return None

    data = raw_data[idx + 8:idx + 8 + dlc]
    pos_u16 = (data[0] << 8) | data[1]
    vel_u16 = (data[2] << 8) | data[3]
    torque_u16 = (data[4] << 8) | data[5]

    op = ACTUATOR_TYPES[actuator_type]
    position = uint_to_float(pos_u16, -op["position"], op["position"])
    velocity = uint_to_float(vel_u16, -op["velocity"], op["velocity"])
    torque = uint_to_float(torque_u16, -op["torque"], op["torque"])

    return (resp_motor_id, position, velocity, torque)


# ─── 电机测试函数 ──────────────────────────────────────────────────────────

class MotorTester:
    """VBot 电机串口测试接口。"""

    def __init__(self, port, baudrate=921600, channel=0x00, master_id=0xFF,
                 actuator_type=2, motor_ids=None):
        self.port = port
        self.baudrate = baudrate
        self.channel = channel
        self.master_id = master_id
        self.actuator_type = actuator_type
        self.motor_ids = motor_ids or [0x11, 0x21, 0x31, 0x12, 0x22, 0x32]
        self.ser = None
        self.op = ACTUATOR_TYPES[actuator_type]

    def open(self):
        """打开串口。"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
            )
            print(f"[OK] 已打开 {self.port}，波特率 {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"[FAIL] 无法打开 {self.port}: {e}")
            return False

    def close(self):
        """关闭串口。"""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _send_frame(self, frame):
        """发送原始 CAN 帧并返回响应数据。"""
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        self.ser.flush()
        time.sleep(0.005)
        return self.ser.read(256)

    def scan_motors(self):
        """扫描已连接的电机（向每个 ID 发送使能指令并检查响应）。"""
        print(f"\n{'='*60}")
        print(f"扫描电机 - 端口: {self.port}（执行器: {self.op['name']}）")
        print(f"{'='*60}")
        found = []
        for mid in self.motor_ids:
            # 尝试使能电机并检查响应
            frame = build_enable_command(self.channel, self.master_id, mid)
            resp = self._send_frame(frame)
            result = parse_status_frame(resp, self.actuator_type)
            if result is not None:
                mid_resp, pos, vel, tau = result
                print(f"  电机 0x{mid:02X}: [OK] pos={pos:.3f} vel={vel:.3f} torque={tau:.3f}")
                found.append(mid)
            else:
                print(f"  电机 0x{mid:02X}: [NO RESPONSE]")
        print(f"\n发现 {len(found)}/{len(self.motor_ids)} 个电机")
        return found

    def test_enable_disable(self, motor_id):
        """测试电机使能/关闭序列。"""
        print(f"\n--- 使能/关闭测试：电机 0x{motor_id:02X} ---")

        print("  正在使能电机...")
        frame = build_enable_command(self.channel, self.master_id, motor_id)
        resp = self._send_frame(frame)
        result = parse_status_frame(resp, self.actuator_type)
        if result:
            print(f"  [OK] 电机已使能。pos={result[1]:.3f}")
        else:
            print(f"  [FAIL] 使能指令无响应")
            return False

        time.sleep(0.2)

        print("  正在关闭电机...")
        frame = build_disable_command(self.channel, self.master_id, motor_id)
        self._send_frame(frame)
        print(f"  [OK] 电机已关闭")
        return True

    def test_position_control(self, motor_id, target_pos=0.0, kp=20.0, kd=2.0):
        """测试位置控制：使能电机，移动到目标位置，读取反馈。"""
        print(f"\n--- 位置控制测试：电机 0x{motor_id:02X} ---")

        # 使能
        frame = build_enable_command(self.channel, self.master_id, motor_id)
        self._send_frame(frame)
        time.sleep(0.05)

        # 设置为运动控制模式
        frame = build_set_mode_command(self.channel, self.master_id, motor_id, 0)
        self._send_frame(frame)
        time.sleep(0.05)

        # 移动到目标位置
        print(f"  正在移动到位置 {target_pos:.3f} rad...")
        frame = build_motion_command(
            self.channel, self.master_id, motor_id, self.actuator_type,
            torque=0.0, position=target_pos, velocity=0.0, kp=kp, kd=kd
        )
        resp = self._send_frame(frame)
        result = parse_status_frame(resp, self.actuator_type)

        if result:
            _, pos, vel, tau = result
            error = abs(pos - target_pos)
            print(f"  位置: {pos:.3f} rad（目标: {target_pos:.3f}, 误差: {error:.4f}）")
            print(f"  速度: {vel:.3f} rad/s")
            print(f"  力矩: {tau:.3f} Nm")
            if error < 0.1:
                print(f"  [OK] 位置误差在 0.1 rad 以内")
            else:
                print(f"  [WARN] 位置误差超过 0.1 rad")
            return pos, vel, tau
        else:
            print(f"  [FAIL] 无有效状态响应")
            return None

    def test_sweep(self, motor_id, kp=20.0, kd=2.0):
        """执行位置扫描测试，验证电机运动范围。"""
        print(f"\n--- 位置扫描测试：电机 0x{motor_id:02X} ---")

        # 使能并设置模式
        frame = build_enable_command(self.channel, self.master_id, motor_id)
        self._send_frame(frame)
        time.sleep(0.02)
        frame = build_set_mode_command(self.channel, self.master_id, motor_id, 0)
        self._send_frame(frame)
        time.sleep(0.02)

        # 从 -1.0 到 1.0 rad 逐步扫描
        test_positions = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, 0.0]
        results = []

        for target in test_positions:
            frame = build_motion_command(
                self.channel, self.master_id, motor_id, self.actuator_type,
                torque=0.0, position=target, velocity=0.0, kp=kp, kd=kd
            )
            resp = self._send_frame(frame)
            time.sleep(0.1)

            # 多发几条指令让电机稳定
            for _ in range(3):
                frame = build_motion_command(
                    self.channel, self.master_id, motor_id, self.actuator_type,
                    torque=0.0, position=target, velocity=0.0, kp=kp, kd=kd
                )
                resp = self._send_frame(frame)
                time.sleep(0.02)

            result = parse_status_frame(resp, self.actuator_type)
            if result:
                _, pos, vel, tau = result
                error = abs(pos - target)
                status = "OK" if error < 0.15 else "LAG"
                print(f"  目标: {target:+.3f} -> 实际: {pos:+.4f} (误差={error:.4f}) [{status}]")
                results.append((target, pos, error < 0.15))

        passed = sum(1 for _, _, ok in results if ok)
        print(f"\n  扫描结果: {passed}/{len(results)} 个位置在容差范围内")
        return results

    def test_all(self):
        """对所有已配置的电机运行完整测试套件。"""
        print(f"\n{'='*60}")
        print(f"VBot 电机测试套件")
        print(f"端口: {self.port}, 执行器: {self.op['name']}")
        print(f"{'='*60}")

        if not self.open():
            return

        try:
            for i, mid in enumerate(self.motor_ids):
                print(f"\n{'─'*60}")
                print(f"电机 {i}（ID: 0x{mid:02X}）")
                print(f"{'─'*60}")

                self.test_enable_disable(mid)
                time.sleep(0.1)
                self.test_position_control(mid, target_pos=0.0)
                time.sleep(0.1)
                self.test_sweep(mid)
                time.sleep(0.2)
        finally:
            self.close()


# ─── 主入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VBot 电机测试工具")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="串口设备（默认: /dev/ttyUSB0）")
    parser.add_argument("--baudrate", type=int, default=921600, help="波特率（默认: 921600）")
    parser.add_argument("--channel", type=int, default=0x00, help="CAN 通道（默认: 0x00）")
    parser.add_argument("--master-id", type=int, default=0xFF, help="主站 ID（默认: 0xFF）")
    parser.add_argument("--actuator-type", type=int, default=2, choices=range(7),
                        help="执行器类型 0-6（默认: 2 = ROBSTRIDE_02）")
    parser.add_argument("--motor", type=int, default=None,
                        help="测试单个电机（索引 0-11），不指定则测试全部")
    parser.add_argument("--scan", action="store_true", help="仅扫描已连接的电机")
    parser.add_argument("--sweep", action="store_true", help="仅运行位置扫描测试")
    parser.add_argument("--kp", type=float, default=20.0, help="位置增益（默认: 20）")
    parser.add_argument("--kd", type=float, default=2.0, help="速度增益（默认: 2）")
    args = parser.parse_args()

    motor_ids = [0x11, 0x21, 0x31, 0x12, 0x22, 0x32,
                 0x13, 0x23, 0x33, 0x14, 0x24, 0x34]

    tester = MotorTester(
        port=args.port,
        baudrate=args.baudrate,
        channel=args.channel,
        master_id=args.master_id,
        actuator_type=args.actuator_type,
        motor_ids=motor_ids,
    )

    if not tester.open():
        sys.exit(1)

    try:
        if args.scan:
            tester.scan_motors()
        elif args.motor is not None:
            mid = motor_ids[args.motor]
            if args.sweep:
                tester.test_sweep(mid, kp=args.kp, kd=args.kd)
            else:
                tester.test_enable_disable(mid)
                tester.test_position_control(mid, kp=args.kp, kd=args.kd)
                tester.test_sweep(mid, kp=args.kp, kd=args.kd)
        else:
            tester.test_all()
    finally:
        tester.close()


if __name__ == "__main__":
    main()
