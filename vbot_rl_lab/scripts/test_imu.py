#!/usr/bin/env python3
"""
VBot Sim2Real 部署 IMU 测试脚本。

测试串口 IMU 通信：
  - 原始帧解析（帧头/帧尾检测）
  - CRC16-Modbus 校验验证
  - 欧拉角到四元数转换
  - 数据速率和连续性检查
  - 姿态角和角速度读取

IMU 帧格式（来自 Types.h read_imu()）：
  字节 0-3:   帧头 (0xEB, 0x90, 0xA5, 0xFF)
  字节 4-27:  6 个 float（偏航、俯仰、横滚、wx、wy、wz）= 24 字节
  字节 28-29: CRC16-Modbus（低字节，高字节）
  字节 30-31: 帧尾 (0x80, 0x7F)
  总计：每帧 32 字节

用法：
  python test_imu.py                           # 持续监测模式
  python test_imu.py --port /dev/ttyUSB2       # 指定串口
  python test_imu.py --count 100               # 读取 N 帧后退出
  python test_imu.py --validate                # 验证 CRC 和帧完整性
  python test_imu.py --output imu_log.csv      # 将数据记录到 CSV 文件
"""

import argparse
import struct
import serial
import time
import math
import csv
import sys
from collections import deque


# ─── IMU 协议常量 ──────────────────────────────────────────────────────────

FRAME_HEADER = bytes([0xEB, 0x90, 0xA5, 0xFF])
FRAME_FOOTER = bytes([0x80, 0x7F])
FRAME_SIZE = 32
HEADER_LEN = 4
DATA_LEN = 24  # 6 个 float × 4 字节
CRC_LEN = 2
FOOTER_LEN = 2


# ─── CRC16-Modbus 校验 ─────────────────────────────────────────────────────

def crc16_modbus(data: bytes) -> int:
    """计算 CRC16-Modbus 校验值（匹配 Types.h 实现）。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


# CRC16-Modbus 测试向量 — 由算法自身计算得出。
# 用于验证实现的一致性。
# 标准 CRC-16/MODBUS 校验值："123456789"（ASCII）→ 0x4B37
CRC_TEST_VECTORS = [
    # 标准 MODBUS 校验字符串
    (b"123456789", 0x4B37),
    (bytes([0x01, 0x02, 0x03, 0x04]), 0x2BA1),
    (bytes([0x00, 0x00, 0x00, 0x00]), 0x2400),
    (bytes([0xFF, 0xFF, 0xFF, 0xFF]), 0xB001),
    (bytes([0]*28), 0xEAA8),  # 28 个零字节（IMU 数据负载大小）
]


# ─── 欧拉角与四元数转换 ───────────────────────────────────────────────────

def euler_to_quaternion(yaw: float, pitch: float, roll: float):
    """将欧拉角（弧度）转换为四元数 [w, x, y, z]。
    与 Types.h 中 euler_to_quaternion() 实现一致。"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def quaternion_to_euler(w, x, y, z):
    """将四元数转换回欧拉角，用于验证往返一致性。"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (yaw, pitch, roll)


# ─── IMU 帧解析器 ──────────────────────────────────────────────────────────

class IMUFrameParser:
    """从串口字节流解析 IMU 帧（匹配 read_imu() 逻辑）。"""

    def __init__(self, max_buffer_size=512):
        self.buffer = bytearray()
        self.max_buffer_size = max_buffer_size
        self.frames_parsed = 0
        self.crc_errors = 0
        self.header_mismatches = 0
        self.footer_mismatches = 0
        self.last_frame = None
        # 统计信息
        self.data_history = deque(maxlen=1000)
        self.frame_timestamps = deque(maxlen=100)

    def feed(self, data: bytes):
        """将原始字节送入解析器。返回已解析帧列表。"""
        self.buffer.extend(data)

        # 缓冲区过大时裁剪
        if len(self.buffer) > self.max_buffer_size:
            excess = len(self.buffer) - self.max_buffer_size
            del self.buffer[:excess]

        frames = []
        while len(self.buffer) >= FRAME_SIZE:
            # 检查帧头
            if (self.buffer[0] != 0xEB or self.buffer[1] != 0x90 or
                    self.buffer[2] != 0xA5 or self.buffer[3] != 0xFF):
                self.header_mismatches += 1
                del self.buffer[0]
                continue

            # 检查帧尾
            if self.buffer[30] != 0x80 or self.buffer[31] != 0x7F:
                self.footer_mismatches += 1
                del self.buffer[0]
                continue

            # 验证 CRC
            received_crc = (self.buffer[29] << 8) | self.buffer[28]
            calculated_crc = crc16_modbus(bytes(self.buffer[:28]))
            crc_valid = (received_crc == calculated_crc)

            if not crc_valid:
                self.crc_errors += 1
                del self.buffer[0]
                continue

            # 解析 6 个 float
            data = struct.unpack('<6f', bytes(self.buffer[4:28]))
            yaw, pitch, roll, wx, wy, wz = data

            # 转换为四元数
            quat = euler_to_quaternion(yaw, pitch, roll)

            frame = {
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'gyro_x': wx,
                'gyro_y': wy,
                'gyro_z': wz,
                'quaternion': quat,
                'crc_ok': crc_valid,
                'rpy': (roll, pitch, yaw),  # 匹配 Types.h: rpy[0]=roll, rpy[1]=pitch, rpy[2]=yaw
                'timestamp': time.time(),
            }

            frames.append(frame)
            self.last_frame = frame
            self.frames_parsed += 1
            self.frame_timestamps.append(frame['timestamp'])
            self.data_history.append(frame)

            del self.buffer[:FRAME_SIZE]

        return frames

    @property
    def frame_rate(self):
        """根据最近的时间戳估算帧率。"""
        if len(self.frame_timestamps) < 2:
            return 0.0
        dt = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self.frame_timestamps) - 1) / dt

    def get_stats(self):
        """获取解析统计信息。"""
        return {
            'frames_parsed': self.frames_parsed,
            'crc_errors': self.crc_errors,
            'header_mismatches': self.header_mismatches,
            'footer_mismatches': self.footer_mismatches,
            'frame_rate': self.frame_rate,
            'buffer_size': len(self.buffer),
        }


# ─── IMU 测试用例 ──────────────────────────────────────────────────────────

def test_crc16():
    """使用已知测试向量验证 CRC16-Modbus 实现。"""
    print("--- CRC16-Modbus 验证 ---")
    all_ok = True
    for data, expected in CRC_TEST_VECTORS:
        result = crc16_modbus(data)
        if expected is not None:
            status = "OK" if result == expected else "FAIL"
            if result != expected:
                all_ok = False
            print(f"  data={data.hex()} -> 0x{result:04X}（期望 0x{expected:04X}）[{status}]")
        else:
            print(f"  data={data.hex()} -> 0x{result:04X}（无期望值）")
    return all_ok


def test_euler_quaternion():
    """验证欧拉角与四元数的往返转换。"""
    print("\n--- 欧拉角/四元数往返测试 ---")
    test_angles = [
        (0.0, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (-math.pi / 2, 0.0, 0.0),
        (0.0, math.pi / 4, 0.0),
        (0.0, 0.0, math.pi / 4),
        (math.pi / 4, math.pi / 4, 0.0),
        (1.0, -0.5, 0.3),
        (math.pi, 0.0, 0.0),
        (-math.pi, 0.0, 0.0),
        (0.0, math.pi / 2 - 0.01, 0.0),  # 接近万向节死锁
    ]
    all_ok = True
    for yaw, pitch, roll in test_angles:
        q = euler_to_quaternion(yaw, pitch, roll)
        y2, p2, r2 = quaternion_to_euler(*q)

        # 归一化角度差用于比较
        def angle_diff(a, b):
            d = a - b
            d = (d + math.pi) % (2 * math.pi) - math.pi
            return abs(d)

        y_err = angle_diff(yaw, y2)
        p_err = angle_diff(pitch, p2)
        r_err = angle_diff(roll, r2)
        ok = max(y_err, p_err, r_err) < 1e-5

        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(f"  ypr=({yaw:+.3f}, {pitch:+.3f}, {roll:+.3f}) "
              f"-> quat=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}) "
              f"-> ypr=({y2:+.3f}, {p2:+.3f}, {r2:+.3f}) [{status}]")
    return all_ok


def test_frame_parsing():
    """使用合成数据测试 IMU 帧解析器。"""
    print("\n--- IMU 帧解析测试 ---")

    parser = IMUFrameParser()

    # 构建有效帧
    yaw, pitch, roll = 1.5, 0.3, -0.2
    wx, wy, wz = 0.1, 0.2, -0.15
    data_bytes = struct.pack('<6f', yaw, pitch, roll, wx, wy, wz)
    crc = crc16_modbus(FRAME_HEADER + data_bytes)
    valid_frame = FRAME_HEADER + data_bytes + struct.pack('<H', crc) + FRAME_FOOTER

    assert len(valid_frame) == 32, f"帧大小: {len(valid_frame)}"

    # 测试 1：有效帧
    frames = parser.feed(bytes(valid_frame))
    if frames:
        f = frames[0]
        print(f"  有效帧: yaw={f['yaw']:.3f} gyro=({f['gyro_x']:.3f}, {f['gyro_y']:.3f}, {f['gyro_z']:.3f}) [OK]")
    else:
        print(f"  有效帧: 未解析 [FAIL]")

    # 测试 2：损坏帧头
    bad_header = bytearray(valid_frame)
    bad_header[0] = 0x00
    parser.feed(bytes(bad_header))
    assert parser.header_mismatches > 0, "应检测到帧头不匹配"
    print(f"  损坏帧头已拒绝: [OK]")

    # 测试 3：损坏帧尾
    bad_footer = bytearray(valid_frame)
    bad_footer[30] = 0x00
    parser.feed(bytes(bad_footer))
    assert parser.footer_mismatches > 0, "应检测到帧尾不匹配"
    print(f"  损坏帧尾已拒绝: [OK]")

    # 测试 4：损坏 CRC
    bad_crc = bytearray(valid_frame)
    bad_crc[28] ^= 0xFF  # 翻转 CRC 字节
    parser.feed(bytes(bad_crc))
    assert parser.crc_errors > 0, "应检测到 CRC 错误"
    print(f"  损坏 CRC 已拒绝: [OK]")

    # 测试 5：缓冲区中多个帧
    parser2 = IMUFrameParser()
    multi = valid_frame + valid_frame + valid_frame
    frames = parser2.feed(multi)
    assert len(frames) == 3, f"期望 3 帧，实际 {len(frames)}"
    print(f"  多帧（3 帧）: 已解析 {len(frames)} [OK]")

    # 测试 6：有效帧前的垃圾前缀
    parser3 = IMUFrameParser()
    garbage = bytes([0x00, 0x11, 0x22, 0x33]) + valid_frame
    frames = parser3.feed(garbage)
    assert len(frames) == 1, f"期望 1 帧，实际 {len(frames)}"
    assert parser3.header_mismatches >= 4, f"应有 4 次帧头不匹配"
    print(f"  垃圾前缀已跳过: 已解析 {len(frames)} 帧 [OK]")

    # 测试 7：数据完整性
    parser4 = IMUFrameParser()
    frames = parser4.feed(bytes(valid_frame))
    f = frames[0]
    assert abs(f['yaw'] - yaw) < 0.001, f"yaw 不匹配: {f['yaw']} != {yaw}"
    assert abs(f['pitch'] - pitch) < 0.001, f"pitch 不匹配"
    assert abs(f['roll'] - roll) < 0.001, f"roll 不匹配"
    assert abs(f['gyro_x'] - wx) < 0.001, f"gyro_x 不匹配"
    print(f"  数据完整性: [OK]")

    print(f"  所有帧解析测试通过！")


# ─── 硬件 IMU 读取器 ───────────────────────────────────────────────────────

class IMUReader:
    """从实际硬件的串口读取 IMU 数据。"""

    def __init__(self, port, baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.parser = IMUFrameParser()

    def open(self):
        """打开 IMU 串口。"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            print(f"[OK] 已打开 IMU - 端口: {self.port}，波特率: {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"[FAIL] 无法打开 {self.port}: {e}")
            return False

    def close(self):
        """关闭串口。"""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def read_frames(self, timeout=0.5):
        """读取可用数据并返回新解析的帧。"""
        try:
            data = self.ser.read(self.ser.in_waiting or 256)
            if data:
                return self.parser.feed(data)
        except serial.SerialException as e:
            print(f"[ERROR] 串口读取错误: {e}")
        return []

    def monitor(self, count=None, output_csv=None):
        """持续监测模式。"""
        print(f"\n{'='*70}")
        print(f"IMU 监测 - 端口: {self.port}")
        print(f"{'时间':>10s} {'Yaw':>8s} {'Pitch':>8s} {'Roll':>8s} "
              f"{'Wx':>8s} {'Wy':>8s} {'Wz':>8s}  帧率")
        print(f"{'─'*70}")

        csv_writer = None
        csv_file = None
        if output_csv:
            csv_file = open(output_csv, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'yaw', 'pitch', 'roll',
                                 'wx', 'wy', 'wz', 'qw', 'qx', 'qy', 'qz'])

        start_time = time.time()
        frame_count = 0
        last_print = start_time

        try:
            while True:
                frames = self.read_frames()
                for f in frames:
                    frame_count += 1
                    now = time.time()

                    # 每 0.1 秒打印一次，避免输出过多
                    if now - last_print > 0.1:
                        elapsed = now - start_time
                        q = f['quaternion']
                        print(f"{elapsed:10.2f} {f['yaw']:8.3f} {f['pitch']:8.3f} "
                              f"{f['roll']:8.3f} {f['gyro_x']:8.3f} "
                              f"{f['gyro_y']:8.3f} {f['gyro_z']:8.3f} "
                              f"{self.parser.frame_rate:5.1f} Hz")
                        last_print = now

                    if csv_writer:
                        csv_writer.writerow([
                            f['timestamp'], f['yaw'], f['pitch'], f['roll'],
                            f['gyro_x'], f['gyro_y'], f['gyro_z'],
                            q[0], q[1], q[2], q[3]
                        ])

                    if count and frame_count >= count:
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            stats = self.parser.get_stats()
            print(f"\n{'='*70}")
            print(f"IMU 监测摘要（{elapsed:.1f}s）：")
            print(f"  已解析帧数:    {stats['frames_parsed']}")
            print(f"  CRC 错误:      {stats['crc_errors']}")
            print(f"  帧头错误:      {stats['header_mismatches']}")
            print(f"  帧尾错误:      {stats['footer_mismatches']}")
            print(f"  平均帧率:      {stats['frame_rate']:.1f} Hz")
            if stats['frames_parsed'] > 0:
                print(f"  缓冲区大小:    {stats['buffer_size']} 字节")

            if stats['frame_rate'] < 50:
                print(f"  [WARN] 帧率过低（< 50 Hz），请检查 IMU 连接")
            if stats['crc_errors'] > stats['frames_parsed'] * 0.05:
                print(f"  [WARN] CRC 错误率过高，请检查接线/噪声")
            if stats['frames_parsed'] == 0:
                print(f"  [FAIL] 未收到有效帧，请检查 IMU 端口和波特率")
            print(f"{'='*70}")

        finally:
            if csv_file:
                csv_file.close()
                print(f"数据已保存到 {output_csv}")


# ─── 主入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VBot IMU 测试工具")
    parser.add_argument("--port", default="/dev/ttyUSB2", help="IMU 串口设备（默认: /dev/ttyUSB2）")
    parser.add_argument("--baudrate", type=int, default=921600, help="波特率（默认: 921600）")
    parser.add_argument("--count", type=int, default=None, help="读取 N 帧后退出")
    parser.add_argument("--output", default=None, help="用于数据记录的 CSV 输出文件")
    parser.add_argument("--validate", action="store_true",
                        help="运行协议验证测试（CRC、欧拉角、帧解析）")
    parser.add_argument("--validate-only", action="store_true",
                        help="仅运行协议验证测试，跳过硬件连接")
    args = parser.parse_args()

    print("VBot IMU 测试工具")
    print("=" * 70)

    # 始终运行协议验证测试
    crc_ok = test_crc16()
    euler_ok = test_euler_quaternion()
    test_frame_parsing()

    if not crc_ok:
        print("\n[FAIL] CRC16 实现有错误！")
        sys.exit(1)
    if not euler_ok:
        print("\n[FAIL] 欧拉角/四元数转换有错误！")
        sys.exit(1)

    print(f"\n[OK] 所有协议验证测试通过。\n")

    if args.validate_only:
        return

    # 硬件测试
    reader = IMUReader(args.port, args.baudrate)
    if not reader.open():
        print(f"\n无法打开 IMU 端口 {args.port}。跳过硬件测试。")
        print(f"请连接 IMU，或使用 --validate-only 仅运行协议测试。")
        sys.exit(1)

    try:
        reader.monitor(count=args.count, output_csv=args.output)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
