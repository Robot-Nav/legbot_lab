#!/usr/bin/env python3
"""Damping-only health check with no third-party Python dependencies."""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time


MOTOR_COUNT = 16
WHEEL_INDICES = (3, 7, 11, 15)
MOTOR_LABELS = (
    "front_left_hip", "front_left_thigh", "front_left_calf", "front_left_wheel",
    "front_right_hip", "front_right_thigh", "front_right_calf", "front_right_wheel",
    "rear_left_hip", "rear_left_thigh", "rear_left_calf", "rear_left_wheel",
    "rear_right_hip", "rear_right_thigh", "rear_right_calf", "rear_right_wheel",
)
COMMAND_MAGIC = 0x5242434D
FEEDBACK_MAGIC = 0x52424642
IMU_COMMAND_MAGIC = 0x494D5543
IMU_FEEDBACK_MAGIC = 0x494D5546
VERSION = 1
MOTOR_COMMAND_HEADER = struct.Struct("<IHHII")
MOTOR_COMMAND = struct.Struct("<fffff")
MOTOR_FEEDBACK_HEADER = struct.Struct("<IHHIIIffff")
MOTOR_FEEDBACK = struct.Struct("<ffffBBH")
IMU_COMMAND = struct.Struct("<IHHII")
IMU_FEEDBACK = struct.Struct("<IHHIIII3f3f3ffff")

STATUS_SAFETY_LATCHED = 1 << 17
STATUS_WAITING_READY = 1 << 18
STATUS_READY = 1 << 19
STATUS_RUNNING = 1 << 20
STATUS_LATCH_TIMEOUT = 1 << 21
STATUS_LATCH_MOTOR_OFFLINE = 1 << 22


def damping_packet(sequence: int) -> bytes:
    packet = bytearray(MOTOR_COMMAND_HEADER.pack(
        COMMAND_MAGIC, VERSION, MOTOR_COUNT, sequence & 0xFFFFFFFF, 0
    ))
    for index in range(MOTOR_COUNT):
        kd = 1.0 if index in WHEEL_INDICES else 2.0
        packet.extend(MOTOR_COMMAND.pack(0.0, 0.0, 0.0, 0.0, kd))
    return bytes(packet)


def receive_latest(sock: socket.socket) -> bytes | None:
    latest = None
    while True:
        try:
            latest, _ = sock.recvfrom(2048)
        except BlockingIOError:
            return latest


def state_name(status: int) -> str:
    if status & STATUS_SAFETY_LATCHED:
        return "LATCHED"
    if status & STATUS_RUNNING:
        return "RUNNING"
    if status & STATUS_READY:
        return "READY"
    if status & STATUS_WAITING_READY:
        return "WAITING"
    return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--motor-port", type=int, default=55100)
    parser.add_argument("--imu-port", type=int, default=55200)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--require-imu", action="store_true")
    args = parser.parse_args()

    motor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    motor_sock.setblocking(False)
    imu_sock.setblocking(False)
    motor_data = None
    imu_data = None
    sequence = 0
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            motor_sock.sendto(damping_packet(sequence), (args.host, args.motor_port))
            imu_sock.sendto(
                IMU_COMMAND.pack(IMU_COMMAND_MAGIC, VERSION, 0, sequence, 0),
                (args.host, args.imu_port),
            )
            sequence = (sequence + 1) & 0xFFFFFFFF
            time.sleep(0.01)
            motor_data = receive_latest(motor_sock) or motor_data
            imu_data = receive_latest(imu_sock) or imu_data
            if motor_data is not None and (imu_data is not None or not args.require_imu):
                break
            time.sleep(0.04)
    finally:
        motor_sock.close()
        imu_sock.close()

    expected_motor_size = MOTOR_FEEDBACK_HEADER.size + MOTOR_COUNT * MOTOR_FEEDBACK.size
    if motor_data is None:
        print("FAIL: no feedback from motor service", file=sys.stderr)
        return 2
    if len(motor_data) != expected_motor_size:
        print(f"FAIL: bad motor feedback size {len(motor_data)}", file=sys.stderr)
        return 2

    header = MOTOR_FEEDBACK_HEADER.unpack_from(motor_data)
    magic, version, count, _, _, status, cycle_us, command_age, rx_hz, tx_hz = header
    if (magic, version, count) != (FEEDBACK_MAGIC, VERSION, MOTOR_COUNT):
        print("FAIL: incompatible motor feedback header", file=sys.stderr)
        return 2

    print(
        f"motor: state={state_name(status)} status=0x{status:08x} "
        f"cycle={cycle_us:.0f}us command_age={command_age:.1f}ms "
        f"rx={rx_hz:.0f}fps control={tx_hz:.0f}Hz"
    )
    failed = False
    offset = MOTOR_FEEDBACK_HEADER.size
    offline = []
    joint_faults = []
    for index in range(MOTOR_COUNT):
        position, velocity, torque, temperature, error, pattern, age = (
            MOTOR_FEEDBACK.unpack_from(motor_data, offset)
        )
        offset += MOTOR_FEEDBACK.size
        online = age < 200
        if not online:
            offline.append(index)
        if index not in WHEEL_INDICES and error:
            joint_faults.append(index)
        print(
            f"  {index:02d} {MOTOR_LABELS[index]:24s} "
            f"online={str(online):5s} age={age:5d}ms temp={temperature:5.1f}C "
            f"pos={position:+8.3f} vel={velocity:+8.3f} torque={torque:+7.2f} "
            + (f"raw=0x{pattern:02x}" if index in WHEEL_INDICES else f"fault=0x{error:02x}")
        )

    if status & STATUS_SAFETY_LATCHED:
        reasons = []
        if status & STATUS_LATCH_TIMEOUT:
            reasons.append("command timeout")
        if status & STATUS_LATCH_MOTOR_OFFLINE:
            reasons.append("motor feedback loss")
        print("FAIL: motor safety latched: " + (", ".join(reasons) or "unknown"), file=sys.stderr)
        failed = True
    if offline:
        print(
            "FAIL: offline motors: " + ", ".join(MOTOR_LABELS[i] for i in offline),
            file=sys.stderr,
        )
        failed = True
    if joint_faults:
        print(
            "FAIL: joint faults: " + ", ".join(MOTOR_LABELS[i] for i in joint_faults),
            file=sys.stderr,
        )
        failed = True
    if not (status & (STATUS_READY | STATUS_RUNNING)):
        print("FAIL: motor service is not ready", file=sys.stderr)
        failed = True

    imu_healthy = False
    if imu_data is not None and len(imu_data) == IMU_FEEDBACK.size:
        imu = IMU_FEEDBACK.unpack(imu_data)
        imu_healthy = (
            imu[0] == IMU_FEEDBACK_MAGIC
            and imu[1] == VERSION
            and not (imu[6] & 1)
            and imu[16] < 100.0
        )
        print(
            f"imu: healthy={imu_healthy} status=0x{imu[6]:08x} "
            f"sensor_age={imu[16]:.1f}ms sample={imu[17]:.0f}Hz tx={imu[18]:.0f}Hz"
        )
    else:
        print("imu: no valid feedback")
    if args.require_imu and not imu_healthy:
        print("FAIL: IMU is not healthy", file=sys.stderr)
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
