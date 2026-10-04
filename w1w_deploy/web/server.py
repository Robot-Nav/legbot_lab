#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮足四足机器人网页监控服务。

与板上的 can_service 通过 UDP 通信，协议与 src/can_service.cpp 完全一致：
  - 命令包 CommandPacketNet  -> 发往 can_service 监听端口 (默认 55100)
  - 反馈包 FeedbackPacketNet  <- can_service 回送给命令来源地址

显示 12 个 RobStride 关节和 4 个 W190 轮毂的反馈与 CAN 在线状态。

仅依赖 Python 标准库，可直接跑在板子的 Buildroot 系统上。
"""

import argparse
import json
import math
import socket
import struct
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

try:
    from device_license import require_authorized
except ModuleNotFoundError:  # source-tree tests; sealed builds place it beside server
    from controller.device_license import require_authorized

# ---------------------------------------------------------------------------
# 协议常量 (必须与 can_service.cpp 保持一致)
# ---------------------------------------------------------------------------
MOTOR_COUNT = 16
PROTOCOL_VERSION = 2
LEGACY_PROTOCOL_VERSION = 1
COMMAND_MAGIC = 0x5242434D   # "RBCM"
FEEDBACK_MAGIC = 0x52424642  # "RBFb"

# CommandPacketNet: magic(I) version(H) motor_count(H) sequence(I) reserved(I)
#   + MotorCommandNet[16] = (torque, position, velocity, kp, kd) 各 float
CMD_HEADER = struct.Struct("<IHHII")
CMD_MOTOR = struct.Struct("<fffff")

# FeedbackPacketNet: magic(I) version(H) motor_count(H) sequence(I)
#   command_sequence(I) status_bits(I) cycle_time_us(f) command_age_ms(f)
#   rx_rate_hz(f) tx_rate_hz(f) + MotorFeedbackNet[16]
# MotorFeedbackNet: position(f) velocity(f) torque(f) temperature(f)
#   error_code(B) pattern(B) comm_age_ms(H)
FB_HEADER = struct.Struct("<IHHIIIffff")
FB_MOTOR = struct.Struct("<ffffBBH")
FB_MOTOR_V2 = struct.Struct("<fffffBBH")

# 电机布局与 can_service_wheel.cpp 一致，每条腿为 hip/thigh/calf/wheel。
MOTOR_LABELS = [
    "左前髋", "左前大腿", "左前小腿", "左前轮毂",
    "右前髋", "右前大腿", "右前小腿", "右前轮毂",
    "左后髋", "左后大腿", "左后小腿", "左后轮毂",
    "右后髋", "右后大腿", "右后小腿", "右后轮毂",
]
WHEEL_INDICES = {3, 7, 11, 15}

# RS02 故障位解析 (通信类型2 反馈帧 29位ID bit16~21, 即 error_code 的 bit0~5)
FAULT_BITS = [
    (0, "欠压故障 (电压<12V)"),
    (1, "三相电流故障"),
    (2, "过温故障"),
    (3, "磁编码故障"),
    (4, "堵转过载故障"),
    (5, "未标定"),
]

# 模式状态 (pattern)
PATTERN_NAMES = {0: "Reset/复位", 1: "Cali/标定", 2: "Motor/运行"}
# 通信在线判定阈值：超过该毫秒数未收到该电机反馈视为离线
OFFLINE_THRESHOLD_MS = 200
STATUS_COMMAND_TIMEOUT = 1 << 0
STATUS_SAFETY_LATCHED = 1 << 17
STATUS_WAITING_READY = 1 << 18
STATUS_READY = 1 << 19
STATUS_RUNNING = 1 << 20
STATUS_LATCH_TIMEOUT = 1 << 21
STATUS_LATCH_MOTOR_OFFLINE = 1 << 22
STATUS_RELEASE_SUPPORTED = 1 << 23

W190_FAULT_BITS = [
    (1, "过压故障"),
    (2, "过流故障"),
    (3, "过温故障"),
    (4, "过速故障"),
    (5, "编码器故障"),
    (6, "欠压故障"),
]


def decode_faults(error_code: int, wheel: bool = False) -> List[str]:
    """把 6 位故障码解析为可读故障名列表。"""
    if wheel:
        return [name for bit, name in W190_FAULT_BITS if error_code & (1 << bit)]
    faults = []
    for bit, name in FAULT_BITS:
        if error_code & (1 << bit):
            faults.append(name)
    return faults


def build_fixed_damping_packet(sequence: int) -> bytes:
    packet = bytearray(CMD_HEADER.pack(
        COMMAND_MAGIC, PROTOCOL_VERSION, MOTOR_COUNT, sequence & 0xFFFFFFFF, 0
    ))
    for index in range(MOTOR_COUNT):
        kd = 1.0 if index in WHEEL_INDICES else 5.0
        packet.extend(CMD_MOTOR.pack(0.0, 0.0, 0.0, 0.0, kd))
    return bytes(packet)


class MotorBridge:
    """与 can_service 的 UDP 通信桥：持续发送命令、接收反馈。"""

    def __init__(self, controller_host: str, controller_port: int, tx_hz: float) -> None:
        self._addr = (controller_host, controller_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)

        self._lock = threading.Lock()
        self._latest: Dict[str, object] = {}
        self._last_feedback_monotonic = 0.0
        self._seq = 0

        self._tx_period = 1.0 / max(tx_hz, 1.0)
        self._stop = threading.Event()
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread.start()
        self._rx_thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def get_command_view(self) -> Dict[str, object]:
        return {
            "mode": "fixed_damping",
            "joint_kd": 5.0,
            "wheel_kd": 1.0,
            "active_control": False,
        }

    # ---- 发送循环 ----
    def _build_command(self) -> bytes:
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            seq = self._seq
        return build_fixed_damping_packet(seq)

    def _tx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sock.sendto(self._build_command(), self._addr)
            except OSError:
                pass
            time.sleep(self._tx_period)

    # ---- 接收循环 ----
    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self._parse_feedback(data)

    def _parse_feedback(self, data: bytes) -> None:
        if len(data) < FB_HEADER.size:
            return
        (magic, version, motor_count, seq, cmd_seq, status_bits,
         cycle_us, cmd_age_ms, rx_hz, tx_hz) = FB_HEADER.unpack_from(data, 0)
        if version == PROTOCOL_VERSION:
            motor_struct = FB_MOTOR_V2
        elif version == LEGACY_PROTOCOL_VERSION:
            motor_struct = FB_MOTOR
        else:
            return
        expected_size = FB_HEADER.size + MOTOR_COUNT * motor_struct.size
        if (magic != FEEDBACK_MAGIC or motor_count != MOTOR_COUNT or
                len(data) != expected_size):
            return

        motors = []
        off = FB_HEADER.size
        for i in range(min(motor_count, MOTOR_COUNT)):
            if off + FB_MOTOR.size > len(data):
                break
            if version == PROTOCOL_VERSION:
                pos, vel, tor, temp, voltage, err, pattern, comm_age = \
                    FB_MOTOR_V2.unpack_from(data, off)
            else:
                pos, vel, tor, temp, err, pattern, comm_age = \
                    FB_MOTOR.unpack_from(data, off)
                voltage = None
            off += motor_struct.size
            online = comm_age < OFFLINE_THRESHOLD_MS
            wheel = i in WHEEL_INDICES
            motors.append({
                "index": i,
                "label": MOTOR_LABELS[i] if i < len(MOTOR_LABELS) else f"motor{i}",
                "position": round(pos, 4),
                "velocity": round(vel, 4),
                "torque": round(tor, 4),
                "temperature": round(temp, 2),
                "voltage": (
                    round(voltage, 2)
                    if voltage is not None and math.isfinite(voltage)
                    else None
                ),
                "pattern": pattern,
                "pattern_name": (
                    ("使能" if pattern & 0x01 else "失能") + f" · 0x{pattern:02X}" if wheel
                    else PATTERN_NAMES.get(pattern, f"未知({pattern})")
                ),
                "error_code": err,
                "faults": decode_faults(pattern if wheel else err, wheel),
                "raw_status": pattern if wheel else None,
                "enabled": bool(pattern & 0x01) if wheel else None,
                "comm_age_ms": comm_age,
                "online": online,
            })

        if status_bits & STATUS_SAFETY_LATCHED:
            safety_state = "latched"
        elif status_bits & STATUS_RUNNING:
            safety_state = "running"
        elif status_bits & STATUS_READY:
            safety_state = "ready"
        else:
            safety_state = "waiting_ready"
        latch_reasons = []
        if status_bits & STATUS_LATCH_TIMEOUT:
            latch_reasons.append("命令超过100ms")
        if status_bits & STATUS_LATCH_MOTOR_OFFLINE:
            latch_reasons.append("电机反馈掉线")

        with self._lock:
            self._last_feedback_monotonic = time.monotonic()
            self._latest = {
                "ok": True,
                "timestamp": time.time(),
                "sequence": seq,
                "command_sequence": cmd_seq,
                "status_bits": status_bits,
                "command_stale": bool(status_bits & STATUS_COMMAND_TIMEOUT),
                "safety_state": safety_state,
                "safety_latched": bool(status_bits & STATUS_SAFETY_LATCHED),
                "release_supported": bool(status_bits & STATUS_RELEASE_SUPPORTED),
                "latch_reasons": latch_reasons,
                "cycle_time_us": round(cycle_us, 1),
                "command_age_ms": round(cmd_age_ms, 2),
                "rx_rate_hz": round(rx_hz, 1),
                "tx_rate_hz": round(tx_hz, 1),
                "motors": motors,
            }

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            if not self._latest:
                return {"ok": False, "motors": [], "reason": "尚未收到 can_service 反馈"}
            feedback_age_ms = (time.monotonic() - self._last_feedback_monotonic) * 1000.0
            if feedback_age_ms > 1000.0:
                return {
                    "ok": False,
                    "motors": [],
                    "reason": f"can_service 反馈中断 {feedback_age_ms:.0f} ms",
                }
            state = json.loads(json.dumps(self._latest, allow_nan=False))
            state["web_feedback_age_ms"] = round(feedback_age_ms, 1)
            return state


class Handler(BaseHTTPRequestHandler):
    bridge: Optional[MotorBridge] = None
    static_dir: Path = Path(__file__).resolve().parent

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/state"):
            self._send_json(self.bridge.get_state())
            return
        if self.path.startswith("/api/command_view"):
            self._send_json(self.bridge.get_command_view())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/damping":
            self._send_json({"ok": True, "mode": "fixed_damping"})
            return
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Monitor is read-only")

    # ---- helpers ----
    def _read_json(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _send_json(self, obj: Dict) -> None:
        data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, filename: str, content_type: str) -> None:
        path = self.static_dir / filename
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    require_authorized("w1w-web")
    parser = argparse.ArgumentParser(description="轮足机器人只读网页监控服务")
    parser.add_argument("--http-host", default="0.0.0.0", help="网页监听地址")
    parser.add_argument("--http-port", type=int, default=8080, help="网页端口")
    parser.add_argument("--controller-host", default="127.0.0.1", help="can_service 地址")
    parser.add_argument("--controller-port", type=int, default=55100, help="can_service UDP 端口")
    parser.add_argument("--tx-hz", type=float, default=100.0, help="命令下发频率")
    args = parser.parse_args()

    bridge = MotorBridge(args.controller_host, args.controller_port, args.tx_hz)
    Handler.bridge = bridge

    server = ThreadingHTTPServer((args.http_host, args.http_port), Handler)
    print(f"网页服务: http://{args.http_host}:{args.http_port}")
    print(f"对接 can_service: {args.controller_host}:{args.controller_port}, 命令 {args.tx_hz}Hz")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        server.server_close()


if __name__ == "__main__":
    main()
