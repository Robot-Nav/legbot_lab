#!/usr/bin/env python3
"""Client for the unified 12-joint and 4-wheel motor service."""

from __future__ import annotations

import socket
import secrets
import struct
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


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
PROTOCOL_VERSION = 1
COMMAND_HEADER = struct.Struct("<IHHII")
COMMAND_MOTOR = struct.Struct("<fffff")
FEEDBACK_HEADER = struct.Struct("<IHHIIIffff")
FEEDBACK_MOTOR = struct.Struct("<ffffBBH")
COMMAND_PACKET_SIZE = COMMAND_HEADER.size + MOTOR_COUNT * COMMAND_MOTOR.size
FEEDBACK_PACKET_SIZE = FEEDBACK_HEADER.size + MOTOR_COUNT * FEEDBACK_MOTOR.size

STATUS_COMMAND_TIMEOUT = 1 << 0
STATUS_MOTOR_OFFLINE_MASK = ((1 << MOTOR_COUNT) - 1) << 1
STATUS_SAFETY_LATCHED = 1 << 17
STATUS_WAITING_READY = 1 << 18
STATUS_READY = 1 << 19
STATUS_RUNNING = 1 << 20
STATUS_LATCH_TIMEOUT = 1 << 21
STATUS_LATCH_MOTOR_OFFLINE = 1 << 22
STATUS_RELEASE_SUPPORTED = 1 << 23
COMMAND_FLAG_RELEASE_CONTROL = 1 << 0


class MotorProtocolError(RuntimeError):
    """The motor service returned an invalid packet."""


class MotorSafetyError(RuntimeError):
    """Active control is unsafe or the motor service has latched damping."""


@dataclass(frozen=True)
class MotorFeedback:
    position: float
    velocity: float
    torque: float
    temperature: float
    error_code: int
    pattern: int
    comm_age_ms: int


@dataclass(frozen=True)
class FeedbackPacket:
    sequence: int
    command_sequence: int
    status_bits: int
    cycle_time_us: float
    command_age_ms: float
    rx_rate_hz: float
    tx_rate_hz: float
    motors: tuple[MotorFeedback, ...]

    @property
    def safety_latched(self) -> bool:
        return bool(self.status_bits & STATUS_SAFETY_LATCHED)

    @property
    def waiting_ready(self) -> bool:
        return bool(self.status_bits & STATUS_WAITING_READY)

    @property
    def ready(self) -> bool:
        return bool(self.status_bits & STATUS_READY)

    @property
    def running(self) -> bool:
        return bool(self.status_bits & STATUS_RUNNING)

    @property
    def offline_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(MOTOR_COUNT) if self.status_bits & (1 << (i + 1)))

    @property
    def latch_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.status_bits & STATUS_LATCH_TIMEOUT:
            reasons.append("command timeout")
        if self.status_bits & STATUS_LATCH_MOTOR_OFFLINE:
            reasons.append("motor feedback loss")
        return tuple(reasons)


def damping_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zeros = np.zeros(MOTOR_COUNT, dtype=np.float32)
    kd = np.full(MOTOR_COUNT, 2.0, dtype=np.float32)
    kd[list(WHEEL_INDICES)] = 1.0
    return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), kd


def build_command_packet(
    sequence: int,
    torques: Sequence[float],
    positions: Sequence[float],
    velocities: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    flags: int = 0,
) -> bytes:
    arrays = [np.asarray(values, dtype=np.float32) for values in (torques, positions, velocities, kp, kd)]
    if any(values.shape != (MOTOR_COUNT,) for values in arrays):
        raise ValueError(f"all command arrays must have shape ({MOTOR_COUNT},)")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("motor command contains NaN or Inf")

    packet = bytearray(COMMAND_HEADER.pack(
        COMMAND_MAGIC, PROTOCOL_VERSION, MOTOR_COUNT, sequence & 0xFFFFFFFF, flags & 0xFFFFFFFF
    ))
    for values in zip(*arrays):
        packet.extend(COMMAND_MOTOR.pack(*(float(value) for value in values)))
    return bytes(packet)


def parse_feedback_packet(data: bytes) -> FeedbackPacket:
    if len(data) != FEEDBACK_PACKET_SIZE:
        raise MotorProtocolError(
            f"expected {FEEDBACK_PACKET_SIZE} feedback bytes, received {len(data)}"
        )
    header = FEEDBACK_HEADER.unpack_from(data)
    magic, version, motor_count, sequence, command_sequence, status_bits, cycle, age, rx, tx = header
    if magic != FEEDBACK_MAGIC:
        raise MotorProtocolError(f"bad feedback magic 0x{magic:08x}")
    if version != PROTOCOL_VERSION or motor_count != MOTOR_COUNT:
        raise MotorProtocolError(
            f"unsupported feedback version/count {version}/{motor_count}"
        )

    motors = []
    offset = FEEDBACK_HEADER.size
    for _ in range(MOTOR_COUNT):
        motors.append(MotorFeedback(*FEEDBACK_MOTOR.unpack_from(data, offset)))
        offset += FEEDBACK_MOTOR.size
    return FeedbackPacket(
        sequence, command_sequence, status_bits, cycle, age, rx, tx, tuple(motors)
    )


class MotorClient:
    """Single-socket client for all 16 motors.

    Damping packets do not claim active control in can_service_wheel. The first
    non-damping packet does. Once claimed, this client must keep transmitting
    faster than the service watchdog until the process exits.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 55100) -> None:
        self.address = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sequence = secrets.randbits(32)
        self.last_feedback: Optional[FeedbackPacket] = None
        self.last_feedback_monotonic = 0.0
        self.max_feedback_gap_ms = 0.0

        self.positions = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.velocities = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.torques = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.temperatures = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.error_codes = np.zeros(MOTOR_COUNT, dtype=np.uint8)
        self.patterns = np.zeros(MOTOR_COUNT, dtype=np.uint8)
        self.comm_ages = np.full(MOTOR_COUNT, 65535, dtype=np.uint16)

    def close(self) -> None:
        self.sock.close()

    def send_command(
        self,
        positions: Sequence[float],
        velocities: Sequence[float],
        torques: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        flags: int = 0,
    ) -> int:
        sequence = self.sequence
        packet = build_command_packet(
            sequence, torques, positions, velocities, kp, kd, flags=flags
        )
        self.sock.sendto(packet, self.address)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return sequence

    def send_damping(self) -> None:
        torques, positions, velocities, kp, kd = damping_arrays()
        self.send_command(positions, velocities, torques, kp, kd)

    set_damping_mode = send_damping

    def release_to_damping(self, timeout_ms: float = 50.0) -> FeedbackPacket:
        feedback = self.last_feedback
        if feedback is None or not (feedback.status_bits & STATUS_RELEASE_SUPPORTED):
            raise MotorSafetyError("motor service does not advertise safe control release")
        torques, positions, velocities, kp, kd = damping_arrays()
        self.send_command(
            positions, velocities, torques, kp, kd,
            flags=COMMAND_FLAG_RELEASE_CONTROL,
        )
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            self.update_motor_states()
            feedback = self.last_feedback
            if feedback is not None:
                if feedback.safety_latched:
                    raise MotorSafetyError("motor service latched during control release")
                if feedback.ready and not feedback.offline_indices:
                    return feedback
            time.sleep(0.001)
        raise MotorSafetyError("motor service did not confirm safe control release")

    def update_motor_states(self) -> bool:
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            latest = parse_feedback_packet(data)
        if latest is None:
            return False

        now = time.monotonic()
        if self.last_feedback_monotonic:
            gap_ms = (now - self.last_feedback_monotonic) * 1000.0
            self.max_feedback_gap_ms = max(self.max_feedback_gap_ms, gap_ms)
        self.last_feedback = latest
        self.last_feedback_monotonic = now
        for index, motor in enumerate(latest.motors):
            self.positions[index] = motor.position
            self.velocities[index] = motor.velocity
            self.torques[index] = motor.torque
            self.temperatures[index] = motor.temperature
            self.error_codes[index] = motor.error_code
            self.patterns[index] = motor.pattern
            self.comm_ages[index] = motor.comm_age_ms
        return True

    def confirm_active_command(self, sequence: int, timeout_ms: float = 50.0) -> FeedbackPacket:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            self.update_motor_states()
            feedback = self.last_feedback
            if feedback is not None:
                if feedback.safety_latched:
                    reasons = ", ".join(feedback.latch_reasons) or "unknown reason"
                    raise MotorSafetyError(f"motor service latched during acquire: {reasons}")
                if feedback.running and feedback.command_sequence == (sequence & 0xFFFFFFFF):
                    return feedback
            time.sleep(0.001)
        feedback = self.last_feedback
        detail = "no feedback" if feedback is None else (
            f"status=0x{feedback.status_bits:08x}, "
            f"command_sequence={feedback.command_sequence}, expected={sequence & 0xFFFFFFFF}"
        )
        raise MotorSafetyError(f"active motor control was not acquired: {detail}")

    @property
    def feedback_age_ms(self) -> float:
        if self.last_feedback is None:
            return float("inf")
        return (time.monotonic() - self.last_feedback_monotonic) * 1000.0

    def reset_feedback_deadline(self) -> None:
        self.max_feedback_gap_ms = 0.0

    def wait_until_ready(self, timeout_s: float, damping_rate_hz: float = 50.0) -> FeedbackPacket:
        deadline = time.monotonic() + timeout_s
        period = 1.0 / damping_rate_hz
        while time.monotonic() < deadline:
            started = time.monotonic()
            self.send_damping()
            time.sleep(min(period * 0.25, 0.005))
            self.update_motor_states()
            feedback = self.last_feedback
            if feedback is not None:
                if feedback.safety_latched:
                    reasons = ", ".join(feedback.latch_reasons) or "unknown reason"
                    raise MotorSafetyError(
                        f"motor service safety is latched ({reasons}); restart is required"
                    )
                if feedback.ready and not feedback.offline_indices:
                    return feedback
            delay = period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

        feedback = self.last_feedback
        if feedback is None:
            detail = "no feedback from motor service"
        elif feedback.offline_indices:
            names = ", ".join(MOTOR_LABELS[i] for i in feedback.offline_indices)
            detail = f"offline motors: {names}"
        else:
            detail = f"service status 0x{feedback.status_bits:08x}"
        raise TimeoutError(f"motor service was not ready within {timeout_s:.1f}s: {detail}")

    def assert_control_safe(self, max_feedback_age_ms: float = 100.0) -> FeedbackPacket:
        feedback = self.last_feedback
        if feedback is None or self.feedback_age_ms > max_feedback_age_ms:
            raise MotorSafetyError(
                f"motor feedback is stale ({self.feedback_age_ms:.1f} ms)"
            )
        if self.max_feedback_gap_ms > max_feedback_age_ms:
            raise MotorSafetyError(
                f"motor feedback deadline was missed ({self.max_feedback_gap_ms:.1f} ms)"
            )
        if feedback.safety_latched:
            reasons = ", ".join(feedback.latch_reasons) or "unknown reason"
            raise MotorSafetyError(f"motor service latched damping: {reasons}")
        if feedback.offline_indices:
            names = ", ".join(MOTOR_LABELS[i] for i in feedback.offline_indices)
            raise MotorSafetyError(f"motor feedback lost: {names}")
        if not (feedback.ready or feedback.running):
            raise MotorSafetyError(f"motor service is not ready: 0x{feedback.status_bits:08x}")
        return feedback

    def get_joint_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.velocities.copy()

    def get_joint_torques(self) -> np.ndarray:
        return self.torques.copy()
