#!/usr/bin/env python3
"""Non-blocking UDP client for dm_imu_server."""

from __future__ import annotations

import socket
import struct
import time

import numpy as np


COMMAND_MAGIC = 0x494D5543
FEEDBACK_MAGIC = 0x494D5546
PROTOCOL_VERSION = 1
COMMAND_STRUCT = struct.Struct("<IHHII")
FEEDBACK_STRUCT = struct.Struct("<IHHIIII3f3f3ffff")
STATUS_DATA_STALE = 1 << 0


class ImuClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 55200) -> None:
        self.address = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sequence = 0
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.gyroscope = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.status_bits = STATUS_DATA_STALE
        self.sensor_data_age_ms = float("inf")
        self.last_feedback_monotonic = 0.0

    def close(self) -> None:
        self.sock.close()

    def _request(self) -> None:
        packet = COMMAND_STRUCT.pack(
            COMMAND_MAGIC, PROTOCOL_VERSION, 0, self.sequence & 0xFFFFFFFF, 0
        )
        self.sock.sendto(packet, self.address)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF

    def update(self) -> bool:
        self._request()
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            latest = data
        if latest is None:
            return False
        if len(latest) != FEEDBACK_STRUCT.size:
            return False

        values = FEEDBACK_STRUCT.unpack(latest)
        if values[0] != FEEDBACK_MAGIC or values[1] != PROTOCOL_VERSION:
            return False
        self.status_bits = values[6]
        self.acceleration[:] = values[7:10]
        self.gyroscope[:] = values[10:13]
        self.rpy[:] = values[13:16]
        self.sensor_data_age_ms = float(values[16])
        self.quaternion[:] = self._rpy_to_quaternion(self.rpy)
        self.last_feedback_monotonic = time.monotonic()
        return True

    @staticmethod
    def _rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float32)

    @property
    def feedback_age_ms(self) -> float:
        if self.last_feedback_monotonic == 0.0:
            return float("inf")
        return (time.monotonic() - self.last_feedback_monotonic) * 1000.0

    @property
    def healthy(self) -> bool:
        return (
            self.feedback_age_ms < 100.0
            and self.sensor_data_age_ms < 100.0
            and not (self.status_bits & STATUS_DATA_STALE)
            and np.all(np.isfinite(self.quaternion))
        )

    def get_gyroscope(self) -> np.ndarray:
        return self.gyroscope.copy()

    def get_quaternion(self) -> np.ndarray:
        return self.quaternion.copy()

