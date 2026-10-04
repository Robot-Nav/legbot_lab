#!/usr/bin/env python3
"""Safety-oriented UDP client for the local ELRS/CRSF service."""

from __future__ import annotations

from collections import deque
import math
import secrets
import socket
import struct
import time

import numpy as np


COMMAND_MAGIC = 0x454C5243  # "ELRC"
FEEDBACK_MAGIC = 0x454C5246  # "ELRF"
PROTOCOL_VERSION = 1
COMMAND_VERSION = 2
STATUS_DATA_STALE = 1 << 0
COMMAND_FLAG_MOTOR_TEMPERATURES = 1 << 0
LEGACY_COMMAND_STRUCT = struct.Struct("<IHHII")
COMMAND_STRUCT = struct.Struct("<IHHII16H")
FEEDBACK_STRUCT = struct.Struct("<IHHIIII16HBBfff")


def sequence_is_newer(candidate: int, previous: int) -> bool:
    """Compare wrapping uint32 sequence numbers."""
    delta = (int(candidate) - int(previous)) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


def normalize_channel(value: int, minimum: int = 172, maximum: int = 1811) -> float:
    """Normalize a CRSF channel to the bounded range [-1, 1]."""
    center = (minimum + maximum) * 0.5
    half_range = (maximum - minimum) * 0.5
    return float(np.clip((int(value) - center) / half_range, -1.0, 1.0))


def three_position(value: float) -> int:
    if value < -0.3:
        return -1
    if value > 0.3:
        return 1
    return 0


class ElrsClient:
    """Read all 16 ELRS channels from the loopback UDP service.

    The socket is connected to one endpoint, feedback must echo one of this
    client's recent request sequences, and feedback sequence numbers must move
    forward. A link that times out after becoming healthy remains latched until
    this client process is restarted.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 55201,
        peer_timeout_ms: float = 500.0,
        emergency_active_high: bool = False,
    ) -> None:
        self.address = (str(host), int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.sock.connect(self.address)
        self.sock.setblocking(False)

        self.sequence = secrets.randbits(32)
        self._recent_commands: deque[int] = deque(maxlen=64)
        self._accepted_commands: set[int] = set()
        self.last_feedback_sequence: int | None = None
        self.last_data_sequence: int | None = None
        self.last_receive_monotonic = 0.0
        self.peer_timeout_ms = float(peer_timeout_ms)
        self.emergency_active_high = bool(emergency_active_high)
        self.server_data_age_ms = float("inf")
        self.feedback_age_ms = float("inf")
        self.data_age_ms = float("inf")
        self.data_stale = True
        self.link_timeout_latched = False
        self._ever_fresh = False
        self._server_reported_stale = True

        self.raw_channels = np.zeros(16, dtype=np.uint16)
        self.channels = np.zeros(16, dtype=np.float32)
        self.frame_id = 0
        self.sample_rate_hz = 0.0
        self.tx_rate_hz = 0.0
        self.motor_temperatures_deci_c = np.zeros(16, dtype=np.uint16)
        self.motor_temperatures_valid = False

    def close(self) -> None:
        self.sock.close()

    def send_command(self) -> int:
        sequence = self.sequence
        flags = COMMAND_FLAG_MOTOR_TEMPERATURES if self.motor_temperatures_valid else 0
        packet = COMMAND_STRUCT.pack(
            COMMAND_MAGIC,
            COMMAND_VERSION,
            flags,
            sequence,
            0,
            *(int(value) for value in self.motor_temperatures_deci_c),
        )
        self.sock.send(packet)
        self._recent_commands.append(sequence)
        self.sequence = (sequence + 1) & 0xFFFFFFFF
        return sequence

    def set_motor_temperatures(self, temperatures) -> None:
        values = np.asarray(temperatures, dtype=np.float32)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            self.motor_temperatures_valid = False
            return
        scaled = np.rint(np.clip(values, 0.0, 6553.5) * 10.0)
        self.motor_temperatures_deci_c[:] = scaled.astype(np.uint16)
        self.motor_temperatures_valid = True

    def update(self) -> bool:
        now = time.monotonic()
        # Observe a missed deadline before accepting a newly recovered packet.
        self._refresh_age(now)
        if self.link_timeout_latched:
            return False
        try:
            self.send_command()
        except (ConnectionRefusedError, ConnectionResetError):
            return False

        accepted = False
        while True:
            try:
                packet = self.sock.recv(1024)
            except BlockingIOError:
                break
            except (ConnectionRefusedError, ConnectionResetError):
                break
            if len(packet) != FEEDBACK_STRUCT.size:
                continue
            fields = FEEDBACK_STRUCT.unpack(packet)
            accepted = self._accept_feedback(fields, time.monotonic()) or accepted
            if self.link_timeout_latched:
                break

        self._refresh_age(time.monotonic())
        return accepted

    def _accept_feedback(self, fields: tuple, now: float) -> bool:
        if fields[0] != FEEDBACK_MAGIC or fields[1] != PROTOCOL_VERSION:
            return False
        feedback_sequence = int(fields[3])
        command_sequence = int(fields[5])
        if command_sequence not in self._recent_commands:
            return False
        if command_sequence in self._accepted_commands:
            return False
        if (
            self.last_feedback_sequence is not None
            and not sequence_is_newer(feedback_sequence, self.last_feedback_sequence)
        ):
            return False

        raw_channels = fields[7:23]
        server_age_ms = float(fields[25])
        sample_rate_hz = float(fields[26])
        tx_rate_hz = float(fields[27])
        if any(not 0 <= int(value) <= 2047 for value in raw_channels):
            return False
        if not all(math.isfinite(value) for value in (server_age_ms, sample_rate_hz, tx_rate_hz)):
            return False
        if server_age_ms < 0.0:
            return False

        self.last_feedback_sequence = feedback_sequence
        self._accepted_commands.add(command_sequence)
        while self._recent_commands and self._recent_commands[0] != command_sequence:
            self._accepted_commands.discard(self._recent_commands.popleft())
        self.last_data_sequence = int(fields[4])
        self.last_receive_monotonic = float(now)
        self._server_reported_stale = bool(int(fields[6]) & STATUS_DATA_STALE)
        self.server_data_age_ms = server_age_ms
        self.raw_channels[:] = raw_channels
        for index, value in enumerate(raw_channels):
            self.channels[index] = normalize_channel(value)
        self.frame_id = int(fields[23])
        self.sample_rate_hz = sample_rate_hz
        self.tx_rate_hz = tx_rate_hz
        self._refresh_age(now)
        return True

    def _refresh_age(self, now: float) -> None:
        if self.last_receive_monotonic == 0.0:
            self.feedback_age_ms = float("inf")
            self.data_age_ms = float("inf")
        else:
            self.feedback_age_ms = max(
                0.0, (float(now) - self.last_receive_monotonic) * 1000.0
            )
            self.data_age_ms = self.server_data_age_ms + self.feedback_age_ms
        self.data_stale = (
            self._server_reported_stale
            or self.feedback_age_ms > self.peer_timeout_ms
            or self.data_age_ms > self.peer_timeout_ms
        )
        if self._ever_fresh and self.data_stale:
            self.link_timeout_latched = True
        elif not self.data_stale:
            self._ever_fresh = True

    @property
    def mode_level(self) -> int:
        """CH6: -1 damping, 0 pose hold, 1 RL control."""
        if not self._ever_fresh:
            return -1
        return three_position(float(self.channels[5]))

    @property
    def speed_level(self) -> int:
        """CH7: 0 low, 1 medium, 2 high."""
        if not self._ever_fresh:
            return 0
        return three_position(float(self.channels[6])) + 1

    @property
    def emergency(self) -> bool:
        """Return the configured CH8 emergency level; stale startup is unsafe."""
        if not self._ever_fresh:
            return True
        value = float(self.channels[7])
        # Only the explicit released switch position is safe. A centered or
        # ambiguous CH8 value must hold the controller in emergency damping.
        return bool(value > -0.3 if self.emergency_active_high else value < 0.3)

    @property
    def speed_lock(self) -> bool:
        """CH9 high forces all commanded body velocities to zero."""
        if not self._ever_fresh:
            return True
        return bool(self.channels[8] > 0.3)

    def get_velocity_command(self) -> tuple[float, float, float]:
        """Map CH3/CH4/CH1 to normalized forward/lateral/yaw commands."""
        vx = float(self.channels[2])
        vy = -float(self.channels[3])
        wz = -float(self.channels[0])
        return vx, vy, wz
