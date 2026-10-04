"""Small, deterministic helpers for controller safety decisions."""

from __future__ import annotations


class PermanentSafetyLatch:
    """Records only the first safety reason and never clears it."""

    def __init__(self) -> None:
        self.reason: str | None = None

    @property
    def latched(self) -> bool:
        return self.reason is not None

    def latch(self, reason: str) -> bool:
        if self.reason is not None:
            return False
        self.reason = str(reason)
        return True


def command_deadline_missed(
    active_control_started: bool,
    now_monotonic: float,
    last_send_monotonic: float,
    deadline_ms: float,
) -> bool:
    if not active_control_started:
        return False
    return (now_monotonic - last_send_monotonic) * 1000.0 > deadline_ms

