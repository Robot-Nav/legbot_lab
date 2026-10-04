#!/usr/bin/env python3
"""Read-only ELRS channel check using the local UDP service."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controller"))

from config import Config  # noqa: E402
from elrs_client import ElrsClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/etc/w1w/controller.yaml")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--require-safe-start", action="store_true")
    args = parser.parse_args()
    config = Config(args.config)
    client = ElrsClient(
        config.elrs_host,
        config.elrs_port,
        peer_timeout_ms=config.elrs_timeout_ms,
        emergency_active_high=config.elrs_emergency_active_high,
    )
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            client.update()
            if not client.data_stale:
                break
            time.sleep(0.01)
        vx, vy, wz = client.get_velocity_command()
        print("raw channels: " + " ".join(str(int(value)) for value in client.raw_channels))
        print(
            f"elrs: stale={client.data_stale} age={client.data_age_ms:.1f}ms "
            f"sample={client.sample_rate_hz:.0f}Hz udp={client.tx_rate_hz:.0f}Hz "
            f"mode={client.mode_level} speed={client.speed_level} "
            f"emergency={client.emergency} speed_lock={client.speed_lock} "
            f"cmd=({vx:+.3f},{vy:+.3f},{wz:+.3f})"
        )
        failed = client.data_stale or client.link_timeout_latched
        if args.require_safe_start and (
            client.mode_level != -1 or client.emergency
        ):
            print(
                "FAIL: require CH6=damping and CH8=emergency released",
                file=sys.stderr,
            )
            failed = True
        return 1 if failed else 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
