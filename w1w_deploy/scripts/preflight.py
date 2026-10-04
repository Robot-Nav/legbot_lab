#!/usr/bin/env python3
"""Read-only/safe-command health check for the W1W motor and IMU services."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controller"))

try:
    from config import Config  # noqa: E402
    from imu_client import ImuClient  # noqa: E402
    from motor_client import MOTOR_LABELS, MotorClient  # noqa: E402
except ModuleNotFoundError as error:
    raise SystemExit(
        "controller Python dependencies are missing; run setup_policy_env.sh"
    ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/etc/w1w/controller.yaml")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--require-imu", action="store_true")
    args = parser.parse_args()

    config = Config(args.config)
    motor = MotorClient(config.motor_host, config.motor_port)
    imu = ImuClient(config.imu_host, config.imu_port)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            motor.send_damping()
            imu.update()
            time.sleep(0.01)
            motor.update_motor_states()
            feedback = motor.last_feedback
            if feedback is not None and (not args.require_imu or imu.healthy):
                break
            time.sleep(0.04)

        feedback = motor.last_feedback
        if feedback is None:
            print("FAIL: no feedback from motor service", file=sys.stderr)
            return 2

        state = (
            "LATCHED" if feedback.safety_latched else
            "RUNNING" if feedback.running else
            "READY" if feedback.ready else
            "WAITING"
        )
        print(
            f"motor: state={state} status=0x{feedback.status_bits:08x} "
            f"rx={feedback.rx_rate_hz:.0f}fps control={feedback.tx_rate_hz:.0f}Hz"
        )
        for leg in range(4):
            fields = []
            for index in range(leg * 4, leg * 4 + 4):
                item = feedback.motors[index]
                fields.append(
                    f"{MOTOR_LABELS[index]} age={item.comm_age_ms}ms "
                    f"temp={item.temperature:.1f}C"
                )
            print("  " + " | ".join(fields))

        failed = False
        if feedback.safety_latched:
            print(
                "FAIL: motor safety latched: "
                + (", ".join(feedback.latch_reasons) or "unknown"),
                file=sys.stderr,
            )
            failed = True
        if feedback.offline_indices:
            names = ", ".join(MOTOR_LABELS[i] for i in feedback.offline_indices)
            print(f"FAIL: offline motors: {names}", file=sys.stderr)
            failed = True
        if not (feedback.ready or feedback.running):
            print("FAIL: motor service is not ready", file=sys.stderr)
            failed = True

        print(
            f"imu: healthy={imu.healthy} feedback_age={imu.feedback_age_ms:.1f}ms "
            f"sensor_age={imu.sensor_data_age_ms:.1f}ms status=0x{imu.status_bits:08x}"
        )
        if args.require_imu and not imu.healthy:
            print("FAIL: IMU is not healthy", file=sys.stderr)
            failed = True
        return 1 if failed else 0
    finally:
        imu.close()
        motor.close()


if __name__ == "__main__":
    raise SystemExit(main())
