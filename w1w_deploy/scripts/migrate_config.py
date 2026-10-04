#!/usr/bin/env python3
"""Migrate a preserved W1W controller YAML from WiFi input to ELRS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import yaml


OLD_DEFAULT_ANGLES = [
    0.0, -0.6, 1.0, 0.0, 0.0, 0.6, -1.0, 0.0,
    0.0, 0.6, -1.0, 0.0, 0.0, -0.6, 1.0, 0.0,
]
TRAINING_DEFAULT_ANGLES = [
    0.0, -0.7, 1.5, 0.0, 0.0, 0.7, -1.5, 0.0,
    0.0, -0.7, 1.5, 0.0, 0.0, 0.7, -1.5, 0.0,
]
OLD_KP = [80, 80, 80, 0] * 4
TRAINING_KP = [100, 100, 100, 0] * 4


def migrate(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("controller configuration root must be a mapping")

    network = config.setdefault("network", {})
    safety = config.setdefault("safety", {})
    robot = config.setdefault("robot", {})
    if not all(isinstance(item, dict) for item in (network, safety, robot)):
        raise ValueError("network, safety and robot configuration must be mappings")

    changed = False
    if "elrs" not in network:
        network["elrs"] = {
            "host": "127.0.0.1",
            "port": 55201,
            "emergency_active_high": False,
        }
        changed = True
    if "remote" in network:
        del network["remote"]
        changed = True
    if "elrs_timeout_ms" not in safety:
        safety["elrs_timeout_ms"] = float(safety.get("remote_timeout_ms", 500.0))
        changed = True
    if "remote_timeout_ms" in safety:
        del safety["remote_timeout_ms"]
        changed = True
    if robot.get("default_angles") == OLD_DEFAULT_ANGLES:
        robot["default_angles"] = TRAINING_DEFAULT_ANGLES
        changed = True
    if robot.get("kp") == OLD_KP:
        robot["kp"] = TRAINING_KP
        changed = True
    if config.get("max_cmd") == [2.0, 1.0, 2.0]:
        config["max_cmd"] = [3.0, 1.0, 3.14]
        changed = True
    if not changed:
        return False

    backup = path.with_name(path.name + ".pre-elrs")
    if not backup.exists():
        shutil.copy2(path, backup)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if migrate(args.path):
        print(f"migrated {args.path}; backup: {args.path}.pre-elrs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
