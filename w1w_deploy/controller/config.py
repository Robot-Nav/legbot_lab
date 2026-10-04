#!/usr/bin/env python3
"""Validated YAML configuration for the W1W policy controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class Config:
    def __init__(self, config_path: str) -> None:
        self.path = Path(config_path).resolve()
        with self.path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        self.control_dt = float(cfg["control"]["dt"])
        observation = cfg["observation"]
        self.num_actions = int(observation["num_actions"])
        self.num_obs = int(observation["num_obs"])
        self.obs_history_len = int(observation["history_len"])
        self.wheel_sim_indices = tuple(int(i) for i in observation["wheel_sim_indices"])

        robot = cfg["robot"]
        self.joint_names = tuple(robot["joint_names"])
        self.default_angles = np.asarray(robot["default_angles"], dtype=np.float32)
        self.down_angles = np.asarray(robot["down_angles"], dtype=np.float32)
        self.kps = np.asarray(robot["kp"], dtype=np.float32)
        self.kds = np.asarray(robot["kd"], dtype=np.float32)

        scales = cfg["scales"]
        self.ang_vel_scale = float(scales["ang_vel"])
        self.dof_err_scale = float(scales["dof_err"])
        self.dof_vel_scale = float(scales["dof_vel"])
        self.action_scale = float(scales["action"])
        self.vel_scale = float(scales["vel_scale"])
        self.cmd_scale = np.asarray(scales["cmd"], dtype=np.float32)
        self.max_cmd = np.asarray(cfg["max_cmd"], dtype=np.float32)

        network = cfg["network"]
        self.motor_host = str(network["motor"]["host"])
        self.motor_port = int(network["motor"]["port"])
        self.imu_host = str(network["imu"]["host"])
        self.imu_port = int(network["imu"]["port"])
        self.elrs_host = str(network["elrs"]["host"])
        self.elrs_port = int(network["elrs"]["port"])
        self.elrs_emergency_active_high = bool(
            network["elrs"].get("emergency_active_high", False)
        )

        safety = cfg["safety"]
        self.motor_ready_timeout_s = float(safety["motor_ready_timeout_s"])
        self.motor_feedback_timeout_ms = float(safety["motor_feedback_timeout_ms"])
        self.controller_send_deadline_ms = float(safety["controller_send_deadline_ms"])
        self.active_acquire_timeout_ms = float(safety["active_acquire_timeout_ms"])
        self.imu_timeout_ms = float(safety["imu_timeout_ms"])
        self.elrs_timeout_ms = float(safety["elrs_timeout_ms"])
        self.max_temperature_c = float(safety["max_temperature_c"])
        self.rl_entry_blend_s = float(safety["rl_entry_blend_s"])

        model = cfg["model"]
        self.model_path = self._resolve(model["path"])
        self._validate()

    def _resolve(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            configured = self.path.parent / path
            packaged = Path(__file__).resolve().parent / path
            path = configured if configured.exists() else packaged
        return str(path.resolve())

    def _validate(self) -> None:
        expected_wheels = (3, 7, 11, 15)
        if self.control_dt <= 0 or self.control_dt >= 0.1:
            raise ValueError("control.dt must be between 0 and 0.1 seconds")
        if not (self.control_dt * 1000.0 < self.controller_send_deadline_ms < 100.0):
            raise ValueError("controller send deadline must be above one cycle and below 100 ms")
        if not (0.0 < self.active_acquire_timeout_ms < 100.0):
            raise ValueError("active acquire timeout must be between 0 and 100 ms")
        if (self.num_actions, self.num_obs) != (16, 57):
            raise ValueError("this deployment requires 16 actions and 57 observations")
        if self.wheel_sim_indices != expected_wheels:
            raise ValueError(f"wheel_sim_indices must be {list(expected_wheels)}")
        for name, values in (
            ("joint_names", self.joint_names),
            ("default_angles", self.default_angles),
            ("down_angles", self.down_angles),
            ("kp", self.kps),
            ("kd", self.kds),
        ):
            if len(values) != self.num_actions:
                raise ValueError(f"robot.{name} must contain 16 values")
        for name, values in (
            ("default_angles", self.default_angles),
            ("down_angles", self.down_angles),
            ("kp", self.kps),
            ("kd", self.kds),
            ("cmd", self.cmd_scale),
            ("max_cmd", self.max_cmd),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if self.elrs_host not in ("127.0.0.1", "localhost"):
            raise ValueError("ELRS UDP service must use the local loopback interface")
        if not 0 < self.elrs_port <= 65535:
            raise ValueError("ELRS UDP port must be between 1 and 65535")
        if not 100.0 <= self.elrs_timeout_ms <= 1000.0:
            raise ValueError("ELRS timeout must be between 100 and 1000 ms")
