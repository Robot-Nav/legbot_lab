#!/usr/bin/env python3
"""W1W wheel-legged policy deployment with non-recoverable safety latching."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from config import Config
from device_license import decrypt_model, require_authorized
from elrs_client import ElrsClient
from imu_client import ImuClient
from motor_client import MOTOR_LABELS, MotorClient, MotorSafetyError
from safety import PermanentSafetyLatch, command_deadline_missed


class ControllerSafetyLatch(RuntimeError):
    """A controller-level safety condition that requires a process restart."""


class Policy:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.input_buffer = np.zeros(
            (1, config.num_obs * config.obs_history_len), dtype=np.float32
        )
        self.term_slices = []
        offset = 0
        for dimension in (3, 3, 3, 16, 16, 16):
            self.term_slices.append((offset, offset + dimension))
            offset += dimension
        self._init_onnx()
        for _ in range(3):
            action = self.infer(
                np.zeros((config.obs_history_len, config.num_obs), dtype=np.float32)
            )
            if action.shape != (config.num_actions,):
                raise RuntimeError(f"policy output has unexpected shape {action.shape}")

    def _init_onnx(self) -> None:
        import onnxruntime as ort

        if not Path(self.config.model_path).is_file():
            raise FileNotFoundError(self.config.model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model_bytes = decrypt_model(self.config.model_path)
        self.session = ort.InferenceSession(model_bytes, sess_options=options)
        self.input_name = self.session.get_inputs()[0].name
        self.run_options = ort.RunOptions()
        self.run_options.log_severity_level = 3
        self.backend = "onnx"

    def infer(self, history: np.ndarray) -> np.ndarray:
        write_offset = 0
        for start, end in self.term_slices:
            term = history[:, start:end]
            self.input_buffer[0, write_offset:write_offset + term.size] = term.ravel()
            write_offset += term.size
        output = self.session.run(
            None, {self.input_name: self.input_buffer}, self.run_options
        )[0]
        action = np.asarray(output, dtype=np.float32).ravel()
        if action.shape != (16,) or not np.all(np.isfinite(action)):
            raise ControllerSafetyLatch(f"invalid policy output: {action}")
        return action


class W1WController:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.motor = MotorClient(config.motor_host, config.motor_port)
        self.imu = ImuClient(config.imu_host, config.imu_port)
        self.elrs = ElrsClient(
            config.elrs_host,
            config.elrs_port,
            peer_timeout_ms=config.elrs_timeout_ms,
            emergency_active_high=config.elrs_emergency_active_high,
        )
        self.policy = Policy(config)
        self.wheels = np.asarray(config.wheel_sim_indices, dtype=np.int64)
        self.legs = np.asarray([i for i in range(16) if i not in self.wheels], dtype=np.int64)

        self.action = np.zeros(16, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.observation = np.zeros(57, dtype=np.float32)
        self.history = np.zeros((config.obs_history_len, 57), dtype=np.float32)
        self.position_command = np.zeros(16, dtype=np.float32)
        self.velocity_command = np.zeros(16, dtype=np.float32)
        self.zeros = np.zeros(16, dtype=np.float32)
        self.gravity = np.zeros(3, dtype=np.float32)
        self.position_error = np.zeros(16, dtype=np.float32)
        self.blend_from = config.default_angles.copy()
        self.blend_step = 0
        self.blend_steps = 0
        self.safety_latch = PermanentSafetyLatch()
        self.running = True
        self.active_control_started = False
        self.last_active_send_monotonic = 0.0

    def close(self) -> None:
        self.elrs.close()
        self.imu.close()
        self.motor.close()

    def request_stop(self, *_args) -> None:
        self.running = False

    def _update_io(self) -> None:
        self.elrs.update()
        self.imu.update()
        self.motor.update_motor_states()
        self.elrs.set_motor_temperatures(self.motor.temperatures)

    def _guard_active_deadline(self) -> None:
        if not self.active_control_started:
            return
        now = time.monotonic()
        elapsed_ms = (now - self.last_active_send_monotonic) * 1000.0
        if command_deadline_missed(
            self.active_control_started,
            now,
            self.last_active_send_monotonic,
            self.config.controller_send_deadline_ms,
        ):
            raise ControllerSafetyLatch(
                f"controller command deadline missed: {elapsed_ms:.1f} ms"
            )

    def wait_for_preflight(self) -> None:
        print("[preflight] waiting for all 16 motors")
        feedback = self.motor.wait_until_ready(self.config.motor_ready_timeout_s)
        self.motor.reset_feedback_deadline()
        print(
            f"[preflight] motors ready, rx={feedback.rx_rate_hz:.0f} fps, "
            f"control={feedback.tx_rate_hz:.0f} Hz"
        )
        print("[preflight] waiting for fresh IMU and ELRS in damping mode")
        deadline = time.monotonic() + 30.0
        next_status_log = 0.0
        last_missing = "not sampled"
        while self.running and time.monotonic() < deadline:
            started = time.monotonic()
            self.motor.send_damping()
            self._update_io()
            imu_ready = self.imu.healthy
            elrs_fresh = not self.elrs.data_stale
            mode = self.elrs.mode_level
            emergency = self.elrs.emergency
            missing = []
            if not imu_ready:
                missing.append("IMU not healthy")
            if not elrs_fresh:
                missing.append("ELRS stale")
            if mode != -1:
                missing.append(f"CH6 mode={mode}, expected -1")
            if emergency:
                missing.append("CH8 emergency active")
            last_missing = ", ".join(missing) or "none"
            if not missing:
                print("[preflight] IMU and ELRS ready; controller remains in damping")
                return
            now = time.monotonic()
            if now >= next_status_log:
                print(
                    f"[preflight] blocked by: {last_missing} | "
                    f"IMU healthy={imu_ready} feedback_age={self.imu.feedback_age_ms:.1f}ms "
                    f"sensor_age={self.imu.sensor_data_age_ms:.1f}ms "
                    f"status=0x{self.imu.status_bits:08x} | "
                    f"ELRS stale={self.elrs.data_stale} age={self.elrs.data_age_ms:.1f}ms "
                    f"CH6 raw={int(self.elrs.raw_channels[5])} "
                    f"norm={float(self.elrs.channels[5]):+.3f} mode={mode} | "
                    f"CH8 raw={int(self.elrs.raw_channels[7])} "
                    f"norm={float(self.elrs.channels[7]):+.3f} emergency={emergency}",
                    flush=True,
                )
                next_status_log = now + 1.0
            self._sleep_period(started)
        raise TimeoutError(
            "IMU/ELRS preflight failed; require CH6=damping "
            f"and CH8=emergency released; blocked by: {last_missing}"
        )

    def _check_safety(self, require_elrs: bool = True) -> None:
        self._guard_active_deadline()
        self.motor.assert_control_safe(self.config.motor_feedback_timeout_ms)
        if not self.imu.healthy or self.imu.feedback_age_ms > self.config.imu_timeout_ms:
            raise ControllerSafetyLatch(
                f"IMU stale: feedback={self.imu.feedback_age_ms:.1f} ms, "
                f"sensor={self.imu.sensor_data_age_ms:.1f} ms"
            )
        if require_elrs and (
            self.elrs.link_timeout_latched
            or self.elrs.data_stale
            or self.elrs.data_age_ms > self.config.elrs_timeout_ms
        ):
            raise ControllerSafetyLatch(
                f"ELRS stale: {self.elrs.data_age_ms:.1f} ms"
            )
        if self.elrs.emergency:
            raise ControllerSafetyLatch("ELRS CH8 emergency stop")
        hot = np.where(self.motor.temperatures > self.config.max_temperature_c)[0]
        if hot.size:
            details = ", ".join(
                f"{MOTOR_LABELS[i]}={self.motor.temperatures[i]:.1f}C" for i in hot
            )
            raise ControllerSafetyLatch(f"motor over-temperature: {details}")
        joint_faults = [
            i for i in self.legs if int(self.motor.error_codes[i]) != 0
        ]
        if joint_faults:
            details = ", ".join(
                f"{MOTOR_LABELS[i]}=0x{int(self.motor.error_codes[i]):02x}"
                for i in joint_faults
            )
            raise ControllerSafetyLatch(f"joint motor fault: {details}")

    def _damping_step(self) -> None:
        self.motor.send_damping()
        self._update_io()

    def release_active_control(self) -> None:
        if not self.active_control_started:
            return
        self.motor.release_to_damping(timeout_ms=self.config.active_acquire_timeout_ms)
        self.active_control_started = False
        self.last_active_send_monotonic = 0.0

    def _active_send(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        self._check_safety()
        sequence = self.motor.send_command(
            positions=positions,
            velocities=velocities,
            torques=self.zeros,
            kp=self.config.kps,
            kd=self.config.kds,
        )
        if not self.active_control_started:
            self.motor.confirm_active_command(
                sequence,
                timeout_ms=self.config.active_acquire_timeout_ms,
            )
            self.active_control_started = True
        self.last_active_send_monotonic = time.monotonic()

    def smooth_move(
        self,
        target: np.ndarray,
        duration_s: float,
    ) -> bool:
        # Match the proven Y2 transition: refresh feedback before taking the
        # initial pose, then interpolate linearly to the target.
        for _ in range(5):
            self._update_io()
            self._check_safety()
            time.sleep(0.005)
        initial = self.motor.positions.copy()
        steps = max(1, int(duration_s / self.config.control_dt))
        for step in range(steps):
            if not self.running:
                return False
            started = time.monotonic()
            self._update_io()
            self._check_safety()
            alpha = (step + 1) / steps
            command = initial * (1.0 - alpha) + target * alpha
            command[self.wheels] = 0.0
            self._active_send(command, self.zeros)
            self._sleep_period(started)
        return True

    def _gravity_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quaternion
        self.gravity[0] = 2.0 * (-qz * qx + qw * qy)
        self.gravity[1] = -2.0 * (qz * qy + qw * qx)
        self.gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
        return self.gravity

    def _speed_limits(self) -> tuple[float, float, float]:
        if self.elrs.speed_level == 0:
            return 0.5, 0.5, 0.5
        if self.elrs.speed_level == 1:
            return 1.0, 1.0, 1.5
        return tuple(float(value) for value in self.config.max_cmd)

    def update_observation(self) -> None:
        self._update_io()
        self._check_safety()
        vx, vy, wz = self.elrs.get_velocity_command()
        limits = self._speed_limits()
        values = (vx, vy, wz)
        for i, value in enumerate(values):
            self.command[i] = 0.0 if abs(value) < 0.03 else limits[i] * value
        np.subtract(self.motor.positions, self.config.default_angles, out=self.position_error)
        self.position_error *= self.config.dof_err_scale
        self.position_error[self.wheels] = 0.0
        self.observation[:3] = self.imu.gyroscope * self.config.ang_vel_scale
        self.observation[3:6] = self._gravity_from_quaternion(self.imu.quaternion)
        self.observation[6:9] = self.command * self.config.cmd_scale
        self.observation[9:25] = self.position_error
        self.observation[25:41] = self.motor.velocities * self.config.dof_vel_scale
        self.observation[41:57] = self.action
        if not np.all(np.isfinite(self.observation)):
            raise ControllerSafetyLatch("observation contains NaN or Inf")
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.observation

    def prepare_rl(self, hold_target: np.ndarray) -> None:
        self.action.fill(0.0)
        self.blend_from[:] = hold_target
        self.blend_step = 0
        self.blend_steps = max(1, int(self.config.rl_entry_blend_s / self.config.control_dt))
        self.update_observation()
        self.history[:] = self.observation

    def rl_step(self) -> None:
        started = time.monotonic()
        self.update_observation()
        policy_action = np.clip(self.policy.infer(self.history), -10.0, 10.0)
        self.position_command[:] = (
            self.config.default_angles + policy_action * self.config.action_scale
        )
        self.position_command[self.wheels] = 0.0
        self.velocity_command.fill(0.0)
        self.velocity_command[self.wheels] = policy_action[self.wheels] * self.config.vel_scale

        if self.blend_step < self.blend_steps:
            alpha = (self.blend_step + 1) / self.blend_steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            self.position_command[self.legs] = (
                self.blend_from[self.legs] * (1.0 - alpha)
                + self.position_command[self.legs] * alpha
            )
            self.velocity_command[self.wheels] *= alpha
            policy_action[self.legs] = (
                self.position_command[self.legs] - self.config.default_angles[self.legs]
            ) / self.config.action_scale
            policy_action[self.wheels] *= alpha
            self.blend_step += 1
        self._active_send(self.position_command, self.velocity_command)
        self.action[:] = policy_action
        self._sleep_period(started)

    def _sleep_period(self, started: float) -> None:
        remaining = self.config.control_dt - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    def latch(self, reason: str) -> None:
        if self.safety_latch.latch(reason):
            try:
                # Apply damping immediately using the already locked source,
                # then stop transmitting so the service watchdog also latches.
                self.motor.send_damping()
            except OSError as error:
                print(f"[SAFETY] immediate damping send failed: {error}", file=sys.stderr)
            print(f"[SAFETY] controller latched damping: {reason}", file=sys.stderr)

    def run_latched_damping(self) -> None:
        # Do not refresh the active-controller watchdog. The motor service will
        # latch within 100 ms and continues generating damping on its own.
        while self.running:
            time.sleep(0.1)

    def run(self) -> None:
        self.wait_for_preflight()
        state = "DAMPING"
        previous_mode = -1
        damping_ready = True
        hold_target = self.config.default_angles.copy()
        print("[state] DAMPING; use stand only with the robot safely supported")

        try:
            while self.running and not self.safety_latch.latched:
                started = time.monotonic()
                self._damping_step() if state == "DAMPING" else self._update_io()
                self._check_safety()
                raw_mode = self.elrs.mode_level
                mode = previous_mode if abs(raw_mode - previous_mode) > 1 else raw_mode
                mode_changed = mode != previous_mode
                previous_mode = mode

                if state == "DAMPING":
                    if mode == -1:
                        damping_ready = True
                    if damping_ready and mode_changed and mode == 0:
                        print("[state] standing through crouch pose")
                        if not self.smooth_move(self.config.down_angles, 2.0):
                            break
                        if not self.smooth_move(self.config.default_angles, 2.0):
                            break
                        hold_target[:] = self.config.default_angles
                        state = "HOLDING"
                        damping_ready = False
                        print("[state] HOLDING")
                    else:
                        self._sleep_period(started)
                elif state == "HOLDING":
                    command = hold_target.copy()
                    command[self.wheels] = 0.0
                    self._active_send(command, self.zeros)
                    if mode_changed and mode == 1:
                        self.prepare_rl(hold_target)
                        state = "RL_CONTROL"
                        print("[state] RL_CONTROL")
                    elif mode_changed and mode == -1:
                        if not self.smooth_move(self.config.down_angles, 2.0):
                            break
                        self.release_active_control()
                        state = "DAMPING"
                        damping_ready = False
                        print("[state] DAMPING")
                    else:
                        self._sleep_period(started)
                elif state == "RL_CONTROL":
                    if mode_changed and mode == 0:
                        hold_target[:] = self.motor.positions
                        hold_target[self.wheels] = 0.0
                        state = "HOLDING"
                        print("[state] HOLDING")
                    else:
                        self.rl_step()
        except (ControllerSafetyLatch, MotorSafetyError, TimeoutError, OSError) as error:
            self.latch(str(error))

        if self.running and self.safety_latch.latched:
            self.run_latched_damping()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="controller YAML configuration",
    )
    return parser.parse_args()


def main() -> int:
    require_authorized("w1w-controller")
    args = parse_args()
    config = Config(args.config)
    controller = W1WController(config)
    signal.signal(signal.SIGINT, controller.request_stop)
    signal.signal(signal.SIGTERM, controller.request_stop)
    try:
        controller.run()
    except Exception as error:
        controller.latch(f"startup failure: {error}")
        if controller.running:
            controller.run_latched_damping()
        return 1
    finally:
        # Stopping an active controller intentionally lets the motor service's
        # 100 ms watchdog latch. Do not send a disable command here.
        if controller.active_control_started:
            try:
                controller.motor.send_damping()
            except OSError:
                pass
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
