#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "param.h"

struct MotorState;
struct MotorCmd;
struct IMUState;

namespace deploy {

// Safety protection configuration. All limits are loaded from config.yaml[safety].
// Real hardware cannot trust network outputs directly — every command must be clamped
// and every feedback must be monitored for runaway conditions.
struct SafetyConfig {
    bool enabled = true;

    // action clip: policy output raw clip range
    float action_clip = 100.0f;

    // joint position limits (per-joint, motor order FR,FL,RR,RL)
    std::vector<float> joint_pos_lower;   // size 12
    std::vector<float> joint_pos_upper;   // size 12

    // torque limit (Nm): applied to commanded tau
    float torque_limit = 40.0f;
    // tau_est monitor: if measured torque exceeds this, trigger Passive
    float tau_est_limit = 45.0f;

    // velocity limit (rad/s): if |dq| exceeds this, trigger Passive
    float velocity_limit = 30.0f;

    // joint angle delta limit per tick (rad): limits |q_des[t] - q_des[t-1]|
    // Prevents neural network output spikes from commanding sudden large position
    // jumps that could damage motors or destabilize the robot.
    // At 1kHz FSM, 0.05 rad/tick ~ 50 rad/s equivalent (generous; tune down for safety).
    float delta_q_limit_per_tick = 0.05f;
    // joint velocity delta limit per tick (rad/s): limits |dq_des[t] - dq_des[t-1]|
    float delta_dq_limit_per_tick = 1.0f;

    // roll/pitch protection (rad): if |roll| or |pitch| exceeds threshold, trigger Passive
    float roll_threshold = 0.5f;    // ~28 deg
    float pitch_threshold = 0.5f;    // ~28 deg

    // temperature protection (degC): if motor temp exceeds this, trigger Passive
    float temperature_limit = 80.0f;

    // emergency stop flag (set by joystick/keyboard)
    bool emergency_stop = false;
};

inline SafetyConfig safety_config;

// Persistent previous-tick commanded q_des / dq_des for delta limiting.
// Initialized to NaN; first tick uses the commanded value as baseline (no clamp).
inline std::vector<float> prev_q_des;
inline std::vector<float> prev_dq_des;

// Load safety config from config.yaml[safety]. Called once at startup.
inline void load_safety_config() {
    auto s = param::config["safety"];
    if (!s) {
        spdlog::warn("Safety: no [safety] section in config.yaml, using defaults");
        return;
    }
    if (s["enabled"]) safety_config.enabled = s["enabled"].as<bool>();
    if (s["action_clip"]) safety_config.action_clip = s["action_clip"].as<float>();
    if (s["torque_limit"]) safety_config.torque_limit = s["torque_limit"].as<float>();
    if (s["tau_est_limit"]) safety_config.tau_est_limit = s["tau_est_limit"].as<float>();
    if (s["velocity_limit"]) safety_config.velocity_limit = s["velocity_limit"].as<float>();
    if (s["roll_threshold"]) safety_config.roll_threshold = s["roll_threshold"].as<float>();
    if (s["pitch_threshold"]) safety_config.pitch_threshold = s["pitch_threshold"].as<float>();
    if (s["temperature_limit"]) safety_config.temperature_limit = s["temperature_limit"].as<float>();
    if (s["joint_pos_lower"]) safety_config.joint_pos_lower = s["joint_pos_lower"].as<std::vector<float>>();
    if (s["joint_pos_upper"]) safety_config.joint_pos_upper = s["joint_pos_upper"].as<std::vector<float>>();
    if (s["delta_q_limit_per_tick"]) safety_config.delta_q_limit_per_tick = s["delta_q_limit_per_tick"].as<float>();
    if (s["delta_dq_limit_per_tick"]) safety_config.delta_dq_limit_per_tick = s["delta_dq_limit_per_tick"].as<float>();

    spdlog::info("Safety: enabled={}, action_clip={:.2f}, torque_limit={:.1f}, "
                 "velocity_limit={:.1f}, delta_q/tick={:.4f}, delta_dq/tick={:.3f}, "
                 "roll_thresh={:.2f}, pitch_thresh={:.2f}, temp_limit={:.1f}",
                 safety_config.enabled, safety_config.action_clip,
                 safety_config.torque_limit, safety_config.velocity_limit,
                 safety_config.delta_q_limit_per_tick, safety_config.delta_dq_limit_per_tick,
                 safety_config.roll_threshold, safety_config.pitch_threshold,
                 safety_config.temperature_limit);
}

// ---- Checks (return true => fault => transition to Passive) ----

// Roll/pitch fault from IMU rpy.
inline bool roll_pitch_fault(const float rpy[3]) {
    if (!safety_config.enabled) return false;
    return std::fabs(rpy[0]) > safety_config.roll_threshold ||
           std::fabs(rpy[1]) > safety_config.pitch_threshold;
}

// Motor feedback fault: temperature, velocity, tau_est.
// Uses MotorState fields: temperature, dq, tau_est.
inline bool motor_state_fault(const std::vector<MotorState>& states);

// Communication timeout fault (delegates to interface->is_timeout()).
inline bool comm_timeout_fault(bool is_timeout) {
    if (!safety_config.enabled) return false;
    return is_timeout;
}

// Emergency stop requested (joystick/keyboard).
inline bool emergency_stop_requested() {
    return safety_config.emergency_stop;
}

inline void request_emergency_stop() {
    safety_config.emergency_stop = true;
    spdlog::critical("Safety: EMERGENCY STOP requested");
}

inline void clear_emergency_stop() {
    safety_config.emergency_stop = false;
}

// ---- Clamps (applied to commands before sending) ----

// Clamp policy action output.
inline void clamp_action(std::vector<float>& action) {
    if (!safety_config.enabled) return;
    for (auto& a : action) {
        a = std::clamp(a, -safety_config.action_clip, safety_config.action_clip);
    }
}

// Clamp a motor command: q_des to joint limits, tau to torque limit.
inline void clamp_motor_cmd(MotorCmd& cmd, int idx);

// Clamp all motor commands: absolute limits first, then delta (rate-of-change) limits.
// Delta limiting prevents neural network output spikes from commanding sudden large
// position/velocity jumps between consecutive control ticks.
// Implementation in Types.h (needs full MotorCmd definition).
inline void clamp_motor_cmds(std::vector<MotorCmd>& cmds);

// Reset delta-limit baseline (call on state transition so the first command of
// the new state is not clamped against the previous state's last command).
inline void reset_delta_baseline(const std::vector<MotorCmd>& cmds);

}  // namespace deploy
