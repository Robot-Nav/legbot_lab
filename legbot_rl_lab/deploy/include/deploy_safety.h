// 文件用途：部署层安全保护。所有阈值从 config.yaml[safety] 加载，
// 对策略输出和电机反馈进行限幅与故障监测，防止真实硬件失控。
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

// 安全保护配置。真实硬件不能直接信任网络输出，所有指令必须经过限幅，所有反馈必须被监控。
struct SafetyConfig {
    bool enabled = true;                 // 总开关

    float action_clip = 100.0f;          // 策略原始输出限幅范围

    std::vector<float> joint_pos_lower;  // 各关节位置下限（电机顺序 FR,FL,RR,RL，长度 12）
    std::vector<float> joint_pos_upper;  // 各关节位置上限（电机顺序 FR,FL,RR,RL，长度 12）

    float torque_limit = 40.0f;          // 力矩指令上限（Nm）
    float tau_est_limit = 45.0f;         // 估计力矩监测阈值，超过则进入 Passive

    float velocity_limit = 30.0f;        // 关节速度监测阈值（rad/s），超过则进入 Passive

    // 单控制周期（1kHz）内关节角度最大变化量（rad），防止策略输出突变损坏电机或导致失稳。
    // 0.05 rad/周期 约等价于 50 rad/s，裕量较大，实际部署可酌情收紧。
    float delta_q_limit_per_tick = 0.05f;
    float delta_dq_limit_per_tick = 1.0f; // 单控制周期内关节速度指令最大变化量（rad/s）

    float roll_threshold = 0.5f;         // 横滚角阈值（rad，约 28°），超过则进入 Passive
    float pitch_threshold = 0.5f;        // 俯仰角阈值（rad，约 28°），超过则进入 Passive

    float temperature_limit = 80.0f;     // 电机温度阈值（℃），超过则进入 Passive

    bool emergency_stop = false;         // 急停标志（手柄/键盘触发）
};

inline SafetyConfig safety_config;

// 上一周期的目标关节角度/速度，用于变化率限幅。初始化为 NaN，首周期以当前指令为基准，不做限幅。
inline std::vector<float> prev_q_des;
inline std::vector<float> prev_dq_des;

// 从 config.yaml[safety] 加载安全参数，启动时调用一次。
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

// ---- 故障检测（返回 true 表示触发故障，FSM 将切换到 Passive）----

// IMU 横滚/俯仰角越限检测。
inline bool roll_pitch_fault(const float rpy[3]) {
    if (!safety_config.enabled) return false;
    return std::fabs(rpy[0]) > safety_config.roll_threshold ||
           std::fabs(rpy[1]) > safety_config.pitch_threshold;
}

// 电机反馈故障检测：温度、速度、估计力矩任一越限。
inline bool motor_state_fault(const std::vector<MotorState>& states);

// 通信超时故障：由接口层的 is_timeout() 提供状态。
inline bool comm_timeout_fault(bool is_timeout) {
    if (!safety_config.enabled) return false;
    return is_timeout;
}

// 急停请求状态。
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

// ---- 指令限幅（发送给硬件前执行）----

// 对策略原始输出进行全局限幅。
inline void clamp_action(std::vector<float>& action) {
    if (!safety_config.enabled) return;
    for (auto& a : action) {
        a = std::clamp(a, -safety_config.action_clip, safety_config.action_clip);
    }
}

// 对单个电机指令限幅：q_des 限制在关节范围内，tau 限制在力矩范围内。
inline void clamp_motor_cmd(MotorCmd& cmd, int idx);

// 对所有电机指令限幅：先执行绝对限幅，再执行周期变化率限幅。
// 变化率限幅可防止相邻控制周期之间出现位置/速度突变。
// 完整实现位于 Types.h（需要 MotorCmd 的完整定义）。
inline void clamp_motor_cmds(std::vector<MotorCmd>& cmds);

// 重置变化率限幅基准。在状态切换时调用，避免新状态首周期指令被旧状态最后一周期指令约束。
inline void reset_delta_baseline(const std::vector<MotorCmd>& cmds);

}  // namespace deploy
