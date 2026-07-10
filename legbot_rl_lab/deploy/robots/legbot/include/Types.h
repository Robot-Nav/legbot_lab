// 文件用途：Legbot 部署类型与 DDS 接口定义。包含电机指令/反馈结构、IMU 状态、
// 抽象机器人接口、DDS 发布订阅实现，以及安全限幅函数的具体实现。
#pragma once

#include <unitree/dds_wrapper/robots/go2/go2.h>
#include "deploy_safety.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

#define HEIGHT_SCAN_PORT 19876 // 地形高度扫描 UDP 端口号
#define HEIGHT_SCAN_SIZE 187   // 高度扫描点数

// 电机指令（控制器 -> 机器人）。mode=0 失能，mode=1 使能；网关检测 mode 边沿发送 CAN 使能/失能帧。
struct MotorCmd {
    float q = 0.0f;
    float dq = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float tau = 0.0f;
    uint8_t mode = 1; // 默认使能
};

// 电机反馈（机器人 -> 控制器）。q_raw 为原始编码器值（诊断用），temperature 用于过温保护。
struct MotorState {
    float q = 0.0f;
    float dq = 0.0f;
    float tau_est = 0.0f;
    float q_raw = 0.0f;
    float temperature = 0.0f;
};

// IMU 状态：四元数、角速度、加速度、欧拉角。
struct IMUState {
    float quaternion[4] = {1, 0, 0, 0};
    float gyroscope[3] = {0, 0, 0};
    float accelerometer[3] = {0, 0, 0};
    float rpy[3] = {0, 0, 0};
};

using LowCmdPublisher = unitree::robot::go2::publisher::LowCmd;
using LowStateSubscriber = unitree::robot::go2::subscription::LowState;

// 抽象机器人接口。当前仅保留 DDS 实现，同一协议同时支持 MuJoCo sim2sim 与真机 sim2real。
class VBotInterface {
public:
    virtual ~VBotInterface() = default;

    virtual void init() = 0;
    virtual void send_cmd(const std::vector<MotorCmd> &cmds) = 0;
    virtual void get_state(std::vector<MotorState> &states, IMUState &imu) = 0;
    virtual void enable_motors() = 0;
    virtual void disable_motors() = 0;
    virtual bool is_timeout() = 0;
    virtual void update() = 0;
    virtual unitree::common::UnitreeJoystick* get_joystick() = 0;

    virtual std::vector<float> get_height_scan() { return std::vector<float>(HEIGHT_SCAN_SIZE, 0.0f); }
};

// DDS 接口：发布 rt/lowcmd，订阅 rt/lowstate。
// 连接 serial_dds_gateway 时走 sim2real；连接 MuJoCo 时走 sim2sim。
class DDSInterface : public VBotInterface {
public:
    std::shared_ptr<LowCmdPublisher> dds_lowcmd;
    std::shared_ptr<LowStateSubscriber> dds_lowstate;

    int height_scan_sock_ = -1;          // 高度扫描 UDP 套接字
    std::vector<float> height_scan_data_;

    DDSInterface() {
        dds_lowcmd = std::make_shared<LowCmdPublisher>();
        dds_lowstate = std::make_shared<LowStateSubscriber>();
        height_scan_data_.resize(HEIGHT_SCAN_SIZE, 0.0f);
    }

    void init() override {
        spdlog::info("DDS: Waiting for rt/lowstate connection...");
        dds_lowstate->wait_for_connection();
        spdlog::info("DDS: Connected to rt/lowstate.");

        // 初始化高度扫描 UDP 接收（sim2sim 时 MuJoCo 发送；sim2real 时未用到，失败可忽略）。
        height_scan_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (height_scan_sock_ >= 0) {
            int flags = fcntl(height_scan_sock_, F_GETFL, 0);
            fcntl(height_scan_sock_, F_SETFL, flags | O_NONBLOCK);

            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(HEIGHT_SCAN_PORT);
            addr.sin_addr.s_addr = INADDR_ANY;

            if (bind(height_scan_sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
                close(height_scan_sock_);
                height_scan_sock_ = -1;
                spdlog::warn("DDS: HeightScan UDP bind failed (ignored in sim2real)");
            } else {
                spdlog::info("DDS: HeightScan UDP receiver on port {}", HEIGHT_SCAN_PORT);
            }
        }
    }

    void send_cmd(const std::vector<MotorCmd> &cmds) override {
        dds_lowcmd->lock();
        for (int i = 0; i < (int)cmds.size() && i < (int)dds_lowcmd->msg_.motor_cmd().size(); i++) {
            auto &m = dds_lowcmd->msg_.motor_cmd()[i];
            m.mode() = cmds[i].mode;
            m.q() = cmds[i].q;
            m.dq() = cmds[i].dq;
            m.kp() = cmds[i].kp;
            m.kd() = cmds[i].kd;
            m.tau() = cmds[i].tau;
        }
        dds_lowcmd->unlockAndPublish();
    }

    void get_state(std::vector<MotorState> &states, IMUState &imu) override {
        for (int i = 0; i < (int)states.size() && i < (int)dds_lowstate->msg_.motor_state().size(); i++) {
            const auto &ms = dds_lowstate->msg_.motor_state()[i];
            states[i].q = ms.q();
            states[i].dq = ms.dq();
            states[i].tau_est = ms.tau_est();
            states[i].q_raw = ms.q_raw();
            states[i].temperature = static_cast<float>(ms.temperature());
        }
        const auto &im = dds_lowstate->msg_.imu_state();
        for (int i = 0; i < 4; i++) {
            imu.quaternion[i] = im.quaternion()[i];
        }
        for (int i = 0; i < 3; i++) {
            imu.gyroscope[i] = im.gyroscope()[i];
            imu.accelerometer[i] = im.accelerometer()[i];
            imu.rpy[i] = im.rpy()[i];
        }
    }

    bool is_timeout() override { return dds_lowstate->isTimeout(); }

    void update() override {
        dds_lowstate->update();

        // 非阻塞接收最新高度扫描包，只保留完整一帧。
        if (height_scan_sock_ >= 0) {
            float buffer[HEIGHT_SCAN_SIZE];
            while (true) {
                ssize_t n = recvfrom(height_scan_sock_, buffer, sizeof(buffer), MSG_DONTWAIT, NULL, NULL);
                if (n == (ssize_t)sizeof(buffer)) {
                    memcpy(height_scan_data_.data(), buffer, sizeof(buffer));
                } else {
                    break;
                }
            }
        }
    }

    unitree::common::UnitreeJoystick* get_joystick() override {
        return &dds_lowstate->joystick;
    }

    std::vector<float> get_height_scan() override {
        return height_scan_data_;
    }

    void enable_motors() override {}
    void disable_motors() override {
        // 发布一次 mode=0，触发网关发送失能 CAN 帧。
        dds_lowcmd->lock();
        for (int i = 0; i < (int)dds_lowcmd->msg_.motor_cmd().size(); i++) {
            dds_lowcmd->msg_.motor_cmd()[i].mode() = 0;
        }
        dds_lowcmd->unlockAndPublish();
        spdlog::info("DDS: Published mode=0 (disable) for all motors");
    }
};

// ---- deploy_safety.h 中声明函数的具体实现（需要 MotorState/MotorCmd 完整定义）----
namespace deploy {

inline bool motor_state_fault(const std::vector<MotorState>& states) {
    if (!safety_config.enabled) return false;
    for (int i = 0; i < (int)states.size(); ++i) {
        if (states[i].temperature > safety_config.temperature_limit) {
            spdlog::critical("Safety: motor[{}] temperature={:.1f}C > limit {:.1f}C",
                             i, states[i].temperature, safety_config.temperature_limit);
            return true;
        }
        if (std::fabs(states[i].dq) > safety_config.velocity_limit) {
            spdlog::critical("Safety: motor[{}] velocity={:.2f} > limit {:.2f}",
                             i, states[i].dq, safety_config.velocity_limit);
            return true;
        }
        if (std::fabs(states[i].tau_est) > safety_config.tau_est_limit) {
            spdlog::critical("Safety: motor[{}] tau_est={:.2f} > limit {:.2f}",
                             i, states[i].tau_est, safety_config.tau_est_limit);
            return true;
        }
    }
    return false;
}

inline void clamp_motor_cmd(MotorCmd& cmd, int idx) {
    // 将目标位置限制在关节范围内
    if (idx < (int)safety_config.joint_pos_lower.size()) {
        cmd.q = std::clamp(cmd.q, safety_config.joint_pos_lower[idx], safety_config.joint_pos_upper[idx]);
    }
    // 将力矩限制在允许范围内
    cmd.tau = std::clamp(cmd.tau, -safety_config.torque_limit, safety_config.torque_limit);
}

inline void clamp_motor_cmds(std::vector<MotorCmd>& cmds) {
    if (!safety_config.enabled) return;

    // 首次调用或电机数量变化时重置持久化缓冲。
    if (prev_q_des.size() != cmds.size()) {
        prev_q_des.assign(cmds.size(), std::numeric_limits<float>::quiet_NaN());
        prev_dq_des.assign(cmds.size(), 0.0f);
    }

    for (int i = 0; i < (int)cmds.size(); ++i) {
        // 1) 绝对限幅：关节位置范围、力矩范围。
        clamp_motor_cmd(cmds[i], i);

        // 2) 单周期位置变化率限幅。
        if (safety_config.delta_q_limit_per_tick > 0.0f &&
            !std::isnan(prev_q_des[i])) {
            float delta = cmds[i].q - prev_q_des[i];
            if (std::fabs(delta) > safety_config.delta_q_limit_per_tick) {
                const float sign = (delta > 0.0f) ? 1.0f : -1.0f;
                cmds[i].q = prev_q_des[i] + sign * safety_config.delta_q_limit_per_tick;
            }
        }

        // 3) 单周期速度变化率限幅。
        if (safety_config.delta_dq_limit_per_tick > 0.0f &&
            !std::isnan(prev_q_des[i])) {
            float delta = cmds[i].dq - prev_dq_des[i];
            if (std::fabs(delta) > safety_config.delta_dq_limit_per_tick) {
                const float sign = (delta > 0.0f) ? 1.0f : -1.0f;
                cmds[i].dq = prev_dq_des[i] + sign * safety_config.delta_dq_limit_per_tick;
            }
        }

        // 保存当前值供下一周期使用。
        prev_q_des[i] = cmds[i].q;
        prev_dq_des[i] = cmds[i].dq;
    }
}

inline void reset_delta_baseline(const std::vector<MotorCmd>& cmds) {
    prev_q_des.assign(cmds.size(), std::numeric_limits<float>::quiet_NaN());
    prev_dq_des.assign(cmds.size(), 0.0f);
}

}  // namespace deploy
