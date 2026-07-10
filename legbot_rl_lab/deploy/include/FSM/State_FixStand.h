// 文件用途：FSM 固定站立状态。通过分段线性插值将机器人从当前姿态平滑引导到目标姿态，
// 作为进入速度控制（RL 策略）前的准备状态，同时记录 50Hz CSV 诊断数据。
#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"
#include "deploy_csv_logger.h"

#include <chrono>
#include <memory>
#include <iomanip>
#include <sstream>

#include <eigen3/Eigen/Dense>

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand")
    : FSMState(state, state_string)
    {
        // 读取插值时间点与目标关节角度序列，二者长度必须一致。
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());

        if (param::config["diagnostics"] && param::config["diagnostics"]["fixstand_log_seconds"]) {
            fixstand_log_seconds_ = param::config["diagnostics"]["fixstand_log_seconds"].as<float>();
        }
    }

    void enter()
    {
        // 配置 PD 参数：站立姿态刚度较大，保证插值过程中姿态稳定。
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kp.size() && i < (int)motor_cmds.size(); ++i)
        {
            motor_cmds[i].mode = 1;
            motor_cmds[i].kp = kp[i];
            motor_cmds[i].kd = kd[i];
            motor_cmds[i].dq = 0;
            motor_cmds[i].tau = 0;
        }

        // 插值起点 q0 设为当前实际关节角度，保证轨迹连续。
        std::vector<float> q0;
        for(int i(0); i < kp.size() && i < (int)motor_states.size(); ++i) {
            q0.push_back(motor_states[i].q);
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;

        // 初始化 CSV 日志器。
        log_tick_ = 0;
        csv_t0_ = std::chrono::steady_clock::now();
        csv_logger_.reset();
        if (param::csv_log_enabled) {
            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf{};
            localtime_r(&tt, &tm_buf);
            std::ostringstream name;
            name << "fixstand_" << std::put_time(&tm_buf, "%Y-%m-%d_%H-%M-%S") << ".csv";
            const auto path = param::csv_log_dir() / name.str();
            csv_logger_ = std::make_unique<DeployCsvLogger>(path);
            if (csv_logger_->ok()) {
                std::cout << "[CSV] FixStand logging to " << path.string()
                          << " for " << fixstand_log_seconds_ << "s at 50Hz\n";
            } else {
                std::cerr << "[CSV] failed to open " << path.string() << "\n";
                csv_logger_.reset();
            }
        }
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);

        for(int i(0); i < (int)q.size() && i < (int)motor_cmds.size(); ++i) {
            motor_cmds[i].mode = 1;
            motor_cmds[i].q = q[i];
            motor_cmds[i].dq = 0;
            motor_cmds[i].tau = 0;
        }

        // 在设定时长内按 50Hz 记录 CSV 诊断数据（每 20 个 1kHz 周期一次）。
        if (csv_logger_ && t <= fixstand_log_seconds_) {
            if (++log_tick_ >= 20) {
                log_tick_ = 0;
                logFixStandSample(t);
            }
        }
    }

    void exit()
    {
        if (csv_logger_) {
            std::cout << "[CSV] FixStand log closed\n";
            csv_logger_.reset();
        }
    }

private:
    static constexpr int kLogEveryTicks = 20;  // 1kHz / 20 = 50Hz

    void logFixStandSample(float t_sec)
    {
        DeployCsvRow row{};
        row.phase = "fixstand";
        row.t_sec = t_sec;

        // 由 IMU 四元数计算机体坐标系下的投影重力。
        Eigen::Quaternionf quat(
            FSMState::imu_state.quaternion[0],
            FSMState::imu_state.quaternion[1],
            FSMState::imu_state.quaternion[2],
            FSMState::imu_state.quaternion[3]);
        const Eigen::Vector3f grav = quat.conjugate() * Eigen::Vector3f(0.f, 0.f, -1.f);
        row.grav[0] = grav[0];
        row.grav[1] = grav[1];
        row.grav[2] = grav[2];
        row.ang_vel[0] = FSMState::imu_state.gyroscope[0];
        row.ang_vel[1] = FSMState::imu_state.gyroscope[1];
        row.ang_vel[2] = FSMState::imu_state.gyroscope[2];

        // 电机顺序原始数据。
        for (int mot = 0; mot < 12 && mot < (int)motor_states.size(); ++mot) {
            row.q_motor_raw[mot] = motor_states[mot].q_raw;
            row.temperature[mot] = motor_states[mot].temperature;
        }
        // 策略顺序数据。
        for (int i = 0; i < 12 && i < (int)motor_cmds.size(); ++i) {
            row.q_target[i] = motor_cmds[i].q;
            row.q_actual[i] = (i < (int)motor_states.size()) ? motor_states[i].q : 0.f;
            row.tau_est[i] = (i < (int)motor_states.size()) ? motor_states[i].tau_est : 0.f;
            row.joint_vel[i] = (i < (int)motor_states.size()) ? motor_states[i].dq : 0.f;
        }
        if ((int)motor_cmds.size() > 0) {
            row.kp = motor_cmds[0].kp;
            row.kd = motor_cmds[0].kd;
        }

        csv_logger_->write_row(row);
    }

    double t0_{0.0};                                   // 进入状态的起始时间（秒）
    int log_tick_{0};                                  // CSV 采样计数
    float fixstand_log_seconds_{10.0f};                // CSV 记录时长
    std::vector<float> ts_;                            // 插值时间点
    std::vector<std::vector<float>> qs_;               // 对应目标关节角度
    std::unique_ptr<DeployCsvLogger> csv_logger_;      // CSV 日志器
    std::chrono::steady_clock::time_point csv_t0_{};   // 日志起始时刻
};

REGISTER_FSM(State_FixStand)
