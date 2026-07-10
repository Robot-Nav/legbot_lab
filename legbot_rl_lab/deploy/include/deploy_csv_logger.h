// 文件用途：部署层 CSV 诊断日志，按 50Hz 记录 FixStand / Velocity 阶段的关键状态量与电机指令，
// 用于 sim2real 分层问题定位。日志顺序与策略关节顺序一致，便于离线对比仿真与实际机器人。
#pragma once

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

// 一条 CSV 记录：覆盖阶段、指令、IMU、关节位置/速度/动作、目标/实际角度、力矩、温度等。
struct DeployCsvRow {
    const char* phase = "velocity";
    float t_sec = 0.f;
    float vx = 0.f;
    float vy = 0.f;
    float wz = 0.f;
    float grav[3]{};
    float ang_vel[3]{};
    float joint_pos_rel[12]{};     // 策略输入顺序的相对关节位置
    float joint_vel[12]{};         // 策略输入顺序的关节速度
    float action[12]{};            // 策略输出动作
    float q_target[12]{};          // 策略顺序的目标关节角度（已限幅）
    float q_actual[12]{};          // 策略顺序的实际关节角度
    float q_motor_raw[12]{};       // 电机顺序 FR,FL,RR,RL 的原始编码器角度
    float tau_est[12]{};           // 策略顺序的估计关节力矩
    float temperature[12]{};       // 电机顺序 FR,FL,RR,RL 的温度
    float kp = 0.f;
    float kd = 0.f;
};

// 50Hz CSV 日志器：缓存写入、每 50 行刷盘一次，降低 IO 对 1kHz 控制线程的影响。
class DeployCsvLogger {
public:
    explicit DeployCsvLogger(const std::filesystem::path& path)
    {
        fp_ = std::fopen(path.string().c_str(), "w");
        if (!fp_) {
            return;
        }
        std::setvbuf(fp_, buffer_, _IOFBF, sizeof(buffer_));
        write_header();
    }

    ~DeployCsvLogger() { close(); }

    DeployCsvLogger(const DeployCsvLogger&) = delete;
    DeployCsvLogger& operator=(const DeployCsvLogger&) = delete;

    bool ok() const { return fp_ != nullptr; }

    void write_row(const DeployCsvRow& row)
    {
        if (!fp_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        std::fprintf(fp_, "%s,%.6f,%.6f,%.6f,%.6f",
                     row.phase, row.t_sec, row.vx, row.vy, row.wz);
        for (int i = 0; i < 3; ++i) {
            std::fprintf(fp_, ",%.6f", row.grav[i]);
        }
        for (int i = 0; i < 3; ++i) {
            std::fprintf(fp_, ",%.6f", row.ang_vel[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.joint_pos_rel[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.joint_vel[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.action[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.q_target[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.q_actual[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.q_motor_raw[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.tau_est[i]);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",%.6f", row.temperature[i]);
        }
        std::fprintf(fp_, ",%.6f,%.6f\n", row.kp, row.kd);

        if (++rows_since_flush_ >= 50) {
            std::fflush(fp_);
            rows_since_flush_ = 0;
        }
    }

    void close()
    {
        if (!fp_) {
            return;
        }
        std::fflush(fp_);
        std::fclose(fp_);
        fp_ = nullptr;
    }

private:
    void write_header()
    {
        std::fprintf(fp_, "phase,t,vx_cmd,vy_cmd,wz_cmd,grav_x,grav_y,grav_z,"
                           "ang_vel_x,ang_vel_y,ang_vel_z");
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",joint_pos_rel_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",joint_vel_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",action_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",q_target_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",q_actual_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",q_motor_raw_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",tau_est_%d", i);
        }
        for (int i = 0; i < 12; ++i) {
            std::fprintf(fp_, ",temperature_%d", i);
        }
        std::fprintf(fp_, ",kp_0,kd_0\n");
    }

    FILE* fp_{nullptr};                   // 文件句柄
    char buffer_[64 * 1024]{};            // 64KB 写入缓存，减少系统调用
    std::mutex mu_;                       // 多线程写保护
    int rows_since_flush_{0};             // 距离上次刷盘的行数
};
