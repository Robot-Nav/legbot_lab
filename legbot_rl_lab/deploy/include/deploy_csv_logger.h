#pragma once

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

// One CSV row for FixStand / Velocity layer diagnosis.
// Adapted for legbot interface architecture (uses cached motor_states/imu_state,
// not direct lowstate->msg_ access).
struct DeployCsvRow {
    const char* phase = "velocity";
    float t_sec = 0.f;
    float vx = 0.f;
    float vy = 0.f;
    float wz = 0.f;
    float grav[3]{};
    float ang_vel[3]{};
    float joint_pos_rel[12]{};     // policy order
    float joint_vel[12]{};         // policy order
    float action[12]{};            // policy order
    float q_target[12]{};          // policy order (clamped q_des)
    float q_actual[12]{};         // policy order (model q)
    float q_motor_raw[12]{};       // motor order FR,FL,RR,RL
    float tau_est[12]{};           // policy order
    float temperature[12]{};       // motor order FR,FL,RR,RL
    float kp = 0.f;
    float kd = 0.f;
};

// 50Hz CSV for sim2real layer diagnosis.
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

    FILE* fp_{nullptr};
    char buffer_[64 * 1024]{};
    std::mutex mu_;
    int rows_since_flush_{0};
};
