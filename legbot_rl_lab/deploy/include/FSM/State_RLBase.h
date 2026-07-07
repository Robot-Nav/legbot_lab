#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "deploy_csv_logger.h"

#include <thread>
#include <chrono>
#include <memory>
#include <filesystem>
#include <iomanip>
#include <sstream>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);

    void enter()
    {
        for (int i = 0; i < (int)env->robot->data.joint_stiffness.size() && i < (int)motor_cmds.size(); ++i)
        {
            motor_cmds[i].mode = 1;
            motor_cmds[i].kp = env->robot->data.joint_stiffness[i];
            motor_cmds[i].kd = env->robot->data.joint_damping[i];
            motor_cmds[i].dq = 0;
            motor_cmds[i].tau = 0;
        }

        env->robot->update();
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });

        // CSV logger init
        log_tick_ = 0;
        csv_t0_ = std::chrono::steady_clock::now();
        csv_logger_.reset();
        if (param::csv_log_enabled) {
            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf{};
            localtime_r(&tt, &tm_buf);
            std::ostringstream name;
            name << "run_" << std::put_time(&tm_buf, "%Y-%m-%d_%H-%M-%S") << ".csv";
            const auto path = param::csv_log_dir() / name.str();
            csv_logger_ = std::make_unique<DeployCsvLogger>(path);
            if (csv_logger_->ok()) {
                std::cout << "[CSV] Velocity logging to " << path.string() << " at 50Hz\n";
            } else {
                std::cerr << "[CSV] failed to open " << path.string() << "\n";
                csv_logger_.reset();
            }
        }
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        if (csv_logger_) {
            std::cout << "[CSV] Velocity log closed\n";
            csv_logger_.reset();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // CSV diagnosis logging (50Hz)
    std::unique_ptr<DeployCsvLogger> csv_logger_;
    std::chrono::steady_clock::time_point csv_t0_{};
    int log_tick_{0};
    void logPolicySample();
};

REGISTER_FSM(State_RLBase)
