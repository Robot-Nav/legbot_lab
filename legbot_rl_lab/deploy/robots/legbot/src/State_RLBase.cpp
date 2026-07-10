// 文件用途：Legbot 速度控制状态实现。构造 RL 环境与 ONNX 推理器，
// 每个 FSM 周期读取最新策略动作并映射到电机指令，同时记录 CSV 诊断数据。
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <iostream>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // 构造 Legbot 机器人实体与 RL 环境，并加载 ONNX 策略模型。
    auto robot = std::make_shared<unitree::VBotArticulation>();
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        robot
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // 姿态异常时切换回 Passive。
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();

    // ==========================================================
    // 俯仰补偿：修正 sim2real 中实测的持续前倾偏差（约 3-5°）。
    // 当投影重力 gx > 0 时机头下倾，给前大腿（索引 4、5）加正补偿，
    // 把机身推回水平。增益 0.5 rad 对应约 5° 倾角补偿 2.5°。
    // 若仍前倾可增大增益；若出现振荡则减小。
    //
    // 策略关节顺序：hip[0:FL,1:FR,2:RL,3:RR]
    //               thigh[4:FL,5:FR,6:RL,7:RR]
    //               calf[8:FL,9:FR,10:RL,11:RR]
    // ==========================================================
    {
        static constexpr float kPitchCompGain = 0.5f;
        const float gx = env->robot->data.projected_gravity_b[0];
        const float comp = kPitchCompGain * gx; // gx > 0 表示机头下倾，需正补偿
        action[4] += comp; // 左前大腿
        action[5] += comp; // 右前大腿
    }

    // 对策略输出进行安全限幅。
    deploy::clamp_action(action);

    // 按 joint_ids_map 将策略顺序动作映射到电机顺序指令。
    const auto& map = env->robot->data.joint_ids_map;
    for(int i(0); i < (int)map.size() && i < (int)motor_cmds.size(); i++) {
        motor_cmds[map[i]].mode = 1;
        motor_cmds[map[i]].q = action[i];
    }

    // 50Hz CSV 诊断记录（每 20 个 1kHz 周期一次）。
    if (csv_logger_ && ++log_tick_ >= 20) {
        log_tick_ = 0;
        logPolicySample();
    }
}

void State_RLBase::logPolicySample()
{
    if (!csv_logger_ || !env) {
        return;
    }

    const float t = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - csv_t0_).count();
    const auto& d = env->robot->data;
    const auto& map = d.joint_ids_map;
    const auto action = env->action_manager->processed_actions();

    DeployCsvRow row{};
    row.phase = "velocity";
    row.t_sec = t;
    // 速度指令字段当前留空（若无键盘速度源）
    row.vx = 0.f; row.vy = 0.f; row.wz = 0.f;
    row.grav[0] = d.projected_gravity_b[0];
    row.grav[1] = d.projected_gravity_b[1];
    row.grav[2] = d.projected_gravity_b[2];
    row.ang_vel[0] = d.root_ang_vel_b[0];
    row.ang_vel[1] = d.root_ang_vel_b[1];
    row.ang_vel[2] = d.root_ang_vel_b[2];

    // 策略顺序数据。
    const int n = static_cast<int>(map.size());
    for (int i = 0; i < n && i < 12; ++i) {
        const int mi = static_cast<int>(map[static_cast<size_t>(i)]);
        row.joint_pos_rel[i] = d.joint_pos[i];
        row.joint_vel[i] = d.joint_vel[i];
        row.action[i] = action[i];
        if (mi < (int)motor_cmds.size()) {
            row.q_target[i] = motor_cmds[static_cast<size_t>(mi)].q;
        }
        if (mi < (int)motor_states.size()) {
            row.q_actual[i] = motor_states[static_cast<size_t>(mi)].q;
            row.tau_est[i] = motor_states[static_cast<size_t>(mi)].tau_est;
        }
        if (i == 0 && mi < (int)motor_cmds.size()) {
            row.kp = motor_cmds[static_cast<size_t>(mi)].kp;
            row.kd = motor_cmds[static_cast<size_t>(mi)].kd;
        }
    }
    // 电机顺序原始数据。
    for (int mot = 0; mot < 12 && mot < (int)motor_states.size(); ++mot) {
        row.q_motor_raw[mot] = motor_states[static_cast<size_t>(mot)].q_raw;
        row.temperature[mot] = motor_states[static_cast<size_t>(mot)].temperature;
    }

    csv_logger_->write_row(row);
}
