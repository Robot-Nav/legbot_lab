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

    auto robot = std::make_shared<unitree::VBotArticulation>();
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        robot
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

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

    // ==================================================================
    // Pitch compensation: corrects persistent forward-tilt bias (~3-5°)
    // measured in sim2real deployment.  When gx > 0 (nose down), extend
    // front thighs (idx 4,5) to push the body back toward level.
    //
    // Policy joint layout: hip[0:FL,1:FR,2:RL,3:RR]
    //                      thigh[4:FL,5:FR,6:RL,7:RR]
    //                      calf[8:FL,9:FR,10:RL,11:RR]
    //
    // gain ≈ 0.5 rad compensates ~2.5° pitch per 5° tilt.
    // Increase if body still nose-down, decrease if oscillating.
    // ==================================================================
    {
        static constexpr float kPitchCompGain = 0.5f;
        const float gx = env->robot->data.projected_gravity_b[0];
        const float comp = kPitchCompGain * gx;          // gx > 0 ⇒ nose-down ⇒ +comp
        action[4] += comp;   // FL thigh – extend front-left leg
        action[5] += comp;   // FR thigh – extend front-right leg
    }

    // Safety: clip policy action output before writing to motor commands.
    // Real hardware cannot trust network outputs directly.
    deploy::clamp_action(action);

    const auto& map = env->robot->data.joint_ids_map;
    for(int i(0); i < (int)map.size() && i < (int)motor_cmds.size(); i++) {
        motor_cmds[map[i]].mode = 1;
        motor_cmds[map[i]].q = action[i];
    }

    // CSV diagnosis logging at 50Hz (every 20 ticks at 1kHz FSM)
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
    // velocity command (left empty if no keyboard command source)
    row.vx = 0.f; row.vy = 0.f; row.wz = 0.f;
    row.grav[0] = d.projected_gravity_b[0];
    row.grav[1] = d.projected_gravity_b[1];
    row.grav[2] = d.projected_gravity_b[2];
    row.ang_vel[0] = d.root_ang_vel_b[0];
    row.ang_vel[1] = d.root_ang_vel_b[1];
    row.ang_vel[2] = d.root_ang_vel_b[2];

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
    // motor-order fields
    for (int mot = 0; mot < 12 && mot < (int)motor_states.size(); ++mot) {
        row.q_motor_raw[mot] = motor_states[static_cast<size_t>(mot)].q_raw;
        row.temperature[mot] = motor_states[static_cast<size_t>(mot)].temperature;
    }

    csv_logger_->write_row(row);
}
