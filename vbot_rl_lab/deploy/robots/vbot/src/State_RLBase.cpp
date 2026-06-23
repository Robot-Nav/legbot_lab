#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace
{
bool env_flag(const char* name)
{
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0') return false;

    std::string value(raw);
    return value != "0" && value != "false" && value != "False" && value != "FALSE";
}

std::string format_vec(const std::vector<float>& values)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "[";
    for(size_t i = 0; i < values.size(); ++i) {
        if(i > 0) oss << ", ";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}
}

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
    for(int i(0); i < (int)env->robot->data.joint_ids_map.size() && i < (int)motor_cmds.size(); i++) {
        motor_cmds[env->robot->data.joint_ids_map[i]].q = action[i];
    }

    if(env_flag("VBOT_DEBUG_ACTIONS") && env->episode_length % 50 == 0) {
        static long last_logged_episode_length = -1;
        if(last_logged_episode_length != env->episode_length) {
            last_logged_episode_length = env->episode_length;

            std::vector<float> motor_order_q(motor_cmds.size(), 0.0f);
            for(int i = 0; i < (int)motor_cmds.size(); ++i) {
                motor_order_q[i] = motor_cmds[i].q;
            }

            std::cout << "[rl_action] step=" << env->episode_length
                      << " raw=" << format_vec(env->action_manager->action())
                      << " policy_q=" << format_vec(action)
                      << " motor_q=" << format_vec(motor_order_q)
                      << std::endl;
        }
    }
}
