#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

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
}
