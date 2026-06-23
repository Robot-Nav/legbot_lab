// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include "isaaclab/utils/utils.h"
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace isaaclab
{

inline bool env_flag_or_false(const char* name)
{
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0') return false;

    std::string value(raw);
    return value != "0" && value != "false" && value != "False" && value != "FALSE";
}

inline std::string format_obs_slice(const std::vector<float>& values, size_t begin, size_t count)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "[";
    for(size_t i = 0; i < count && begin + i < values.size(); ++i) {
        if(i > 0) oss << ", ";
        oss << values[begin + i];
    }
    oss << "]";
    return oss.str();
}

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        observation_manager->reset();
        if (alg) alg->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        if(env_flag_or_false("VBOT_DEBUG_OBS") && episode_length % 50 == 0) {
            const auto it = obs.find("obs");
            if(it != obs.end()) {
                const auto& v = it->second;
                std::cout << "[obs] step=" << episode_length
                          << " ang=" << format_obs_slice(v, 0, 3)
                          << " grav=" << format_obs_slice(v, 3, 3)
                          << " cmd=" << format_obs_slice(v, 6, 3)
                          << " qrel=" << format_obs_slice(v, 9, 12)
                          << " dq=" << format_obs_slice(v, 21, 12)
                          << " last=" << format_obs_slice(v, 33, 12)
                          << std::endl;
            }
        }
        auto action = alg->act(obs);
        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};
