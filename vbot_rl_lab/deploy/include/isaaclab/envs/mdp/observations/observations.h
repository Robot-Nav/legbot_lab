#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>

namespace isaaclab
{
namespace mdp
{

inline bool read_env_float(const char* name, float& value)
{
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0') return false;

    char* end = nullptr;
    errno = 0;
    float parsed = std::strtof(raw, &end);
    if(end == raw || errno == ERANGE) return false;

    value = parsed;
    return true;
}

inline bool read_env_flag(const char* name)
{
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0') return false;

    std::string value(raw);
    return value != "0" && value != "false" && value != "False" && value != "FALSE";
}

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {

    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3, 0.0f);
    auto joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    const std::array<float, 3> lower = {
        cfg["lin_vel_x"][0].as<float>(),
        cfg["lin_vel_y"][0].as<float>(),
        cfg["ang_vel_z"][0].as<float>()
    };
    const std::array<float, 3> upper = {
        cfg["lin_vel_x"][1].as<float>(),
        cfg["lin_vel_y"][1].as<float>(),
        cfg["ang_vel_z"][1].as<float>()
    };

    if (joystick) {
        obs[0] = joystick->ly();
        obs[1] = -joystick->lx();
        obs[2] = -joystick->rx();
    }

    float env_value = 0.0f;
    bool has_fixed_cmd = false;
    bool has_scale = false;

    float scale_x = 1.0f;
    float scale_y = 1.0f;
    float scale_yaw = 1.0f;
    has_scale |= read_env_float("VBOT_CMD_SCALE_X", scale_x);
    has_scale |= read_env_float("VBOT_CMD_SCALE_Y", scale_y);
    has_scale |= read_env_float("VBOT_CMD_SCALE_YAW", scale_yaw);
    obs[0] *= scale_x;
    obs[1] *= scale_y;
    obs[2] *= scale_yaw;

    if(read_env_float("VBOT_CMD_X", env_value)) {
        obs[0] = env_value;
        has_fixed_cmd = true;
    }
    if(read_env_float("VBOT_CMD_Y", env_value)) {
        obs[1] = env_value;
        has_fixed_cmd = true;
    }
    if(read_env_float("VBOT_CMD_YAW", env_value)) {
        obs[2] = env_value;
        has_fixed_cmd = true;
    }

    for(size_t i = 0; i < obs.size(); ++i) {
        obs[i] = std::clamp(obs[i], lower[i], upper[i]);
    }

    float ramp_rate = 0.0f;
    const bool has_ramp = read_env_float("VBOT_CMD_RAMP", ramp_rate) && ramp_rate > 0.0f;
    if(has_ramp) {
        static std::vector<float> filtered(3, 0.0f);
        if(env->episode_length <= 1) {
            filtered.assign(3, 0.0f);
        }

        const float max_delta = ramp_rate * env->step_dt;
        for(size_t i = 0; i < obs.size(); ++i) {
            const float delta = std::clamp(obs[i] - filtered[i], -max_delta, max_delta);
            filtered[i] += delta;
        }
        obs = filtered;
    }

    const bool debug_cmd = read_env_flag("VBOT_DEBUG_CMD") || has_fixed_cmd || has_scale || has_ramp;
    if(debug_cmd && env->episode_length % 50 == 0) {
        static long last_logged_episode_length = -1;
        if(last_logged_episode_length != env->episode_length) {
            last_logged_episode_length = env->episode_length;
            std::cout << "[velocity_commands] step=" << env->episode_length
                      << " cmd=(" << obs[0] << ", " << obs[1] << ", " << obs[2] << ")"
                      << " joystick="
                      << (joystick ? "(" + std::to_string(joystick->ly()) + ", "
                                      + std::to_string(-joystick->lx()) + ", "
                                      + std::to_string(-joystick->rx()) + ")"
                                  : "null")
                      << std::endl;
        }
    }

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

REGISTER_OBSERVATION(height_scan)
{
    auto & asset = env->robot;
    auto & height_data = asset->data.height_scan;
    if(height_data.empty()) {
        int scan_size = 187;
        try {
            auto scale_vec = env->cfg["observations"]["height_scan"]["scale"];
            if(scale_vec.IsDefined() && scale_vec.IsSequence()) {
                scan_size = scale_vec.size();
            }
        } catch(const std::exception& e) {}
        return std::vector<float>(scan_size, 0.0f);
    }
    return height_data;
}

}
}
