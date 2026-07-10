// 文件用途：策略终止条件。检测机体姿态是否异常，用于在部署时触发状态切换到 Passive。
#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

// 判断机体是否严重倾斜。通过投影重力 z 分量计算机体与竖直方向的夹角，超过阈值返回 true。
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::fabs(std::acos(-data[2])) > limit_angle;
}

} 
} 