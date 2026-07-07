#pragma once

#include <eigen3/Eigen/Dense>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class MotionLoader;

struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    std::vector<float> joint_stiffness;
    std::vector<float> joint_damping;

    Eigen::VectorXf joint_pos;

    Eigen::VectorXf default_joint_pos;

    Eigen::VectorXf joint_vel;

    Eigen::Vector3f root_ang_vel_b;

    Eigen::Vector3f projected_gravity_b;

    Eigen::Quaternionf root_quat_w;

    std::vector<float> joint_ids_map;

    std::vector<float> height_scan;

    unitree::common::UnitreeJoystick* joystick = nullptr;
};

class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    ArticulationData data;
};

};
