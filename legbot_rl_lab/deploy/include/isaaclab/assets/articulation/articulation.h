// 文件用途：机器人实体数据结构与基类。定义策略观察所需的标准化数据字段，
// 各机型通过继承 update() 将底层反馈同步到 ArticulationData。
#pragma once

#include <eigen3/Eigen/Dense>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class MotionLoader;

// 机器人实体数据：包含关节、IMU、手柄、地形扫描等策略观察输入。
struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f); // 世界系重力方向（机体坐标下为投影重力）
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);  // 机体前向方向

    std::vector<float> joint_stiffness; // 各关节 PD 刚度（下发电机 kp）
    std::vector<float> joint_damping;   // 各关节 PD 阻尼（下发电机 kd）

    Eigen::VectorXf joint_pos;          // 当前关节位置（策略顺序）
    Eigen::VectorXf default_joint_pos;  // 训练默认关节位置，用于计算相对位置观察
    Eigen::VectorXf joint_vel;          // 当前关节速度（策略顺序）

    Eigen::Vector3f root_ang_vel_b;     // 机体角速度（机体坐标系）
    Eigen::Vector3f projected_gravity_b;// 投影重力向量（机体坐标系）
    Eigen::Quaternionf root_quat_w;     // 机体姿态四元数（世界系）

    std::vector<float> joint_ids_map;   // 策略关节索引 → 电机索引映射
    std::vector<float> height_scan;     // 地形高度扫描（若可用）

    unitree::common::UnitreeJoystick* joystick = nullptr; // 手柄指针
};

// 机器人实体基类。子类在 update() 中完成底层数据到 ArticulationData 的同步。
class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    ArticulationData data;
};

};
