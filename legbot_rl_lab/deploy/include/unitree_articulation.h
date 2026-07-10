// 文件用途：Legbot 机器人实体封装。将 DDS 底层反馈（IMU、电机状态）转换为 Isaac Lab 风格的
// ArticulationData，供观察管理器与策略网络使用。
#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include "FSM/FSMState.h"

namespace unitree
{

// Legbot 机器人实体：把底层 motor_states / imu_state / joystick 同步到策略输入数据结构。
class VBotArticulation : public isaaclab::Articulation
{
public:
    VBotArticulation() {}

    void update() override
    {
        data.joystick = FSMState::interface->get_joystick();

        // 机体角速度（机体坐标系）
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = FSMState::imu_state.gyroscope[i];
        }
        // 机体姿态四元数（w, x, y, z）
        data.root_quat_w = Eigen::Quaternionf(
            FSMState::imu_state.quaternion[0],
            FSMState::imu_state.quaternion[1],
            FSMState::imu_state.quaternion[2],
            FSMState::imu_state.quaternion[3]
        );
        // 投影重力向量：世界系重力经机体姿态逆变换到机体坐标系，用于观察输入。
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // 按 joint_ids_map 将电机顺序映射到策略关节顺序。
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            int idx = (int)data.joint_ids_map[i];
            if(idx < (int)FSMState::motor_states.size()) {
                data.joint_pos[i] = FSMState::motor_states[idx].q;
                data.joint_vel[i] = FSMState::motor_states[idx].dq;
            }
        }
    }
};

}
