// 文件用途：Legbot 机器人实体封装（机器人专属版本）。将 DDS 底层反馈转换为
// Isaac Lab 风格数据，并同步地形高度扫描供策略观察使用。
#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include "FSM/FSMState.h"

namespace unitree
{

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
        // 机体姿态四元数
        data.root_quat_w = Eigen::Quaternionf(
            FSMState::imu_state.quaternion[0],
            FSMState::imu_state.quaternion[1],
            FSMState::imu_state.quaternion[2],
            FSMState::imu_state.quaternion[3]
        );
        // 投影重力向量
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // 按 joint_ids_map 将电机顺序映射为策略关节顺序
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            int idx = (int)data.joint_ids_map[i];
            if(idx < (int)FSMState::motor_states.size()) {
                data.joint_pos[i] = FSMState::motor_states[idx].q;
                data.joint_vel[i] = FSMState::motor_states[idx].dq;
            }
        }

        // 同步地形高度扫描（若可用）
        if(FSMState::interface) {
            data.height_scan = FSMState::interface->get_height_scan();
        }
    }
};

}
