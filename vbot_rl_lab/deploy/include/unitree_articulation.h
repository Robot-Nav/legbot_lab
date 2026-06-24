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

        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = FSMState::imu_state.gyroscope[i];
        }
        data.root_quat_w = Eigen::Quaternionf(
            FSMState::imu_state.quaternion[0],
            FSMState::imu_state.quaternion[1],
            FSMState::imu_state.quaternion[2],
            FSMState::imu_state.quaternion[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
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
