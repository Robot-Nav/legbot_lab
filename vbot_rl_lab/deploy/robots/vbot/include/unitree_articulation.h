#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include "FSM/FSMState.h"
#include <cstdlib>

namespace unitree
{

inline float env_float_or(const char* name, float default_value)
{
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0') return default_value;

    char* end = nullptr;
    const float parsed = std::strtof(raw, &end);
    return end == raw ? default_value : parsed;
}

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
        data.root_ang_vel_b[0] *= env_float_or("VBOT_GYRO_SIGN_X", 1.0f);
        data.root_ang_vel_b[1] *= env_float_or("VBOT_GYRO_SIGN_Y", 1.0f);
        data.root_ang_vel_b[2] *= env_float_or("VBOT_GYRO_SIGN_Z", 1.0f);

        data.root_quat_w = Eigen::Quaternionf(
            FSMState::imu_state.quaternion[0],
            FSMState::imu_state.quaternion[1],
            FSMState::imu_state.quaternion[2],
            FSMState::imu_state.quaternion[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        data.projected_gravity_b[0] *= env_float_or("VBOT_GRAV_SIGN_X", 1.0f);
        data.projected_gravity_b[1] *= env_float_or("VBOT_GRAV_SIGN_Y", 1.0f);
        data.projected_gravity_b[2] *= env_float_or("VBOT_GRAV_SIGN_Z", 1.0f);

        for(int i(0); i< data.joint_ids_map.size(); i++) {
            int idx = (int)data.joint_ids_map[i];
            if(idx < (int)FSMState::motor_states.size()) {
                data.joint_pos[i] = FSMState::motor_states[idx].q;
                data.joint_vel[i] = FSMState::motor_states[idx].dq;
            }
        }

        if(FSMState::interface) {
            data.height_scan = FSMState::interface->get_height_scan();
        }
    }
};

}
