#pragma once

#include <unitree/dds_wrapper/robots/go2/go2.h>
#include "robstride_motor.h"

struct MotorCmd {
    float q = 0.0f;
    float dq = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float tau = 0.0f;
};

struct MotorState {
    float q = 0.0f;
    float dq = 0.0f;
    float tau_est = 0.0f;
};

struct IMUState {
    float quaternion[4] = {1, 0, 0, 0};
    float gyroscope[3] = {0, 0, 0};
    float accelerometer[3] = {0, 0, 0};
    float rpy[3] = {0, 0, 0};
};

using LowCmdPublisher = unitree::robot::go2::publisher::LowCmd;
using LowStateSubscriber = unitree::robot::go2::subscription::LowState;

class VBotInterface {
public:
    enum Mode { SIM2SIM, SIM2REAL };
    Mode mode;

    VBotInterface(Mode m) : mode(m) {}
    virtual ~VBotInterface() = default;

    virtual void init() = 0;
    virtual void send_cmd(const std::vector<MotorCmd> &cmds) = 0;
    virtual void get_state(std::vector<MotorState> &states, IMUState &imu) = 0;
    virtual void enable_motors() = 0;
    virtual void disable_motors() = 0;
    virtual bool is_timeout() = 0;
    virtual void update() = 0;
    virtual unitree::common::UnitreeJoystick* get_joystick() = 0;
};

class Sim2SimInterface : public VBotInterface {
public:
    std::shared_ptr<LowCmdPublisher> dds_lowcmd;
    std::shared_ptr<LowStateSubscriber> dds_lowstate;

    Sim2SimInterface() : VBotInterface(SIM2SIM) {
        dds_lowcmd = std::make_shared<LowCmdPublisher>();
        dds_lowstate = std::make_shared<LowStateSubscriber>();
    }

    void init() override {
        spdlog::info("Sim2Sim: Waiting for DDS connection...");
        dds_lowstate->wait_for_connection();
        spdlog::info("Sim2Sim: Connected.");
    }

    void send_cmd(const std::vector<MotorCmd> &cmds) override {
        dds_lowcmd->lock();
        for (int i = 0; i < (int)cmds.size() && i < (int)dds_lowcmd->msg_.motor_cmd().size(); i++) {
            auto &m = dds_lowcmd->msg_.motor_cmd()[i];
            m.q() = cmds[i].q;
            m.dq() = cmds[i].dq;
            m.kp() = cmds[i].kp;
            m.kd() = cmds[i].kd;
            m.tau() = cmds[i].tau;
        }
        dds_lowcmd->unlockAndPublish();
    }

    void get_state(std::vector<MotorState> &states, IMUState &imu) override {
        for (int i = 0; i < (int)states.size() && i < (int)dds_lowstate->msg_.motor_state().size(); i++) {
            states[i].q = dds_lowstate->msg_.motor_state()[i].q();
            states[i].dq = dds_lowstate->msg_.motor_state()[i].dq();
            states[i].tau_est = dds_lowstate->msg_.motor_state()[i].tau_est();
        }
        for (int i = 0; i < 4; i++) {
            imu.quaternion[i] = dds_lowstate->msg_.imu_state().quaternion()[i];
        }
        for (int i = 0; i < 3; i++) {
            imu.gyroscope[i] = dds_lowstate->msg_.imu_state().gyroscope()[i];
            imu.accelerometer[i] = dds_lowstate->msg_.imu_state().accelerometer()[i];
            imu.rpy[i] = dds_lowstate->msg_.imu_state().rpy()[i];
        }
    }

    bool is_timeout() override { return dds_lowstate->isTimeout(); }
    void update() override { dds_lowstate->update(); }

    unitree::common::UnitreeJoystick* get_joystick() override {
        return &dds_lowstate->joystick;
    }

    void enable_motors() override {}
    void disable_motors() override {}
};

class Sim2RealInterface : public VBotInterface {
public:
    std::string can_interface;
    uint8_t master_id;
    int actuator_type;
    std::vector<uint8_t> motor_ids;
    std::vector<std::unique_ptr<RobStrideMotor>> motors;
    std::vector<MotorState> motor_states;
    IMUState imu_state;

    Sim2RealInterface(const std::string &can_if, uint8_t mid,
                      const std::vector<uint8_t> &mids, int atype)
        : VBotInterface(SIM2REAL), can_interface(can_if), master_id(mid),
          actuator_type(atype), motor_ids(mids) {
        motor_states.resize(mids.size());
    }

    void init() override {
        for (auto &mid : motor_ids) {
            motors.push_back(std::make_unique<RobStrideMotor>(
                can_interface, master_id, mid, actuator_type));
        }
        spdlog::info("Sim2Real: Initialized {} motors on {}", motors.size(), can_interface);
    }

    void send_cmd(const std::vector<MotorCmd> &cmds) override {
        for (int i = 0; i < (int)cmds.size() && i < (int)motors.size(); i++) {
            motors[i]->send_motion_command(
                cmds[i].tau, cmds[i].q, cmds[i].dq, cmds[i].kp, cmds[i].kd);
        }
    }

    void get_state(std::vector<MotorState> &states, IMUState &imu) override {
        for (int i = 0; i < (int)motors.size() && i < (int)motor_states.size(); i++) {
            motor_states[i].q = motors[i]->get_position();
            motor_states[i].dq = motors[i]->get_velocity();
            motor_states[i].tau_est = motors[i]->get_torque();
        }
        states = motor_states;
    }

    bool is_timeout() override { return false; }
    void update() override {}

    unitree::common::UnitreeJoystick* get_joystick() override { return nullptr; }

    void enable_motors() override {
        for (auto &m : motors) {
            m->set_mode(move_control_mode);
            usleep(1000);
            m->enable_motor();
            usleep(1000);
        }
        spdlog::info("Sim2Real: All motors enabled.");
    }

    void disable_motors() override {
        for (auto &m : motors) {
            m->disable_motor(0);
        }
        spdlog::info("Sim2Real: All motors disabled.");
    }
};
