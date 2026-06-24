#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand") 
    : FSMState(state, state_string) 
    {
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());
    }

    void enter()
    {
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kp.size() && i < (int)motor_cmds.size(); ++i)
        {
            motor_cmds[i].kp = kp[i];
            motor_cmds[i].kd = kd[i];
            motor_cmds[i].dq = 0;
            motor_cmds[i].tau = 0;
        }

        std::vector<float> q0;
        for(int i(0); i < kp.size() && i < (int)motor_states.size(); ++i) {
            q0.push_back(motor_states[i].q);
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);
        
        for(int i(0); i < (int)q.size() && i < (int)motor_cmds.size(); ++i) {
            motor_cmds[i].q = q[i];
        }
    }

private:
    double t0_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
};

REGISTER_FSM(State_FixStand)
