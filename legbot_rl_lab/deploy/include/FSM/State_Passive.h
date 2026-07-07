#pragma once

#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive") 
    : FSMState(state, state_string) 
    {
    } 

    void enter()
    {
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        static auto mode = param::config["FSM"]["Passive"]["mode"]
                               ? param::config["FSM"]["Passive"]["mode"].as<std::vector<int>>()
                               : std::vector<int>{};
        for(int i(0); i < kd.size() && i < (int)motor_cmds.size(); ++i)
        {
            motor_cmds[i].mode = (i < (int)mode.size()) ? (uint8_t)mode[i] : (uint8_t)1;
            motor_cmds[i].kp = 0;
            motor_cmds[i].kd = kd[i];
            motor_cmds[i].dq = 0;
            motor_cmds[i].tau = 0;
        }
    }

    void run()
    {
        for(int i(0); i < (int)motor_cmds.size() && i < (int)motor_states.size(); ++i)
        {
            motor_cmds[i].q = motor_states[i].q;
        }
    }
};

REGISTER_FSM(State_Passive)
