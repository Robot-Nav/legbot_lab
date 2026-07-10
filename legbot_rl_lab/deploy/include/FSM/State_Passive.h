// 文件用途：FSM 被动安全状态。机器人失能或出现故障时进入该状态，电机保持阻尼模式、
// 不输出主动力矩，防止意外运动。
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
        // 从配置读取阻尼系数与电机模式；位置刚度为 0，仅提供阻尼。
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
        // 每个周期将目标位置设为当前实际位置，保证电机不主动运动。
        for(int i(0); i < (int)motor_cmds.size() && i < (int)motor_states.size(); ++i)
        {
            motor_cmds[i].q = motor_states[i].q;
        }
    }
};

REGISTER_FSM(State_Passive)
