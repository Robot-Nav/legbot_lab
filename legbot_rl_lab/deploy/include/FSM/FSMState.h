// 文件用途：FSM 具体状态的公共基类。负责从配置读取状态切换条件（手柄 DSL 表达式），
// 并在每个控制周期统一执行传感器更新、安全限幅与指令下发。
#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        // 从 config.yaml 读取当前状态向其他状态的切换条件。
        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                // 解析 DSL 表达式并编译为谓词，作为状态切换条件。
                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{
                            static unitree::common::UnitreeJoystick dummy_joy;
                            auto joy = FSMState::interface->get_joystick();
                            return func(joy ? *joy : dummy_joy);
                        },
                        fsm_id
                    )
                );
            }
        }

        // 所有状态共享的保护性切换：通信超时 → Passive。
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return FSMState::interface->is_timeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
        // 姿态越限 → Passive。
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return deploy::roll_pitch_fault(FSMState::imu_state.rpy); },
                FSMStringMap.right.at("Passive")
            )
        );
        // 电机反馈故障（温度/速度/力矩）→ Passive。
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return deploy::motor_state_fault(FSMState::motor_states); },
                FSMStringMap.right.at("Passive")
            )
        );
        // 急停请求 → Passive。
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return deploy::emergency_stop_requested(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        interface->update();
        interface->get_state(motor_states, imu_state);
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        // 发送给硬件前执行安全限幅（关节位置范围、力矩范围、变化率）。
        deploy::clamp_motor_cmds(motor_cmds);
        interface->send_cmd(motor_cmds);
    }

    static std::shared_ptr<VBotInterface> interface;
    static std::vector<MotorCmd> motor_cmds;
    static std::vector<MotorState> motor_states;
    static IMUState imu_state;
    static std::shared_ptr<Keyboard> keyboard;
};
