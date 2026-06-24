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

        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return FSMState::interface->is_timeout(); },
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
        interface->send_cmd(motor_cmds);
    }

    static std::shared_ptr<VBotInterface> interface;
    static std::vector<MotorCmd> motor_cmds;
    static std::vector<MotorState> motor_states;
    static IMUState imu_state;
    static std::shared_ptr<Keyboard> keyboard;
};
