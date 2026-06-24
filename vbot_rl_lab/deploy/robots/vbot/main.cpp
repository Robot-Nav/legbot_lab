#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "Types.h"

std::shared_ptr<VBotInterface> FSMState::interface = nullptr;
std::vector<MotorCmd> FSMState::motor_cmds;
std::vector<MotorState> FSMState::motor_states;
IMUState FSMState::imu_state;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);

    std::string mode = vm["mode"].as<std::string>();

    std::cout << " --- VBot Controller --- \n";
    std::cout << "     Mode: " << mode << "\n";

    std::shared_ptr<VBotInterface> interface;

    if (mode == "sim2real") {
        auto sim2real_cfg = param::config["sim2real"];
        std::string can_if = vm["can"].as<std::string>();
        uint8_t master_id = sim2real_cfg["master_id"] ? sim2real_cfg["master_id"].as<uint8_t>() : 0xFF;
        auto motor_ids_yaml = sim2real_cfg["motor_ids"] ? sim2real_cfg["motor_ids"].as<std::vector<uint8_t>>() :
            std::vector<uint8_t>{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C};
        int actuator_type = sim2real_cfg["actuator_type"] ? sim2real_cfg["actuator_type"].as<int>() : 0;

        interface = std::make_shared<Sim2RealInterface>(
            can_if, master_id, motor_ids_yaml, actuator_type);
        interface->init();
        interface->enable_motors();
    } else {
        unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

        auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
        usleep(0.2 * 1e6);
        if(!lowcmd_sub->isTimeout())
        {
            spdlog::critical("The other process is using the lowcmd channel, please close it first.");
            unitree::robot::go2::shutdown();
        }

        interface = std::make_shared<Sim2SimInterface>();
        interface->init();
    }

    FSMState::interface = interface;
    FSMState::motor_cmds.resize(12);
    FSMState::motor_states.resize(12);

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "And then press [Start] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }

    return 0;
}
