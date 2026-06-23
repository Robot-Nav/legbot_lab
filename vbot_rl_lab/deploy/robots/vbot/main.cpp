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

        uint8_t master_id = sim2real_cfg["master_id"] ? (uint8_t)sim2real_cfg["master_id"].as<int>() : 0xFF;
        int actuator_type = sim2real_cfg["actuator_type"] ? sim2real_cfg["actuator_type"].as<int>() : 2;

        std::vector<MotorSerialConfig> motor_serial_configs;
        if (sim2real_cfg["motor_serials"]) {
            auto motor_serials = sim2real_cfg["motor_serials"];
            for (const auto &ms : motor_serials) {
                MotorSerialConfig cfg;
                cfg.serial_port = ms["serial_port"].as<std::string>();
                cfg.channel = ms["channel"] ? (uint8_t)ms["channel"].as<int>() : 0x00;
                cfg.baudrate = ms["baudrate"] ? ms["baudrate"].as<int>() : 921600;
                cfg.motor_ids = ms["motor_ids"].as<std::vector<uint8_t>>();
                motor_serial_configs.push_back(cfg);
            }
        }

        IMUConfig imu_config;
        if (sim2real_cfg["imu"]) {
            imu_config.serial_port = sim2real_cfg["imu"]["serial_port"].as<std::string>();
            imu_config.baudrate = sim2real_cfg["imu"]["baudrate"] ? sim2real_cfg["imu"]["baudrate"].as<int>() : 921600;
        }

        std::vector<float> zero_offsets(12, 0.0f);
        if (sim2real_cfg["zero_offsets"]) {
            zero_offsets = sim2real_cfg["zero_offsets"].as<std::vector<float>>();
        }

        interface = std::make_shared<Sim2RealInterface>(
            motor_serial_configs, imu_config, master_id, actuator_type, zero_offsets);
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
