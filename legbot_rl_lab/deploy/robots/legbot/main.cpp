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

    const std::string network = vm["network"].as<std::string>();

    std::cout << "\n========== Legbot Controller (DDS + serial_dds_gateway) ==========\n";
    std::cout << " --- Legbot Robotics --- \n";
    std::cout << "[PHASE1] DDS network=" << network << "\n";
    std::cout << "[PHASE1] publish rt/lowcmd, subscribe rt/lowstate\n";
    std::cout << "[PHASE1] terminal 1 must run: dds_to_serial_gateway --network lo\n";

    // Load safety config (joint limits, torque limits, roll/pitch, temperature, etc.)
    deploy::load_safety_config();

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, network);
    std::cout << "[PHASE1] DDS ChannelFactory initialized on \"" << network << "\"\n";

    // Check lowcmd channel not occupied by another controller
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }

    // Single DDS interface: works for both sim2sim (MuJoCo) and sim2real (gateway)
    auto interface = std::make_shared<DDSInterface>();
    interface->init();

    FSMState::interface = interface;
    FSMState::motor_cmds.resize(12);
    FSMState::motor_states.resize(12);

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "[PHASE1] FSM started — current state: Passive\n";
    std::cout << "[PHASE1] Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "[PHASE1] And then press [Start] to start controlling the robot.\n";
    std::cout << "[PHASE1] Optional: --csv-log for 50Hz CSV diagnosis\n";
    std::cout << "===============================================================\n";
    std::cout << "Running Legbot Controller...\n";

    while (true)
    {
        sleep(1);
    }

    return 0;
}
