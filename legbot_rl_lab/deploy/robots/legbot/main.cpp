// 文件用途：Legbot 部署主程序。初始化命令行参数、安全参数、DDS 通信与 FSM，
// 支持 sim2sim（MuJoCo）与 sim2real（serial_dds_gateway）两种运行模式。
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

    // 加载安全参数（关节限位、力矩限幅、姿态/温度阈值等）。
    deploy::load_safety_config();

    // 初始化 Unitree DDS 通道工厂。
    unitree::robot::ChannelFactory::Instance()->Init(0, network);
    std::cout << "[PHASE1] DDS ChannelFactory initialized on \"" << network << "\"\n";

    // 检查 lowcmd 通道是否被其他控制器占用，避免多个进程同时发令。
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }

    // 统一 DDS 接口：发布 rt/lowcmd，订阅 rt/lowstate。
    // 连接 MuJoCo 时为 sim2sim；连接 serial_dds_gateway 时为 sim2real。
    auto interface = std::make_shared<DDSInterface>();
    interface->init();

    FSMState::interface = interface;
    FSMState::motor_cmds.resize(12);
    FSMState::motor_states.resize(12);

    // 初始化并启动 FSM。
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "[PHASE1] FSM started — current state: Passive\n";
    std::cout << "[PHASE1] Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "[PHASE1] And then press [Start] to start controlling the robot.\n";
    std::cout << "[PHASE1] Optional: --csv-log for 50Hz CSV diagnosis\n";
    std::cout << "===============================================================\n";
    std::cout << "Running Legbot Controller...\n";

    // 主线程保持存活，实际控制在 FSM 周期线程中运行。
    while (true)
    {
        sleep(1);
    }

    return 0;
}
