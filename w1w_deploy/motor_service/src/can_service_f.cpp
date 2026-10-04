#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "motor_control/motor_config.h"
#include "motor_control/socketcan_interface.h"

namespace {

constexpr std::size_t kMotorCount = 6;
constexpr std::size_t kBusCount = 2;
constexpr std::size_t kMotorsPerBus = 3;
static_assert(kBusCount * kMotorsPerBus == kMotorCount, "Invalid motor configuration");
constexpr uint32_t kCommandMagic = 0x5242434D;  // "RBCM"
constexpr uint32_t kFeedbackMagic = 0x52424642; // "RBFb"
constexpr uint16_t kProtocolVersion = 1;

struct BusLayout
{
    const char* leg_label;
    std::array<uint8_t, kMotorsPerBus> motor_ids;
};

// 点足双腿机器人: can0=左腿, can1=右腿，每腿 3 关节
// 电机 ID: 3=hip, 2=thigh, 1=calf（顺序沿用旧代码的“每总线降序 ID”）
constexpr std::array<BusLayout, kBusCount> kBusLayouts = {
    BusLayout{"left",  {0x03, 0x02, 0x01}},  // hip, thigh, calf
    BusLayout{"right", {0x03, 0x02, 0x01}},  // hip, thigh, calf
};

std::atomic_bool g_keep_running{true};

void handle_signal(int)
{
    g_keep_running.store(false);
}

#pragma pack(push, 1)
struct MotorCommandNet
{
    float torque;
    float position;
    float velocity;
    float kp;
    float kd;
};

struct CommandPacketNet
{
    uint32_t magic;
    uint16_t version;
    uint16_t motor_count;
    uint32_t sequence;
    uint32_t reserved;
    MotorCommandNet motors[kMotorCount];
};

struct MotorFeedbackNet
{
    float position;
    float velocity;
    float torque;
    float temperature;
    uint8_t error_code;
    uint8_t pattern;
    uint16_t comm_age_ms;  // 距上次收到该电机反馈的毫秒数(封顶 65535)，判断 CAN 在线
};

struct FeedbackPacketNet
{
    uint32_t magic;
    uint16_t version;
    uint16_t motor_count;
    uint32_t sequence;
    uint32_t command_sequence;
    uint32_t status_bits;
    float cycle_time_us;
    float command_age_ms;
    float rx_rate_hz;
    float tx_rate_hz;
    MotorFeedbackNet motors[kMotorCount];
};
#pragma pack(pop)

struct MotorCommand
{
    float torque = 0.0F;
    float position = 0.0F;
    float velocity = 0.0F;
    float kp = 0.0F;
    float kd = 5.0F;
};

struct CommandState
{
    std::array<MotorCommand, kMotorCount> motors{};
    uint32_t sequence = 0;
    std::chrono::steady_clock::time_point timestamp{};
    bool valid = false;
};

struct FeedbackState
{
    std::array<MotorFeedbackNet, kMotorCount> motors{};
};

struct BusWorker
{
    std::string interface_name;
    std::unique_ptr<SocketCanInterface> iface;
    std::unique_ptr<MotorControlSet> controller;
    std::thread rx_thread;
    std::atomic_bool running{false};
    std::atomic_uint64_t rx_count{0};

    ~BusWorker()
    {
        stop();
    }

    void start()
    {
        if (!iface || !controller)
        {
            throw std::runtime_error("BusWorker missing interface or controller");
        }
        running.store(true);
        rx_thread = std::thread([this]() { rx_loop(); });
    }

    void stop()
    {
        running.store(false);
        if (rx_thread.joinable())
        {
            rx_thread.join();
        }
    }

private:
    void rx_loop()
    {
        CanFrame frame{};
        while (running.load() && g_keep_running.load())
        {
            try
            {
                if (iface->receive(frame, std::chrono::milliseconds(5)))
                {
                    controller->handleIncomingFrame(frame);
                    ++rx_count;
                }
            }
            catch (const std::exception& ex)
            {
                std::cerr << "❌ [CAN RX Error] Interface: " << interface_name << " - " << ex.what() << std::endl;
                break;
            }
        }
    }
};

struct BusSlot
{
    MotorControlSet* controller = nullptr;
    std::size_t feedback_index = 0;
    uint8_t motor_id = 0;
    std::size_t bus_index = 0;  // 0=left leg (can0), 1=right leg (can1)
};

bool set_non_blocking(int fd)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
    {
        return false;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        return false;
    }
    return true;
}

float clamp_value(float value, float min_v, float max_v)
{
    return std::clamp(value, min_v, max_v);
}

uint64_t steady_now_ms()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Get position offset in radians based on leg (bus) and motor type.
// 偏移量按电机侧角度定义；calf 标定值来自关节侧，需要乘减速比换算。
//   hip(id3)=25°, thigh(id2)=162°, calf(id1)=165° * 1.4615 = 241.1475°
//   右腿(bus_index=1)大腿机械镜像，offset 取反号
constexpr float kDeg2Rad = 0.017453292519943295f;  // pi/180
constexpr float kHipOffsetDeg = 25.0f;
constexpr float kThighOffsetDeg = 162.0f;
constexpr float kCalfReductionRatio = 1.4615f;
constexpr float kCalfOffsetDeg = 165.0f * kCalfReductionRatio;

bool is_calf_motor(uint8_t motor_id)
{
    return motor_id == 0x01;
}

float get_calf_joint_direction()
{
    return -1.0f;
}

float get_position_offset(std::size_t bus_index, uint8_t motor_id)
{
    switch (motor_id)
    {
        case 0x03:  // hip 髋关节: 右腿取反
            return (bus_index == 1 ? -kHipOffsetDeg : kHipOffsetDeg) * kDeg2Rad;
        case 0x02:                            // thigh 大腿: 右腿镜像取反
            return (bus_index == 1 ? -kThighOffsetDeg : kThighOffsetDeg) * kDeg2Rad;
        case 0x01: return (bus_index == 1 ? -kCalfOffsetDeg : kCalfOffsetDeg) * kDeg2Rad;  // calf 小腿: 右腿取反
        default:   return 0.0f;
    }
}

void apply_motor_command(const BusSlot& slot, const MotorCommand& cmd)
{
    auto& motor = slot.controller->getMotor(slot.motor_id);
    const MotorLimits& limits = getMotorLimits(slot.motor_id);
    const float offset = get_position_offset(slot.bus_index, slot.motor_id);
    float torque = cmd.torque;
    float position = cmd.position + offset;
    float velocity = cmd.velocity;
    float kp = cmd.kp;
    float kd = cmd.kd;

    if (is_calf_motor(slot.motor_id))
    {
        const float direction = get_calf_joint_direction();
        const float ratio_sq = kCalfReductionRatio * kCalfReductionRatio;
        torque = direction * cmd.torque / kCalfReductionRatio;
        position = offset + direction * kCalfReductionRatio * cmd.position;
        velocity = direction * kCalfReductionRatio * cmd.velocity;
        kp = cmd.kp / ratio_sq;
        kd = cmd.kd / ratio_sq;
    }

    torque = clamp_value(torque, limits.T_MIN, limits.T_MAX);
    position = clamp_value(position, limits.P_MIN, limits.P_MAX);
    velocity = clamp_value(velocity, limits.V_MIN, limits.V_MAX);
    kp = clamp_value(kp, limits.KP_MIN, limits.KP_MAX);
    kd = clamp_value(kd, limits.KD_MIN, limits.KD_MAX);
    motor.RobStrite_Motor_move_control(torque, position, velocity, kp, kd);
}

void disable_motors(const std::array<BusSlot, kMotorCount>& slots)
{
    for (const auto& slot : slots)
    {
        try
        {
            slot.controller->getMotor(slot.motor_id).Disenable_Motor(0);
        }
        catch (const std::exception&)
        {
        }
    }
}

FeedbackState gather_feedback(const std::array<BusSlot, kMotorCount>& slots)
{
    FeedbackState state{};
    for (std::size_t idx = 0; idx < slots.size(); ++idx)
    {
        const BusSlot& slot = slots[idx];
        const auto& feedback = slot.controller->getFeedback();
        const auto index = slot.feedback_index;
        const float offset = get_position_offset(slot.bus_index, slot.motor_id);
        if (index < feedback.pos.size())
        {
            float position = feedback.pos[index] - offset;
            if (is_calf_motor(slot.motor_id))
            {
                position = get_calf_joint_direction() * position / kCalfReductionRatio;
            }
            state.motors[idx].position = position;
        }
        if (index < feedback.vel.size())
        {
            float velocity = feedback.vel[index];
            if (is_calf_motor(slot.motor_id))
            {
                velocity = get_calf_joint_direction() * velocity / kCalfReductionRatio;
            }
            state.motors[idx].velocity = velocity;
        }
        if (index < feedback.tor.size())
        {
            float torque = feedback.tor[index];
            if (is_calf_motor(slot.motor_id))
            {
                torque = get_calf_joint_direction() * kCalfReductionRatio * torque;
            }
            state.motors[idx].torque = torque;
        }
        if (index < feedback.temp.size())
        {
            state.motors[idx].temperature = feedback.temp[index];
        }
        if (index < feedback.error_code.size())
        {
            state.motors[idx].error_code = feedback.error_code[index];
        }
        if (index < feedback.pattern.size())
        {
            state.motors[idx].pattern = static_cast<uint8_t>(feedback.pattern[index]);
        }
        // 计算通信 age：从未收到(last_rx_ms==0)则标为 65535(离线)
        if (index < feedback.last_rx_ms.size())
        {
            const uint64_t last = feedback.last_rx_ms[index];
            uint64_t age = 65535;
            if (last != 0)
            {
                const uint64_t now = steady_now_ms();
                age = (now > last) ? (now - last) : 0;
                if (age > 65535) age = 65535;
            }
            state.motors[idx].comm_age_ms = static_cast<uint16_t>(age);
        }
        else
        {
            state.motors[idx].comm_age_ms = 65535;
        }
    }
    return state;
}

} // namespace

int main(int argc, char** argv)
{
    std::array<std::string, kBusCount> can_names = {"can0", "can1"};
    uint16_t udp_port = 55100;
    int cycle_us = 1000;  // ~1 kHz (1000000/1000 = 1000)
    std::chrono::milliseconds command_timeout{50};

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--can0" && i + 1 < argc)
        {
            can_names[0] = argv[++i];
        }
        else if (arg == "--can1" && i + 1 < argc)
        {
            can_names[1] = argv[++i];
        }
        else if (arg == "--port" && i + 1 < argc)
        {
            udp_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--cycle-us" && i + 1 < argc)
        {
            cycle_us = std::stoi(argv[++i]);
        }
        else if (arg == "--timeout-ms" && i + 1 < argc)
        {
            command_timeout = std::chrono::milliseconds(std::stoi(argv[++i]));
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--can0 <iface>] [--can1 <iface>]"
                      << " [--port <udp port>]"
                      << " [--cycle-us <microseconds>] [--timeout-ms <ms>]" << std::endl;
            return 0;
        }
    }

    if (cycle_us <= 0)
    {
        std::cerr << "cycle-us must be > 0" << std::endl;
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try
    {
        std::array<BusWorker, kBusCount> bus_workers{};
        for (std::size_t bus = 0; bus < kBusCount; ++bus)
        {
            bus_workers[bus].interface_name = can_names[bus];
            bus_workers[bus].iface = std::make_unique<SocketCanInterface>(can_names[bus]);
        }

        for (std::size_t bus = 0; bus < kBusCount; ++bus)
        {
            const auto& ids = kBusLayouts[bus].motor_ids;
            const std::string if_name = bus_workers[bus].interface_name;
            bus_workers[bus].controller = std::make_unique<MotorControlSet>(
                [iface = bus_workers[bus].iface.get(), if_name, bus](const CanFrame& frame) {
                    try {
                        iface->send(frame);
                    } catch (const std::exception& ex) {
                        const char* leg_names[] = {"left", "right"};
                        throw std::runtime_error(std::string("❌ [CAN TX Error] Bus: ") + if_name + 
                                               " (" + leg_names[bus] + ") - " + ex.what());
                    }
                },
                std::vector<uint8_t>(ids.begin(), ids.end()));
        }

        std::array<BusSlot, kMotorCount> slots{};
        std::size_t slot_index = 0;
        for (std::size_t bus = 0; bus < kBusCount; ++bus)
        {
            for (std::size_t idx = 0; idx < kMotorsPerBus; ++idx)
            {
                slots[slot_index].controller = bus_workers[bus].controller.get();
                slots[slot_index].feedback_index = idx;
                slots[slot_index].motor_id = kBusLayouts[bus].motor_ids[idx];
                slots[slot_index].bus_index = bus;
                ++slot_index;
            }
        }

        for (auto& worker : bus_workers)
        {
            worker.start();
        }

        // Initialize motors (following master branch approach)
        std::cout << "Initializing motors..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Enable all motors one by one
        std::cout << "Enabling motors..." << std::endl;
        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            slots[i].controller->getMotor(slots[i].motor_id).Enable_Motor();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 增加延迟确保接收
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Start sending damping commands immediately to keep motors enabled
        std::cout << "Starting control loop to maintain motor states..." << std::endl;
        MotorCommand damping_cmd{};  // torque=0, pos=0, vel=0, kp=0, kd=5.0
        
        // Send damping commands for 1 second to ensure all motors are stable
        auto warmup_start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - warmup_start < std::chrono::seconds(1))
        {
            for (std::size_t i = 0; i < slots.size(); ++i)
            {
                apply_motor_command(slots[i], damping_cmd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));  // ~500Hz
        }
        
        std::cout << "✅ Motors enabled and stable. Ready for commands." << std::endl;

        const int udp_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_socket < 0)
        {
            throw std::runtime_error("Failed to open UDP socket");
        }

        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_addr.s_addr = INADDR_ANY;
        bind_addr.sin_port = htons(udp_port);

        int reuse = 1;
        setsockopt(udp_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        if (bind(udp_socket, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0)
        {
            ::close(udp_socket);
            throw std::runtime_error("Failed to bind UDP port");
        }

        if (!set_non_blocking(udp_socket))
        {
            ::close(udp_socket);
            throw std::runtime_error("Failed to set UDP socket non-blocking");
        }

        sockaddr_in client_addr{};
    bool has_client = false;
    uint32_t feedback_sequence = 0;
    uint32_t last_command_sequence = 0;
        std::array<std::uint64_t, kBusCount> bus_rx_last{};
        std::uint64_t applied_count = 0;
        std::uint64_t applied_last = 0;

        CommandState latest_command{};
        auto last_rate_update = std::chrono::steady_clock::now();
    float rx_rate_value = 0.0F;
    float tx_rate_value = 0.0F;

        std::chrono::steady_clock::time_point next_tick = std::chrono::steady_clock::now();
        const auto cycle_period = std::chrono::microseconds(cycle_us);

        while (g_keep_running.load())
        {
            const auto loop_start = std::chrono::steady_clock::now();

            while (true)
            {
                CommandPacketNet packet{};
                sockaddr_in from_addr{};
                socklen_t from_len = sizeof(from_addr);
                const ssize_t bytes = recvfrom(udp_socket, &packet, sizeof(packet), 0,
                                               reinterpret_cast<sockaddr*>(&from_addr), &from_len);
                if (bytes < 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK)
                    {
                        break;
                    }
                    continue;
                }
                if (static_cast<std::size_t>(bytes) < sizeof(CommandPacketNet))
                {
                    continue;
                }
                if (packet.magic != kCommandMagic || packet.version != kProtocolVersion || packet.motor_count != kMotorCount)
                {
                    continue;
                }

                has_client = true;
                client_addr = from_addr;

                latest_command.valid = true;
                latest_command.sequence = packet.sequence;
                latest_command.timestamp = loop_start;
                for (std::size_t i = 0; i < kMotorCount; ++i)
                {
                    latest_command.motors[i].torque = packet.motors[i].torque;
                    latest_command.motors[i].position = packet.motors[i].position;
                    latest_command.motors[i].velocity = packet.motors[i].velocity;
                    latest_command.motors[i].kp = packet.motors[i].kp;
                    latest_command.motors[i].kd = packet.motors[i].kd;
                }
            }

            auto effective_command = latest_command;
            const auto now = std::chrono::steady_clock::now();
            const bool stale_command = !effective_command.valid || (now - effective_command.timestamp) > command_timeout;
            if (stale_command)
            {
                effective_command.motors.fill(MotorCommand{});
            }
            else
            {
                last_command_sequence = effective_command.sequence;
            }

            for (std::size_t i = 0; i < kMotorCount; ++i)
            {
                apply_motor_command(slots[i], effective_command.motors[i]);
            }
            ++applied_count;

            FeedbackState feedback_values = gather_feedback(slots);

            const auto now_after = std::chrono::steady_clock::now();
            if (now_after - last_rate_update >= std::chrono::seconds(1))
            {
                const double period = std::chrono::duration<double>(now_after - last_rate_update).count();
                double total_bus_rate = 0.0;
                for (std::size_t bus = 0; bus < kBusCount; ++bus)
                {
                    const auto current = bus_workers[bus].rx_count.load();
                    const double rate = (current - bus_rx_last[bus]) / period;
                    bus_rx_last[bus] = current;
                    total_bus_rate += rate;
                }
                const double apply_rate = (applied_count - applied_last) / period;

                applied_last = applied_count;
                last_rate_update = now_after;

                rx_rate_value = static_cast<float>(total_bus_rate);
                tx_rate_value = static_cast<float>(apply_rate);

                std::cout << "[can_service] ctrl_freq=" << apply_rate
                          << " Hz, rx_total=" << total_bus_rate << " Hz" << std::endl;
            }

            if (has_client)
            {
                FeedbackPacketNet feedback{};
                feedback.magic = kFeedbackMagic;
                feedback.version = kProtocolVersion;
                feedback.motor_count = kMotorCount;
                feedback.sequence = feedback_sequence++;
                feedback.command_sequence = last_command_sequence;
                feedback.status_bits = stale_command ? 0x1U : 0U;
                feedback.cycle_time_us = static_cast<float>(cycle_us);
                const auto age_duration = effective_command.valid
                                              ? (now_after - effective_command.timestamp)
                                              : command_timeout;
                feedback.command_age_ms = static_cast<float>(
                    std::chrono::duration<double, std::milli>(age_duration).count());
                feedback.rx_rate_hz = rx_rate_value;
                feedback.tx_rate_hz = tx_rate_value;
                std::memcpy(feedback.motors, feedback_values.motors.data(), sizeof(feedback.motors));
                sendto(udp_socket, &feedback, sizeof(feedback), 0,
                       reinterpret_cast<sockaddr*>(&client_addr), sizeof(client_addr));
            }

            next_tick += cycle_period;
            std::this_thread::sleep_until(next_tick);
        }

        disable_motors(slots);
        for (auto& worker : bus_workers)
        {
            worker.stop();
        }
        ::close(udp_socket);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
