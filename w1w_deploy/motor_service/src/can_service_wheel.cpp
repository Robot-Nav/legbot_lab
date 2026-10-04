#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <memory>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "motor_control/motor_config.h"
#include "motor_control/socketcan_interface.h"
#include "motor_control/w190_canfd.h"
#include "w1w/device_guard.h"

namespace {

constexpr std::size_t kLegCount = 4;
constexpr std::size_t kJointsPerLeg = 3;
constexpr std::size_t kMotorCount = 16;
constexpr std::size_t kWheelBusCount = 2;
constexpr std::array<uint8_t, kJointsPerLeg> kJointIds = {4, 3, 2};
constexpr uint32_t kCommandMagic = 0x5242434D;
constexpr uint32_t kFeedbackMagic = 0x52424642;
constexpr uint16_t kProtocolVersion = 1;
constexpr uint16_t kProtocolVersionV2 = 2;
constexpr float kDegToRad = 0.017453292519943295F;
constexpr float kHipOffsetDeg = 25.0F;
constexpr float kThighOffsetDeg = 162.0F;
constexpr float kCalfReductionRatio = 1.4615F;
constexpr float kCalfOffsetDeg = 165.0F * kCalfReductionRatio;

std::atomic_bool g_running{true};

void handle_signal(int)
{
    g_running.store(false);
}

uint64_t steady_now_ms()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint16_t communication_age(uint64_t last_rx_ms)
{
    if (last_rx_ms == 0)
    {
        return 65535;
    }
    const uint64_t now = steady_now_ms();
    return static_cast<uint16_t>(std::min<uint64_t>(now > last_rx_ms ? now - last_rx_ms : 0, 65535));
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
    uint16_t comm_age_ms;
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

struct MotorFeedbackNetV2
{
    float position;
    float velocity;
    float torque;
    float temperature;
    float voltage;
    uint8_t error_code;
    uint8_t pattern;
    uint16_t comm_age_ms;
};

struct FeedbackPacketNetV2
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
    MotorFeedbackNetV2 motors[kMotorCount];
};
#pragma pack(pop)

static_assert(sizeof(CommandPacketNet) == 336, "Unexpected command packet layout");
static_assert(sizeof(FeedbackPacketNet) == 356, "Unexpected feedback packet layout");
static_assert(sizeof(FeedbackPacketNetV2) == 420, "Unexpected V2 feedback packet layout");

struct MotorCommand
{
    float torque = 0.0F;
    float position = 0.0F;
    float velocity = 0.0F;
    float kp = 0.0F;
    float kd = 0.0F;
};

using CommandArray = std::array<MotorCommand, kMotorCount>;

enum class SafetyState
{
    WaitingReady,
    Ready,
    Running,
    Latched,
};

constexpr uint32_t kStatusCommandTimeout = 1U << 0;
constexpr uint32_t kStatusSafetyLatched = 1U << 17;
constexpr uint32_t kStatusWaitingReady = 1U << 18;
constexpr uint32_t kStatusReady = 1U << 19;
constexpr uint32_t kStatusRunning = 1U << 20;
constexpr uint32_t kStatusLatchTimeout = 1U << 21;
constexpr uint32_t kStatusLatchMotorOffline = 1U << 22;
constexpr uint32_t kStatusReleaseSupported = 1U << 23;
constexpr uint32_t kCommandFlagReleaseControl = 1U << 0;
constexpr uint32_t kKnownCommandFlags = kCommandFlagReleaseControl;

CommandArray safe_commands()
{
    CommandArray commands{};
    for (std::size_t leg = 0; leg < kLegCount; ++leg)
    {
        for (std::size_t joint = 0; joint < kJointsPerLeg; ++joint)
        {
            commands[leg * 4 + joint].kd = 2.0F;
        }
        commands[leg * 4 + 3].kd = 1.0F;
    }
    return commands;
}

bool is_safe_command(const CommandArray& commands)
{
    constexpr float epsilon = 1.0e-5F;
    for (std::size_t index = 0; index < commands.size(); ++index)
    {
        const auto& command = commands[index];
        const float expected_kd = ((index + 1) % 4 == 0) ? 1.0F : 2.0F;
        if (std::fabs(command.torque) > epsilon ||
            std::fabs(command.position) > epsilon ||
            std::fabs(command.velocity) > epsilon ||
            std::fabs(command.kp) > epsilon ||
            std::fabs(command.kd - expected_kd) > epsilon)
        {
            return false;
        }
    }
    return true;
}

struct CommandState
{
    CommandArray motors = safe_commands();
    uint32_t sequence = 0;
    std::chrono::steady_clock::time_point timestamp{};
    bool valid = false;
};

struct ClientPeer
{
    sockaddr_in address{};
    std::chrono::steady_clock::time_point last_seen{};
    uint16_t protocol_version = kProtocolVersion;
};

bool same_client(const sockaddr_in& left, const sockaddr_in& right)
{
    return left.sin_family == right.sin_family &&
           left.sin_addr.s_addr == right.sin_addr.s_addr &&
           left.sin_port == right.sin_port;
}

void remember_peer(std::vector<ClientPeer>& peers,
                   const sockaddr_in& address,
                   uint16_t protocol_version,
                   std::chrono::steady_clock::time_point now)
{
    for (auto& peer : peers)
    {
        if (same_client(peer.address, address))
        {
            peer.last_seen = now;
            peer.protocol_version = protocol_version;
            return;
        }
    }
    if (peers.size() >= 8)
    {
        auto oldest = std::min_element(peers.begin(), peers.end(),
            [](const ClientPeer& left, const ClientPeer& right) {
                return left.last_seen < right.last_seen;
            });
        *oldest = ClientPeer{address, now, protocol_version};
        return;
    }
    peers.push_back(ClientPeer{address, now, protocol_version});
}

float position_offset(std::size_t leg_index, uint8_t motor_id)
{
    const bool right = (leg_index % 2) != 0;
    const bool front = leg_index < 2;
    if (motor_id == 4)
    {
        const float sign = front ? (right ? -1.0F : 1.0F) : (right ? 1.0F : -1.0F);
        return sign * kHipOffsetDeg * kDegToRad;
    }
    if (motor_id == 3)
    {
        return (right ? -1.0F : 1.0F) * kThighOffsetDeg * kDegToRad;
    }
    if (motor_id == 2)
    {
        return (right ? -1.0F : 1.0F) * kCalfOffsetDeg * kDegToRad;
    }
    return 0.0F;
}

class LegWorker
{
public:
    LegWorker(std::string interface_name, std::size_t leg_index)
        : interface_name_(std::move(interface_name)), leg_index_(leg_index),
          iface_(std::make_unique<SocketCanInterface>(interface_name_))
    {
        controller_ = std::make_unique<MotorControlSet>(
            [this](const CanFrame& frame) { iface_->send(frame); },
            std::vector<uint8_t>(kJointIds.begin(), kJointIds.end()));
    }

    ~LegWorker()
    {
        stop();
        disable();
    }

    void start()
    {
        running_.store(true);
        rx_thread_ = std::thread([this]() { receive_loop(); });
    }

    void stop()
    {
        running_.store(false);
        if (rx_thread_.joinable())
        {
            rx_thread_.join();
        }
    }

    void enable()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint8_t id : kJointIds)
        {
            controller_->getMotor(id).Enable_Motor();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    void disable() noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (uint8_t id : kJointIds)
            {
                controller_->getMotor(id).Disenable_Motor(0);
            }
        }
        catch (...)
        {
        }
    }

    void send(const std::array<MotorCommand, 3>& commands) noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (std::size_t joint = 0; joint < kJointsPerLeg; ++joint)
            {
                const uint8_t id = kJointIds[joint];
                const MotorCommand& input = commands[joint];
                const MotorLimits& limits = getMotorLimits(id);
                float torque = input.torque;
                float position = input.position + position_offset(leg_index_, id);
                float velocity = input.velocity;
                float kp = input.kp;
                float kd = input.kd;
                if (id == 2)
                {
                    constexpr float direction = -1.0F;
                    const float ratio_squared = kCalfReductionRatio * kCalfReductionRatio;
                    torque = direction * input.torque / kCalfReductionRatio;
                    position = position_offset(leg_index_, id) +
                               direction * kCalfReductionRatio * input.position;
                    velocity = direction * kCalfReductionRatio * input.velocity;
                    kp = input.kp / ratio_squared;
                    kd = input.kd / ratio_squared;
                }
                controller_->getMotor(id).RobStrite_Motor_move_control(
                    std::clamp(torque, limits.T_MIN, limits.T_MAX),
                    std::clamp(position, limits.P_MIN, limits.P_MAX),
                    std::clamp(velocity, limits.V_MIN, limits.V_MAX),
                    std::clamp(kp, limits.KP_MIN, limits.KP_MAX),
                    std::clamp(kd, limits.KD_MIN, limits.KD_MAX));
            }
            tx_count_.fetch_add(1);
        }
        catch (...)
        {
            tx_errors_.fetch_add(1);
        }
    }

    std::array<MotorFeedbackNet, 3> feedback()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::array<MotorFeedbackNet, 3> result{};
        const auto& source = controller_->getFeedback();
        for (std::size_t joint = 0; joint < kJointsPerLeg; ++joint)
        {
            const uint8_t id = kJointIds[joint];
            float position = source.pos[joint] - position_offset(leg_index_, id);
            float velocity = source.vel[joint];
            float torque = source.tor[joint];
            if (id == 2)
            {
                position = -position / kCalfReductionRatio;
                velocity = -velocity / kCalfReductionRatio;
                torque = -kCalfReductionRatio * torque;
            }
            result[joint].position = position;
            result[joint].velocity = velocity;
            result[joint].torque = torque;
            result[joint].temperature = source.temp[joint];
            result[joint].error_code = source.error_code[joint];
            result[joint].pattern = static_cast<uint8_t>(source.pattern[joint]);
            result[joint].comm_age_ms = communication_age(source.last_rx_ms[joint]);
        }
        return result;
    }

    uint64_t rx_count() const { return rx_count_.load(); }
    uint64_t tx_count() const { return tx_count_.load(); }
    uint64_t error_count() const { return rx_errors_.load() + tx_errors_.load(); }

private:
    void receive_loop()
    {
        while (running_.load() && g_running.load())
        {
            try
            {
                CanFrame frame{};
                if (iface_->receive(frame, std::chrono::milliseconds(5)))
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    controller_->handleIncomingFrame(frame);
                    rx_count_.fetch_add(1);
                }
            }
            catch (const std::exception& error)
            {
                rx_errors_.fetch_add(1);
                std::cerr << "[" << interface_name_ << "] RX stopped: " << error.what() << std::endl;
                break;
            }
        }
    }

    std::string interface_name_;
    std::size_t leg_index_;
    std::unique_ptr<SocketCanInterface> iface_;
    std::unique_ptr<MotorControlSet> controller_;
    std::mutex mutex_;
    std::thread rx_thread_;
    std::atomic_bool running_{false};
    std::atomic_uint64_t rx_count_{0};
    std::atomic_uint64_t tx_count_{0};
    std::atomic_uint64_t rx_errors_{0};
    std::atomic_uint64_t tx_errors_{0};
};

class WheelWorker
{
public:
    explicit WheelWorker(std::string interface_name)
        : interface_name_(std::move(interface_name)), bus_(interface_name_)
    {
    }

    ~WheelWorker()
    {
        stop();
        disable();
    }

    void start()
    {
        running_.store(true);
        rx_thread_ = std::thread([this]() { receive_loop(); });
    }

    void stop()
    {
        running_.store(false);
        if (rx_thread_.joinable())
        {
            rx_thread_.join();
        }
    }

    void enable() noexcept
    {
        try { bus_.enable(); } catch (...) { tx_errors_.fetch_add(1); }
    }

    void disable() noexcept
    {
        try { bus_.disable(); } catch (...) { tx_errors_.fetch_add(1); }
    }

    void send(const std::array<MotorCommand, 2>& input) noexcept
    {
        try
        {
            std::array<W190Command, 2> commands{};
            for (std::size_t i = 0; i < commands.size(); ++i)
            {
                commands[i].torque = input[i].torque;
                commands[i].position = input[i].position;
                commands[i].velocity = input[i].velocity;
                commands[i].kp = input[i].kp;
                commands[i].kd = input[i].kd;
            }
            bus_.sendMotion(commands);
            tx_count_.fetch_add(1);
        }
        catch (...)
        {
            tx_errors_.fetch_add(1);
        }
    }

    std::array<MotorFeedbackNet, 2> feedback()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::array<MotorFeedbackNet, 2> result{};
        for (std::size_t i = 0; i < feedback_.size(); ++i)
        {
            result[i].position = feedback_[i].position;
            result[i].velocity = feedback_[i].velocity;
            result[i].torque = feedback_[i].torque;
            result[i].temperature = feedback_[i].temperature;
            // Preserve the W190 V1.2 status byte in pattern. Consumers decode
            // bit0 as enable and bits1..6 as the documented fault flags.
            result[i].error_code = 0;
            result[i].pattern = feedback_[i].status;
            result[i].comm_age_ms = communication_age(feedback_[i].last_rx_ms);
        }
        return result;
    }

    std::array<float, 2> voltages()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return {feedback_[0].voltage, feedback_[1].voltage};
    }

    uint64_t rx_count() const { return rx_count_.load(); }
    uint64_t tx_count() const { return tx_count_.load(); }
    uint64_t error_count() const { return rx_errors_.load() + tx_errors_.load(); }

private:
    void receive_loop()
    {
        while (running_.load() && g_running.load())
        {
            try
            {
                W190Feedback value{};
                if (bus_.receive(value, std::chrono::milliseconds(5)))
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    feedback_[value.motor_id - 1] = value;
                    rx_count_.fetch_add(1);
                }
            }
            catch (const std::exception& error)
            {
                rx_errors_.fetch_add(1);
                std::cerr << "[" << interface_name_ << "] RX stopped: " << error.what() << std::endl;
                break;
            }
        }
    }

    std::string interface_name_;
    W190CanFd bus_;
    std::array<W190Feedback, 2> feedback_{};
    std::mutex mutex_;
    std::thread rx_thread_;
    std::atomic_bool running_{false};
    std::atomic_uint64_t rx_count_{0};
    std::atomic_uint64_t tx_count_{0};
    std::atomic_uint64_t rx_errors_{0};
    std::atomic_uint64_t tx_errors_{0};
};

bool set_non_blocking(int fd)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) >= 0;
}

bool valid_command(const MotorCommandNet& command)
{
    return std::isfinite(command.torque) && std::isfinite(command.position) &&
           std::isfinite(command.velocity) && std::isfinite(command.kp) &&
           std::isfinite(command.kd);
}

} // namespace

int main(int argc, char** argv)
{
    if (!w1w::device_guard::enforce("can_service_wheel"))
    {
        return 77;
    }

    std::array<std::string, 6> can_names = {"can0", "can1", "can2", "can3", "can4", "can5"};
    uint16_t udp_port = 55100;
    std::string bind_host = "127.0.0.1";
    int cycle_us = 1429;
    int wheel_cycle_us = 1429;
    std::chrono::milliseconds command_timeout{100};

    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument.rfind("--can", 0) == 0 && argument.size() == 6 && i + 1 < argc)
        {
            const int index = argument[5] - '0';
            if (index >= 0 && index < 6) can_names[static_cast<std::size_t>(index)] = argv[++i];
        }
        else if (argument == "--port" && i + 1 < argc)
        {
            udp_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        }
        else if (argument == "--bind-host" && i + 1 < argc)
        {
            bind_host = argv[++i];
        }
        else if (argument == "--cycle-us" && i + 1 < argc)
        {
            cycle_us = std::stoi(argv[++i]);
        }
        else if (argument == "--wheel-cycle-us" && i + 1 < argc)
        {
            wheel_cycle_us = std::stoi(argv[++i]);
        }
        else if (argument == "--timeout-ms" && i + 1 < argc)
        {
            command_timeout = std::chrono::milliseconds(std::stoi(argv[++i]));
        }
        else if (argument == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--can0 name] ... [--can5 name] [--bind-host 127.0.0.1] [--port 55100]"
                      << " [--cycle-us 1429] [--wheel-cycle-us 1429]"
                      << " [--timeout-ms 100]" << std::endl;
            return 0;
        }
    }

    if (cycle_us <= 0 || wheel_cycle_us <= 0 || command_timeout.count() <= 0)
    {
        std::cerr << "cycle-us, wheel-cycle-us and timeout-ms must be positive" << std::endl;
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try
    {
        std::array<std::unique_ptr<LegWorker>, kLegCount> legs{};
        for (std::size_t i = 0; i < legs.size(); ++i)
        {
            legs[i] = std::make_unique<LegWorker>(can_names[i], i);
            legs[i]->start();
        }
        std::array<std::unique_ptr<WheelWorker>, kWheelBusCount> wheels{};
        wheels[0] = std::make_unique<WheelWorker>(can_names[4]);
        wheels[1] = std::make_unique<WheelWorker>(can_names[5]);
        for (auto& wheel : wheels) wheel->start();

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        for (auto& leg : legs) leg->enable();
        for (auto& wheel : wheels) wheel->enable();
        const CommandArray startup_damping = safe_commands();
        for (std::size_t leg = 0; leg < kLegCount; ++leg)
        {
            std::array<MotorCommand, 3> commands{};
            std::copy_n(startup_damping.begin() + leg * 4, 3, commands.begin());
            legs[leg]->send(commands);
        }
        wheels[0]->send({startup_damping[3], startup_damping[7]});
        wheels[1]->send({startup_damping[11], startup_damping[15]});

        const int udp_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_socket < 0) throw std::runtime_error("Failed to create UDP socket");
        sockaddr_in bind_address{};
        bind_address.sin_family = AF_INET;
        if (inet_pton(AF_INET, bind_host.c_str(), &bind_address.sin_addr) != 1)
        {
            ::close(udp_socket);
            throw std::runtime_error("Invalid IPv4 bind address: " + bind_host);
        }
        bind_address.sin_port = htons(udp_port);
        if (bind(udp_socket, reinterpret_cast<sockaddr*>(&bind_address), sizeof(bind_address)) < 0)
        {
            ::close(udp_socket);
            throw std::runtime_error(std::string("Failed to bind UDP socket: ") + std::strerror(errno));
        }
        if (!set_non_blocking(udp_socket))
        {
            ::close(udp_socket);
            throw std::runtime_error("Failed to make UDP socket non-blocking");
        }

        std::cout << "can_service_wheel ready: 16 motors, UDP " << bind_host << ":" << udp_port
                  << ", cycle " << cycle_us << " us, watchdog "
                  << command_timeout.count() << " ms, W190 cycle "
                  << wheel_cycle_us << " us" << std::endl;

        CommandState latest{};
        std::vector<ClientPeer> feedback_peers;
        sockaddr_in active_controller_address{};
        bool has_active_controller = false;
        uint32_t feedback_sequence = 0;
        uint32_t last_command_sequence = 0;
        auto next_tick = std::chrono::steady_clock::now();
        auto next_wheel_tick = next_tick;
        auto last_rate_time = next_tick;
        const auto cycle_period = std::chrono::microseconds(cycle_us);
        const auto wheel_period = std::chrono::microseconds(wheel_cycle_us);
        std::array<uint64_t, 6> previous_rx{};
        uint64_t cycles = 0;
        uint64_t previous_cycles = 0;
        float rx_rate = 0.0F;
        float tx_rate = 0.0F;
        SafetyState safety_state = SafetyState::WaitingReady;
        uint32_t latch_reason = 0;
        std::array<bool, kMotorCount> has_ever_been_online{};
        bool active_control_started = false;

        while (g_running.load())
        {
            const auto loop_start = std::chrono::steady_clock::now();
            while (true)
            {
                CommandPacketNet packet{};
                sockaddr_in from{};
                socklen_t from_length = sizeof(from);
                const ssize_t bytes = recvfrom(udp_socket, &packet, sizeof(packet), 0,
                    reinterpret_cast<sockaddr*>(&from), &from_length);
                if (bytes < 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    continue;
                }
                if (bytes != static_cast<ssize_t>(sizeof(packet)) ||
                    packet.magic != kCommandMagic ||
                    (packet.version != kProtocolVersion &&
                     packet.version != kProtocolVersionV2) ||
                    packet.motor_count != kMotorCount ||
                    (packet.reserved & ~kKnownCommandFlags) != 0)
                {
                    continue;
                }
                bool valid = true;
                for (const auto& command : packet.motors) valid = valid && valid_command(command);
                if (!valid) continue;

                CommandArray packet_commands{};
                for (std::size_t i = 0; i < kMotorCount; ++i)
                {
                    packet_commands[i] = MotorCommand{packet.motors[i].torque,
                        packet.motors[i].position, packet.motors[i].velocity,
                        packet.motors[i].kp, packet.motors[i].kd};
                }
                remember_peer(feedback_peers, from, packet.version, loop_start);

                const bool safe_packet = is_safe_command(packet_commands);
                const bool release_requested =
                    (packet.reserved & kCommandFlagReleaseControl) != 0;
                if (release_requested)
                {
                    if (!safe_packet) continue;
                    if (has_active_controller &&
                        same_client(active_controller_address, from))
                    {
                        latest = CommandState{};
                        has_active_controller = false;
                        active_control_started = false;
                        std::cout << "[SAFETY] active controller released; safe damping"
                                  << std::endl;
                    }
                    continue;
                }
                if (!has_active_controller)
                {
                    if (safe_packet)
                    {
                        continue;
                    }
                    active_controller_address = from;
                    has_active_controller = true;
                    active_control_started = true;
                    std::cout << "[SAFETY] active controller locked to "
                              << inet_ntoa(from.sin_addr) << ":" << ntohs(from.sin_port)
                              << std::endl;
                }
                else if (!same_client(active_controller_address, from))
                {
                    continue;
                }

                latest.motors = packet_commands;
                latest.sequence = packet.sequence;
                latest.timestamp = loop_start;
                latest.valid = true;
            }

            const auto now = std::chrono::steady_clock::now();
            FeedbackPacketNet feedback{};
            feedback.magic = kFeedbackMagic;
            feedback.version = kProtocolVersion;
            feedback.motor_count = kMotorCount;
            feedback.sequence = feedback_sequence++;
            feedback.command_sequence = last_command_sequence;
            feedback.status_bits = kStatusReleaseSupported;
            feedback.cycle_time_us = static_cast<float>(cycle_us);
            feedback.command_age_ms = latest.valid
                ? static_cast<float>(std::chrono::duration<double, std::milli>(now - latest.timestamp).count())
                : 0.0F;

            for (std::size_t leg = 0; leg < kLegCount; ++leg)
            {
                const auto values = legs[leg]->feedback();
                for (std::size_t joint = 0; joint < 3; ++joint)
                {
                    feedback.motors[leg * 4 + joint] = values[joint];
                }
            }
            const auto front_wheels = wheels[0]->feedback();
            const auto rear_wheels = wheels[1]->feedback();
            const auto front_wheel_voltages = wheels[0]->voltages();
            const auto rear_wheel_voltages = wheels[1]->voltages();
            feedback.motors[3] = front_wheels[0];
            feedback.motors[7] = front_wheels[1];
            feedback.motors[11] = rear_wheels[0];
            feedback.motors[15] = rear_wheels[1];
            bool all_online = true;
            bool previously_online_motor_dropped = false;
            for (std::size_t motor = 0; motor < kMotorCount; ++motor)
            {
                const bool online = feedback.motors[motor].comm_age_ms < 200;
                if (online)
                {
                    has_ever_been_online[motor] = true;
                }
                else
                {
                    feedback.status_bits |= (1U << (motor + 1));
                    all_online = false;
                    if (has_ever_been_online[motor]) previously_online_motor_dropped = true;
                }
            }

            const bool command_stale = active_control_started &&
                (!latest.valid || now - latest.timestamp > command_timeout);
            if (command_stale)
            {
                latest = CommandState{};
                has_active_controller = false;
                active_control_started = false;
            }
            if (safety_state != SafetyState::Latched)
            {
                if (previously_online_motor_dropped)
                {
                    safety_state = SafetyState::Latched;
                    latch_reason |= kStatusLatchMotorOffline;
                    std::cerr << "[SAFETY] motor feedback loss latched; restart required" << std::endl;
                }
                else if (!all_online)
                {
                    safety_state = SafetyState::WaitingReady;
                }
                else if (active_control_started && !command_stale)
                {
                    safety_state = SafetyState::Running;
                }
                else
                {
                    safety_state = SafetyState::Ready;
                }
            }

            const bool allow_active_command = safety_state == SafetyState::Running &&
                                              !command_stale && all_online;
            const CommandArray commands = allow_active_command ? latest.motors : safe_commands();
            if (allow_active_command) last_command_sequence = latest.sequence;

            for (std::size_t leg = 0; leg < kLegCount; ++leg)
            {
                std::array<MotorCommand, 3> leg_commands{};
                std::copy_n(commands.begin() + leg * 4, 3, leg_commands.begin());
                legs[leg]->send(leg_commands);
            }
            if (now >= next_wheel_tick)
            {
                wheels[0]->send({commands[3], commands[7]});
                wheels[1]->send({commands[11], commands[15]});
                do
                {
                    next_wheel_tick += wheel_period;
                }
                while (next_wheel_tick <= now);
            }
            ++cycles;

            if (command_stale) feedback.status_bits |= kStatusCommandTimeout;
            if (safety_state == SafetyState::WaitingReady) feedback.status_bits |= kStatusWaitingReady;
            if (safety_state == SafetyState::Ready) feedback.status_bits |= kStatusReady;
            if (safety_state == SafetyState::Running) feedback.status_bits |= kStatusRunning;
            if (safety_state == SafetyState::Latched)
            {
                feedback.status_bits |= kStatusSafetyLatched | latch_reason;
            }

            if (now - last_rate_time >= std::chrono::seconds(1))
            {
                const double seconds = std::chrono::duration<double>(now - last_rate_time).count();
                uint64_t received = 0;
                uint64_t errors = 0;
                for (std::size_t i = 0; i < legs.size(); ++i)
                {
                    const uint64_t current = legs[i]->rx_count();
                    received += current - previous_rx[i];
                    previous_rx[i] = current;
                    errors += legs[i]->error_count();
                }
                for (std::size_t i = 0; i < wheels.size(); ++i)
                {
                    const uint64_t current = wheels[i]->rx_count();
                    received += current - previous_rx[i + 4];
                    previous_rx[i + 4] = current;
                    errors += wheels[i]->error_count();
                }
                rx_rate = static_cast<float>(received / seconds);
                tx_rate = static_cast<float>((cycles - previous_cycles) / seconds);
                previous_cycles = cycles;
                last_rate_time = now;
                std::cout << "[can_service_wheel] control=" << tx_rate
                          << "Hz rx=" << rx_rate << "fps state="
                          << static_cast<int>(safety_state)
                          << " errors=" << errors << std::endl;
            }
            feedback.rx_rate_hz = rx_rate;
            feedback.tx_rate_hz = tx_rate;

            const auto peer_expiry = now - std::chrono::seconds(2);
            feedback_peers.erase(
                std::remove_if(feedback_peers.begin(), feedback_peers.end(),
                    [peer_expiry](const ClientPeer& peer) {
                        return peer.last_seen < peer_expiry;
                    }),
                feedback_peers.end());
            for (const auto& peer : feedback_peers)
            {
                if (peer.protocol_version == kProtocolVersionV2)
                {
                    FeedbackPacketNetV2 feedback_v2{};
                    feedback_v2.magic = feedback.magic;
                    feedback_v2.version = kProtocolVersionV2;
                    feedback_v2.motor_count = feedback.motor_count;
                    feedback_v2.sequence = feedback.sequence;
                    feedback_v2.command_sequence = feedback.command_sequence;
                    feedback_v2.status_bits = feedback.status_bits;
                    feedback_v2.cycle_time_us = feedback.cycle_time_us;
                    feedback_v2.command_age_ms = feedback.command_age_ms;
                    feedback_v2.rx_rate_hz = feedback.rx_rate_hz;
                    feedback_v2.tx_rate_hz = feedback.tx_rate_hz;
                    for (std::size_t i = 0; i < kMotorCount; ++i)
                    {
                        feedback_v2.motors[i].position = feedback.motors[i].position;
                        feedback_v2.motors[i].velocity = feedback.motors[i].velocity;
                        feedback_v2.motors[i].torque = feedback.motors[i].torque;
                        feedback_v2.motors[i].temperature = feedback.motors[i].temperature;
                        feedback_v2.motors[i].voltage =
                            std::numeric_limits<float>::quiet_NaN();
                        feedback_v2.motors[i].error_code = feedback.motors[i].error_code;
                        feedback_v2.motors[i].pattern = feedback.motors[i].pattern;
                        feedback_v2.motors[i].comm_age_ms = feedback.motors[i].comm_age_ms;
                    }
                    feedback_v2.motors[3].voltage = front_wheel_voltages[0];
                    feedback_v2.motors[7].voltage = front_wheel_voltages[1];
                    feedback_v2.motors[11].voltage = rear_wheel_voltages[0];
                    feedback_v2.motors[15].voltage = rear_wheel_voltages[1];
                    sendto(udp_socket, &feedback_v2, sizeof(feedback_v2), 0,
                        reinterpret_cast<const sockaddr*>(&peer.address), sizeof(peer.address));
                }
                else
                {
                    sendto(udp_socket, &feedback, sizeof(feedback), 0,
                        reinterpret_cast<const sockaddr*>(&peer.address), sizeof(peer.address));
                }
            }

            next_tick += cycle_period;
            std::this_thread::sleep_until(next_tick);
            if (std::chrono::steady_clock::now() - next_tick > cycle_period * 5)
            {
                next_tick = std::chrono::steady_clock::now();
            }
        }

        for (auto& wheel : wheels) wheel->disable();
        for (auto& leg : legs) leg->disable();
        for (auto& wheel : wheels) wheel->stop();
        for (auto& leg : legs) leg->stop();
        ::close(udp_socket);
    }
    catch (const std::exception& error)
    {
        std::cerr << "can_service_wheel: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
