#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "motor_control/motor_config.h"
#include "motor_control/socketcan_interface.h"

namespace {

constexpr std::size_t kBusCount = 4;
constexpr std::size_t kMotorCount = 3;
constexpr std::array<uint8_t, kMotorCount> kMotorIds = {4, 3, 2};
constexpr std::array<const char*, kBusCount> kLegNames = {
    "front_left", "front_right", "rear_left", "rear_right"
};
constexpr std::array<const char*, kMotorCount> kJointNames = {"hip", "thigh", "calf"};
constexpr float kCalfReductionRatio = 1.4615F;

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

struct JointSnapshot
{
    float position = 0.0F;
    float velocity = 0.0F;
    float torque = 0.0F;
    float temperature = 0.0F;
    uint8_t error_code = 0;
    uint8_t pattern = 0;
    uint64_t last_rx_ms = 0;
};

class JointBus
{
public:
    explicit JointBus(std::string interface_name)
        : interface_name_(std::move(interface_name)),
          interface_(std::make_unique<SocketCanInterface>(interface_name_))
    {
        controller_ = std::make_unique<MotorControlSet>(
            [this](const CanFrame& frame) { interface_->send(frame); },
            std::vector<uint8_t>(kMotorIds.begin(), kMotorIds.end()));
    }

    ~JointBus()
    {
        stop();
        disable();
    }

    void start()
    {
        running_.store(true);
        receive_thread_ = std::thread([this]() { receive_loop(); });
    }

    void stop()
    {
        running_.store(false);
        if (receive_thread_.joinable())
        {
            receive_thread_.join();
        }
    }

    void enable()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint8_t id : kMotorIds)
        {
            controller_->getMotor(id).Enable_Motor();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void send_damping()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint8_t id : kMotorIds)
        {
            float kd = 5.0F;
            if (id == 2)
            {
                kd /= kCalfReductionRatio * kCalfReductionRatio;
            }
            controller_->getMotor(id).RobStrite_Motor_move_control(
                0.0F, 0.0F, 0.0F, 0.0F, kd);
        }
        ++tx_cycles_;
    }

    void disable() noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (uint8_t id : kMotorIds)
            {
                controller_->getMotor(id).Disenable_Motor(0);
            }
        }
        catch (...)
        {
        }
    }

    std::array<JointSnapshot, kMotorCount> snapshot()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& feedback = controller_->getFeedback();
        std::array<JointSnapshot, kMotorCount> result{};
        for (std::size_t index = 0; index < kMotorCount; ++index)
        {
            result[index].position = feedback.pos[index];
            result[index].velocity = feedback.vel[index];
            result[index].torque = feedback.tor[index];
            result[index].temperature = feedback.temp[index];
            result[index].error_code = feedback.error_code[index];
            result[index].pattern = static_cast<uint8_t>(feedback.pattern[index]);
            result[index].last_rx_ms = feedback.last_rx_ms[index];
        }
        return result;
    }

    uint64_t rx_count() const { return rx_count_.load(); }
    uint64_t tx_cycles() const { return tx_cycles_.load(); }
    uint64_t errors() const { return errors_.load(); }

private:
    void receive_loop()
    {
        while (running_.load() && g_running.load())
        {
            try
            {
                CanFrame frame{};
                if (interface_->receive(frame, std::chrono::milliseconds(5)))
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    controller_->handleIncomingFrame(frame);
                    ++rx_count_;
                }
            }
            catch (const std::exception& error)
            {
                ++errors_;
                std::cerr << interface_name_ << " receive error: " << error.what() << std::endl;
                break;
            }
        }
    }

    std::string interface_name_;
    std::unique_ptr<SocketCanInterface> interface_;
    std::unique_ptr<MotorControlSet> controller_;
    std::mutex mutex_;
    std::thread receive_thread_;
    std::atomic_bool running_{false};
    std::atomic_uint64_t rx_count_{0};
    std::atomic_uint64_t tx_cycles_{0};
    std::atomic_uint64_t errors_{0};
};

} // namespace

int main(int argc, char** argv)
{
    double duration_seconds = 5.0;
    double rate_hz = 100.0;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--duration" && index + 1 < argc)
        {
            duration_seconds = std::stod(argv[++index]);
        }
        else if (argument == "--rate" && index + 1 < argc)
        {
            rate_hz = std::stod(argv[++index]);
        }
        else if (argument == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [--duration 5] [--rate 100]" << std::endl;
            return 0;
        }
    }
    if (duration_seconds <= 0.0 || rate_hz <= 0.0)
    {
        std::cerr << "duration and rate must be positive" << std::endl;
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try
    {
        std::array<std::unique_ptr<JointBus>, kBusCount> buses{};
        for (std::size_t bus = 0; bus < kBusCount; ++bus)
        {
            buses[bus] = std::make_unique<JointBus>("can" + std::to_string(bus));
            buses[bus]->start();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        for (auto& bus : buses)
        {
            bus->enable();
        }

        const auto period = std::chrono::duration<double>(1.0 / rate_hz);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration<double>(duration_seconds);
        auto next_tick = std::chrono::steady_clock::now();
        while (g_running.load() && std::chrono::steady_clock::now() < deadline)
        {
            for (auto& bus : buses)
            {
                bus->send_damping();
            }
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next_tick);
        }

        for (auto& bus : buses)
        {
            bus->disable();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        const uint64_t now = steady_now_ms();
        std::size_t online_count = 0;
        std::cout << "RESULT duration=" << duration_seconds << "s rate=" << rate_hz << "Hz" << std::endl;
        for (std::size_t bus = 0; bus < kBusCount; ++bus)
        {
            const auto values = buses[bus]->snapshot();
            std::cout << "BUS can" << bus << " leg=" << kLegNames[bus]
                      << " rx=" << buses[bus]->rx_count()
                      << " tx_cycles=" << buses[bus]->tx_cycles()
                      << " errors=" << buses[bus]->errors() << std::endl;
            for (std::size_t motor = 0; motor < kMotorCount; ++motor)
            {
                const uint64_t age = values[motor].last_rx_ms == 0
                    ? 65535
                    : (now > values[motor].last_rx_ms ? now - values[motor].last_rx_ms : 0);
                const bool online = age < 200;
                if (online) ++online_count;
                std::cout << "  id=" << static_cast<int>(kMotorIds[motor])
                          << " joint=" << kJointNames[motor]
                          << " online=" << online
                          << " age_ms=" << age
                          << " pos=" << values[motor].position
                          << " vel=" << values[motor].velocity
                          << " torque=" << values[motor].torque
                          << " temp=" << values[motor].temperature
                          << " error=0x" << std::hex << static_cast<int>(values[motor].error_code)
                          << std::dec << " pattern=" << static_cast<int>(values[motor].pattern)
                          << std::endl;
            }
        }
        std::cout << "ONLINE " << online_count << "/12" << std::endl;

        for (auto& bus : buses)
        {
            bus->stop();
        }
        return online_count == 12 ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "joint_damping_test: " << error.what() << std::endl;
        return 1;
    }
}
