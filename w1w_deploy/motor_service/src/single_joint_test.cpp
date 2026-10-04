#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "motor_control/motor_config.h"
#include "motor_control/socketcan_interface.h"

namespace {

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

} // namespace

int main(int argc, char** argv)
{
    std::string interface_name = "can3";
    uint8_t motor_id = 2;
    double duration_seconds = 5.0;
    double rate_hz = 100.0;
    float kd = 5.0F;

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--interface" && index + 1 < argc)
        {
            interface_name = argv[++index];
        }
        else if (argument == "--id" && index + 1 < argc)
        {
            motor_id = static_cast<uint8_t>(std::stoi(argv[++index]));
        }
        else if (argument == "--duration" && index + 1 < argc)
        {
            duration_seconds = std::stod(argv[++index]);
        }
        else if (argument == "--rate" && index + 1 < argc)
        {
            rate_hz = std::stod(argv[++index]);
        }
        else if (argument == "--kd" && index + 1 < argc)
        {
            kd = std::stof(argv[++index]);
        }
        else if (argument == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--interface can3] [--id 2] [--duration 5]"
                      << " [--rate 100] [--kd 5]" << std::endl;
            return 0;
        }
    }

    if (motor_id == 0 || duration_seconds <= 0.0 || rate_hz <= 0.0 || kd < 0.0F)
    {
        std::cerr << "invalid test arguments" << std::endl;
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try
    {
        SocketCanInterface interface(interface_name);
        std::mutex controller_mutex;
        MotorControlSet controller(
            [&interface](const CanFrame& frame) { interface.send(frame); },
            std::vector<uint8_t>{motor_id});

        std::atomic_bool receive_running{true};
        std::atomic_uint64_t receive_count{0};
        std::thread receive_thread([&]() {
            while (receive_running.load() && g_running.load())
            {
                try
                {
                    CanFrame frame{};
                    if (interface.receive(frame, std::chrono::milliseconds(5)))
                    {
                        std::lock_guard<std::mutex> lock(controller_mutex);
                        controller.handleIncomingFrame(frame);
                        ++receive_count;
                    }
                }
                catch (const std::exception& error)
                {
                    std::cerr << "receive error: " << error.what() << std::endl;
                    g_running.store(false);
                }
            }
        });

        auto disable = [&]() {
            try
            {
                std::lock_guard<std::mutex> lock(controller_mutex);
                controller.getMotor(motor_id).Disenable_Motor(0);
            }
            catch (...)
            {
            }
        };

        {
            std::lock_guard<std::mutex> lock(controller_mutex);
            controller.getMotor(motor_id).Disenable_Motor(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        {
            std::lock_guard<std::mutex> lock(controller_mutex);
            controller.getMotor(motor_id).Enable_Motor();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        const auto period = std::chrono::duration<double>(1.0 / rate_hz);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration<double>(duration_seconds);
        auto next_tick = std::chrono::steady_clock::now();
        uint64_t send_count = 0;
        while (g_running.load() && std::chrono::steady_clock::now() < deadline)
        {
            {
                std::lock_guard<std::mutex> lock(controller_mutex);
                controller.getMotor(motor_id).RobStrite_Motor_move_control(
                    0.0F, 0.0F, 0.0F, 0.0F, kd);
            }
            ++send_count;
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next_tick);
        }

        disable();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        MotorFeedback feedback{};
        {
            std::lock_guard<std::mutex> lock(controller_mutex);
            feedback = controller.getFeedback();
        }
        const uint64_t now = steady_now_ms();
        const uint64_t last_rx = feedback.last_rx_ms.empty() ? 0 : feedback.last_rx_ms[0];
        const uint64_t age = last_rx == 0 ? 65535 : (now > last_rx ? now - last_rx : 0);
        const bool online = age < 200;

        std::cout << "RESULT interface=" << interface_name
                  << " id=" << static_cast<int>(motor_id)
                  << " sends=" << send_count
                  << " receives=" << receive_count.load()
                  << " online=" << online
                  << " age_ms=" << age;
        if (!feedback.pos.empty())
        {
            std::cout << " position=" << feedback.pos[0]
                      << " velocity=" << feedback.vel[0]
                      << " torque=" << feedback.tor[0]
                      << " temperature=" << feedback.temp[0]
                      << " error=0x" << std::hex
                      << static_cast<int>(feedback.error_code[0]) << std::dec
                      << " pattern=" << static_cast<int>(feedback.pattern[0]);
        }
        std::cout << std::endl;

        receive_running.store(false);
        receive_thread.join();
        return online ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "single_joint_test: " << error.what() << std::endl;
        return 1;
    }
}
