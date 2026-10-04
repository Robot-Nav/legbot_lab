// Simple CAN heartbeat: continuously send zero-torque commands and refresh feedback.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "motor_control/motor_config.h"
#include "motor_control/socketcan_interface.h"

namespace {
std::atomic_bool g_keep_running{true};

void handle_signal(int)
{
    g_keep_running.store(false);
}
}

int main()
{
    const std::string interface_name = "can0";
    const std::vector<uint8_t> motor_ids = {0x01, 0x02, 0x03};

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try
    {
        SocketCanInterface can_iface(interface_name);
        SendCanFrameFn send_fn = [&can_iface](const CanFrame& frame) {
            can_iface.send(frame);
        };

        MotorControlSet controller(send_fn, motor_ids);

        for (uint8_t id : motor_ids)
        {
            auto& motor = controller.getMotor(id);
            motor.Enable_Motor();
            motor.RobStrite_Motor_move_control(0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
        }

        using Clock = std::chrono::steady_clock;
        auto last_report = Clock::now();
        std::uint64_t commands_sent = 0;
        std::uint64_t frames_received = 0;

        while (g_keep_running.load())
        {
            for (uint8_t id : motor_ids)
            {
                controller.getMotor(id).RobStrite_Motor_move_control(0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
                ++commands_sent;
            }

            CanFrame frame;
            while (g_keep_running.load() && can_iface.receive(frame, std::chrono::milliseconds(0)))
            {
                controller.handleIncomingFrame(frame);
                ++frames_received;
            }

            const auto now = Clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_report).count();
            if (elapsed >= 1.0)
            {
                const auto& feedback = controller.getFeedback();
                if (!feedback.pos.empty())
                {
                    std::cout << std::fixed << std::setprecision(3);
                    for (std::size_t i = 0; i < motor_ids.size() && i < feedback.pos.size(); ++i)
                    {
                        std::cout << "id=0x" << std::hex << int(motor_ids[i]) << std::dec
                                  << " pos=" << feedback.pos[i]
                                  << " vel=" << feedback.vel[i]
                                  << " tor=" << feedback.tor[i]
                                  << " temp=" << feedback.temp[i]
                                  << " err=" << int(feedback.error_code[i])
                                  << " pattern=" << int(feedback.pattern[i]) << '\n';
                        if (feedback.error_code[i] == 1)
                        {
                            std::cerr << "Warning: motor 0x" << std::hex << int(motor_ids[i]) << std::dec
                                      << " reported error code 1" << std::endl;
                        }
                    }
                    const double cmd_freq = commands_sent / elapsed;
                    const double rx_freq = frames_received / elapsed;
                    std::cout << "command rate=" << cmd_freq << " Hz, feedback rate=" << rx_freq << " Hz" << std::endl;
                    if (cmd_freq > 0.0)
                    {
                        const double diff_ratio = std::abs(cmd_freq - rx_freq) / cmd_freq;
                        if (diff_ratio > 0.05)
                        {
                            std::cerr << "Warning: command/feedback rate mismatch exceeds 10% (" << diff_ratio * 100.0
                                      << "%)" << std::endl;
                        }
                    }
                }
                last_report = now;
                commands_sent = 0;
                frames_received = 0;
            }
        }

        for (uint8_t id : motor_ids)
        {
            controller.getMotor(id).Disenable_Motor(0);
        }

        controller.shutdown();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
