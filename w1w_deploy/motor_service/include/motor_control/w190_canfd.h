#ifndef MOTOR_CONTROL_W190_CANFD_H
#define MOTOR_CONTROL_W190_CANFD_H

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

struct W190Command
{
    float torque = 0.0F;
    float position = 0.0F;
    float velocity = 0.0F;
    float kp = 0.0F;
    float kd = 0.0F;
};

struct W190Feedback
{
    uint8_t motor_id = 0;
    uint16_t msg_id = 0;
    float position = 0.0F;
    float velocity = 0.0F;
    float torque = 0.0F;
    float temperature = 0.0F;
    float voltage = 0.0F;
    uint8_t status = 0;
    uint64_t last_rx_ms = 0;
};

class W190CanFd
{
public:
    explicit W190CanFd(const std::string& interface_name, uint16_t msg_id = 0x0100);
    ~W190CanFd();

    W190CanFd(const W190CanFd&) = delete;
    W190CanFd& operator=(const W190CanFd&) = delete;

    void enable();
    void disable();
    void sendMotion(const std::array<W190Command, 2>& commands);
    bool receive(W190Feedback& feedback, std::chrono::milliseconds timeout);

private:
    void sendSpecial(uint8_t command_code);
    void sendPayload(const std::array<uint8_t, 64>& payload);

    int socket_fd_ = -1;
    uint16_t msg_id_ = 0x0100;
};

#endif
