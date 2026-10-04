#ifndef MOTOR_CONTROL_NETWORK_PROTOCOL_H
#define MOTOR_CONTROL_NETWORK_PROTOCOL_H

#include <cstdint>

namespace motor_control::net {

constexpr uint16_t kDefaultCommandPort = 15000;
constexpr uint16_t kDefaultStatePort = 15001;
constexpr std::size_t kMaxMotors = 16;

#pragma pack(push, 1)
struct MotorCommand
{
    uint8_t id;
    uint8_t mode; // 0: motion control, 1: position, etc. Currently only 0 used.
    float torque;
    float position;
    float velocity;
    float kp;
    float kd;
};

struct CommandPacket
{
    uint64_t timestamp_ns;
    uint8_t motor_count;
    MotorCommand commands[kMaxMotors];
};

struct MotorState
{
    uint8_t id;
    float position;
    float velocity;
    float torque;
    float temperature;
    uint8_t error_code;
    int8_t pattern;
};

struct StatePacket
{
    uint64_t timestamp_ns;
    uint8_t motor_count;
    MotorState states[kMaxMotors];
};
#pragma pack(pop)

} // namespace motor_control::net

#endif
