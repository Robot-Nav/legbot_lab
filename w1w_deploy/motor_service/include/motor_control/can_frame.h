#ifndef MOTOR_CONTROL_CAN_FRAME_H
#define MOTOR_CONTROL_CAN_FRAME_H

#include <array>
#include <cstdint>
#include <functional>

struct CanFrame {
    bool is_extended = false;
    bool is_rtr = false;
    uint32_t id = 0;
    uint8_t dlc = 0;
    std::array<uint8_t, 8> data{};
};

using SendCanFrameFn = std::function<void(const CanFrame&)>;

#endif
