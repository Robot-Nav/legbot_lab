#pragma once

#include <array>
#include <cstdint>

namespace serial_dds_gateway {

// 旧版 CAN-ID 移位辅助函数：用于暴露移位后 raw 字段的工具。
// 注意：本网关实际使用的灵足 USB-CAN 串口帧不经过此处；具体成帧见 lingzu_serial.hpp 与 lingzu_motor_protocol.hpp。
uint32_t ext_raw_to_can29(const std::array<uint8_t, 4>& ext_raw, int shift_right = 3);
std::array<uint8_t, 4> can29_to_ext_raw(uint32_t can29, int shift_left = 3);

}  // 命名空间 serial_dds_gateway

