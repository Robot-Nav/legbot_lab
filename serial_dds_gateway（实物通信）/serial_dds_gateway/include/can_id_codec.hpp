#pragma once

// 29 位 CAN ID 与 4 字节扩展原始字段的移位互转，用于兼容旧工具。

#include <array>
#include <cstdint>

namespace serial_dds_gateway {

// 灵足 USB-CAN 串口帧不使用本 helper 处理字节 3..6；详见 lingzu_serial.hpp 与 lingzu_motor_protocol.hpp。
// 以下函数仅用于暴露移位后原始 CAN-ID 字段的旧工具。
uint32_t ext_raw_to_can29(const std::array<uint8_t, 4>& ext_raw, int shift_right = 3);
std::array<uint8_t, 4> can29_to_ext_raw(uint32_t can29, int shift_left = 3);

}  // namespace serial_dds_gateway
