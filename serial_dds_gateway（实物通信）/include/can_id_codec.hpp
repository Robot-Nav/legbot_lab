#pragma once

#include <array>
#include <cstdint>

namespace serial_dds_gateway {

// Legacy helpers for tools that expose a shifted raw CAN-ID field.
// The Lingzu USB-CAN serial frame used by this gateway does not pass byte 3..6
// through these helpers; see lingzu_serial.hpp and lingzu_motor_protocol.hpp.
uint32_t ext_raw_to_can29(const std::array<uint8_t, 4>& ext_raw, int shift_right = 3);
std::array<uint8_t, 4> can29_to_ext_raw(uint32_t can29, int shift_left = 3);

}  // namespace serial_dds_gateway

