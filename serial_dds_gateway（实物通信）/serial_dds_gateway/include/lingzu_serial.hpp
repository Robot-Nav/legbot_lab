#pragma once

#include <array>
#include <cstdint>

namespace serial_dds_gateway {

// Lingzu RS02 USB-CAN adapter wire format:
//   header 45 54 ("ET") + channel + frame_type + id_field(2) + master_id + dlc + data + tail 0D 0A.
inline constexpr std::array<uint8_t, 2> kLingzuUsbHeader = {0x45, 0x54};
inline constexpr std::array<uint8_t, 2> kLingzuUsbTail = {0x0D, 0x0A};
inline constexpr uint8_t kLingzuCanStandardFrame = 0x01;
inline constexpr uint8_t kLingzuCanExtendedFrame = 0x02;
inline constexpr uint8_t kLingzuMotorEnableCode = 0x03;
inline constexpr uint8_t kLingzuMotorDisableCode = 0x04;

inline constexpr uint16_t MakeLingzuIdField(uint8_t motor_id, uint8_t control = 0x00) {
  return static_cast<uint16_t>((static_cast<uint16_t>(control) << 8) | motor_id);
}

inline constexpr uint8_t LingzuMotorIdFromField(uint16_t id_field) {
  return static_cast<uint8_t>(id_field & 0xFF);
}

}  // namespace serial_dds_gateway
