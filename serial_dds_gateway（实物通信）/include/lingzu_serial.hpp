#pragma once

#include <array>
#include <cstdint>

namespace serial_dds_gateway {

// 灵足 RS02 USB-CAN 适配器串口帧格式：
//   帧头 45 54（"ET"）+ channel + frame_type + id_field(2B) + master_id + dlc + data + 帧尾 0D 0A。
inline constexpr std::array<uint8_t, 2> kLingzuUsbHeader = {0x45, 0x54};
inline constexpr std::array<uint8_t, 2> kLingzuUsbTail = {0x0D, 0x0A};
inline constexpr uint8_t kLingzuCanStandardFrame = 0x01;   // type1 控制命令下发用标准帧
inline constexpr uint8_t kLingzuCanExtendedFrame = 0x02;   // type2 反馈为扩展帧
inline constexpr uint8_t kLingzuMotorEnableCode = 0x03;    // type3 电机使能
inline constexpr uint8_t kLingzuMotorDisableCode = 0x04;   // type4 电机失能/清错

// 由电机 CAN ID 与控制字节拼出 16 位 id_field。
inline constexpr uint16_t MakeLingzuIdField(uint8_t motor_id, uint8_t control = 0x00) {
  return static_cast<uint16_t>((static_cast<uint16_t>(control) << 8) | motor_id);
}

inline constexpr uint8_t LingzuMotorIdFromField(uint16_t id_field) {
  return static_cast<uint8_t>(id_field & 0xFF);
}

}  // 命名空间 serial_dds_gateway
