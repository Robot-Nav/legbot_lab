#pragma once

// 灵足 RS02 USB-CAN 适配器串口格式常量：帧头、帧尾、通信类型码。

#include <array>
#include <cstdint>

namespace serial_dds_gateway {

// 串口帧格式：头 45 54（"ET"） + channel + frame_type + id_field(2) + master_id + dlc + data + 尾 0D 0A。
inline constexpr std::array<uint8_t, 2> kLingzuUsbHeader = {0x45, 0x54};
inline constexpr std::array<uint8_t, 2> kLingzuUsbTail = {0x0D, 0x0A};
inline constexpr uint8_t kLingzuCanStandardFrame = 0x01;  // 标准运控帧
inline constexpr uint8_t kLingzuCanExtendedFrame = 0x02;  // 扩展反馈帧
inline constexpr uint8_t kLingzuMotorEnableCode = 0x03;   // type3 使能
inline constexpr uint8_t kLingzuMotorDisableCode = 0x04;  // type4 失能/清故障

// 由电机 CAN ID 与控制字节构造 16 位 id_field。
inline constexpr uint16_t MakeLingzuIdField(uint8_t motor_id, uint8_t control = 0x00) {
  return static_cast<uint16_t>((static_cast<uint16_t>(control) << 8) | motor_id);
}

// 从 id_field 提取电机 CAN ID。
inline constexpr uint8_t LingzuMotorIdFromField(uint16_t id_field) {
  return static_cast<uint8_t>(id_field & 0xFF);
}

}  // namespace serial_dds_gateway
