#include "can_id_codec.hpp"

namespace serial_dds_gateway {

// 4 字节扩展 raw 字段拼接为 32 位无符号整数，再右移 shift_right 位取低 29 位得到真实 CAN ID。
uint32_t ext_raw_to_can29(const std::array<uint8_t, 4>& ext_raw, int shift_right) {
  const uint32_t ext_u32 = (static_cast<uint32_t>(ext_raw[0]) << 24) | (static_cast<uint32_t>(ext_raw[1]) << 16) |
                           (static_cast<uint32_t>(ext_raw[2]) << 8) | static_cast<uint32_t>(ext_raw[3]);
  return (ext_u32 >> shift_right) & 0x1FFFFFFF;
}

// 29 位 CAN ID 左移 shift_left 位后拆回 4 字节大端 raw 字段。
std::array<uint8_t, 4> can29_to_ext_raw(uint32_t can29, int shift_left) {
  const uint32_t ext_u32 = ((can29 & 0x1FFFFFFF) << shift_left);
  return {
      static_cast<uint8_t>((ext_u32 >> 24) & 0xFF),
      static_cast<uint8_t>((ext_u32 >> 16) & 0xFF),
      static_cast<uint8_t>((ext_u32 >> 8) & 0xFF),
      static_cast<uint8_t>(ext_u32 & 0xFF),
  };
}

}  // 命名空间 serial_dds_gateway

