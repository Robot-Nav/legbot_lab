#include "can_id_codec.hpp"

namespace serial_dds_gateway {

uint32_t ext_raw_to_can29(const std::array<uint8_t, 4>& ext_raw, int shift_right) {
  const uint32_t ext_u32 = (static_cast<uint32_t>(ext_raw[0]) << 24) | (static_cast<uint32_t>(ext_raw[1]) << 16) |
                           (static_cast<uint32_t>(ext_raw[2]) << 8) | static_cast<uint32_t>(ext_raw[3]);
  return (ext_u32 >> shift_right) & 0x1FFFFFFF;
}

std::array<uint8_t, 4> can29_to_ext_raw(uint32_t can29, int shift_left) {
  const uint32_t ext_u32 = ((can29 & 0x1FFFFFFF) << shift_left);
  return {
      static_cast<uint8_t>((ext_u32 >> 24) & 0xFF),
      static_cast<uint8_t>((ext_u32 >> 16) & 0xFF),
      static_cast<uint8_t>((ext_u32 >> 8) & 0xFF),
      static_cast<uint8_t>(ext_u32 & 0xFF),
  };
}

}  // namespace serial_dds_gateway

