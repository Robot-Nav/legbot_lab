#include "protocol_codec.hpp"

#include <algorithm>
#include <cmath>

namespace serial_dds_gateway {

namespace {
double clampd(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}
}  // namespace

uint16_t float_to_uint(double x, double x_min, double x_max, int bits) {
  const auto levels = (1u << bits) - 1u;
  x = clampd(x, x_min, x_max);
  const double span = x_max - x_min;
  const auto v = static_cast<uint32_t>(std::llround((x - x_min) * levels / span));
  return static_cast<uint16_t>(std::min<uint32_t>(v, levels));
}

double uint_to_float(uint16_t u, double x_min, double x_max, int bits) {
  const auto levels = (1u << bits) - 1u;
  const auto v = std::min<uint32_t>(u, levels);
  return x_min + (x_max - x_min) * (static_cast<double>(v) / levels);
}

uint32_t build_can_id(uint8_t mode, uint16_t data16, uint8_t id8) {
  return (static_cast<uint32_t>(mode & 0x1F) << 24) | (static_cast<uint32_t>(data16) << 8) |
         static_cast<uint32_t>(id8);
}

CanIdFields split_can_id(uint32_t can_id) {
  return CanIdFields{
      .mode = static_cast<uint8_t>((can_id >> 24) & 0x1F),
      .data16 = static_cast<uint16_t>((can_id >> 8) & 0xFFFF),
      .id8 = static_cast<uint8_t>(can_id & 0xFF),
  };
}

std::pair<uint32_t, std::array<uint8_t, 8>> encode_type1(const Type1Command& cmd, const RangeSpec& ranges) {
  const uint16_t q_u16 = float_to_uint(cmd.q, ranges.q_min, ranges.q_max, 16);
  const uint16_t dq_u16 = float_to_uint(cmd.dq, ranges.dq_min, ranges.dq_max, 16);
  const uint16_t kp_u16 = float_to_uint(cmd.kp, ranges.kp_min, ranges.kp_max, 16);
  const uint16_t kd_u16 = float_to_uint(cmd.kd, ranges.kd_min, ranges.kd_max, 16);
  const uint16_t tau_u16 = float_to_uint(cmd.tau, ranges.tau_min, ranges.tau_max, 16);

  const uint32_t can_id = build_can_id(1, tau_u16, cmd.motor_id);
  std::array<uint8_t, 8> data{};
  data[0] = static_cast<uint8_t>((q_u16 >> 8) & 0xFF);
  data[1] = static_cast<uint8_t>(q_u16 & 0xFF);
  data[2] = static_cast<uint8_t>((dq_u16 >> 8) & 0xFF);
  data[3] = static_cast<uint8_t>(dq_u16 & 0xFF);
  data[4] = static_cast<uint8_t>((kp_u16 >> 8) & 0xFF);
  data[5] = static_cast<uint8_t>(kp_u16 & 0xFF);
  data[6] = static_cast<uint8_t>((kd_u16 >> 8) & 0xFF);
  data[7] = static_cast<uint8_t>(kd_u16 & 0xFF);
  return {can_id, data};
}

Type2Feedback decode_type2(uint32_t can_id, const std::array<uint8_t, 8>& data, const RangeSpec& ranges) {
  const auto fields = split_can_id(can_id);
  if (fields.mode != 2) {
    throw std::runtime_error("decode_type2 called with non-type2 CAN ID");
  }
  const uint16_t q_u16 = static_cast<uint16_t>((data[0] << 8) | data[1]);
  const uint16_t dq_u16 = static_cast<uint16_t>((data[2] << 8) | data[3]);
  const uint16_t tau_u16 = static_cast<uint16_t>((data[4] << 8) | data[5]);
  const uint16_t temp_x10 = static_cast<uint16_t>((data[6] << 8) | data[7]);

  return Type2Feedback{
      .motor_id = fields.id8,
      .q = uint_to_float(q_u16, ranges.q_min, ranges.q_max, 16),
      .dq = uint_to_float(dq_u16, ranges.dq_min, ranges.dq_max, 16),
      .tau = uint_to_float(tau_u16, ranges.tau_min, ranges.tau_max, 16),
      .temp_c = static_cast<double>(temp_x10) / 10.0,
  };
}

}  // namespace serial_dds_gateway

