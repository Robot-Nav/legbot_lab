#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace serial_dds_gateway {

struct RangeSpec {
  double q_min{-12.5663706144};  // -4*pi
  double q_max{12.5663706144};   //  4*pi
  double dq_min{-44.0};
  double dq_max{44.0};
  double kp_min{0.0};
  double kp_max{500.0};
  double kd_min{0.0};
  double kd_max{5.0};
  double tau_min{-17.0};
  double tau_max{17.0};
};

struct CanIdFields {
  uint8_t mode{0};
  uint16_t data16{0};
  uint8_t id8{0};
};

struct Type1Command {
  uint8_t motor_id{0};
  double q{0.0};
  double dq{0.0};
  double kp{0.0};
  double kd{0.0};
  double tau{0.0};
};

struct Type2Feedback {
  uint8_t motor_id{0};
  double q{0.0};
  double dq{0.0};
  double tau{0.0};
  double temp_c{0.0};
};

uint16_t float_to_uint(double x, double x_min, double x_max, int bits = 16);
double uint_to_float(uint16_t u, double x_min, double x_max, int bits = 16);

uint32_t build_can_id(uint8_t mode, uint16_t data16, uint8_t id8);
CanIdFields split_can_id(uint32_t can_id);

std::pair<uint32_t, std::array<uint8_t, 8>> encode_type1(const Type1Command& cmd, const RangeSpec& ranges = {});
Type2Feedback decode_type2(uint32_t can_id, const std::array<uint8_t, 8>& data, const RangeSpec& ranges = {});

}  // namespace serial_dds_gateway

