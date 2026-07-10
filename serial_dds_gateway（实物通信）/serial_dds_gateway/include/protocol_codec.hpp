#pragma once

// type1/type2 CAN 量化编解码：16 位无符号定点，以及 29 位 CAN ID 拆装。

#include <array>
#include <cstdint>
#include <stdexcept>

namespace serial_dds_gateway {

// 各物理量的量化范围。
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

// 从 29 位 CAN ID 拆出的模式/数据/ID 字段。
struct CanIdFields {
  uint8_t mode{0};     // 通信类型：1=type1, 2=type2
  uint16_t data16{0};  // 16 位数据域
  uint8_t id8{0};      // 8 位电机 CAN ID
};

// type1 控制指令。
struct Type1Command {
  uint8_t motor_id{0};
  double q{0.0};
  double dq{0.0};
  double kp{0.0};
  double kd{0.0};
  double tau{0.0};
};

// type2 电机反馈。
struct Type2Feedback {
  uint8_t motor_id{0};
  double q{0.0};
  double dq{0.0};
  double tau{0.0};
  double temp_c{0.0};
};

// 浮点与指定位宽无符号整数互转。
uint16_t float_to_uint(double x, double x_min, double x_max, int bits = 16);
double uint_to_float(uint16_t u, double x_min, double x_max, int bits = 16);

// 29 位 CAN ID 组合/拆分。
uint32_t build_can_id(uint8_t mode, uint16_t data16, uint8_t id8);
CanIdFields split_can_id(uint32_t can_id);

// type1 编码为 (CAN ID, 8 字节数据)；type2 从 CAN ID+数据解码。
std::pair<uint32_t, std::array<uint8_t, 8>> encode_type1(const Type1Command& cmd, const RangeSpec& ranges = {});
Type2Feedback decode_type2(uint32_t can_id, const std::array<uint8_t, 8>& data, const RangeSpec& ranges = {});

}  // namespace serial_dds_gateway
