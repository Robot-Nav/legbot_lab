#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace serial_dds_gateway {

// type1/type2 字段的物理量范围；用于 16 位无符号整数与浮点之间的线性量化。
struct RangeSpec {
  double q_min{-12.5663706144};  // -4*pi
  double q_max{12.5663706144};   //  4*pi
  double dq_min{-44.0};
  double dq_max{44.0};
  double kp_min{0.0};
  double kp_max{500.0};
  double kd_min{0.0};
  double kd_max{5.0};
  double tau_min{-16.0};          // hip/thigh: ±16 N·m
  double tau_max{16.0};
  double calf_tau_min{-32.0};     // calf: ±32 N·m
  double calf_tau_max{32.0};
};

// 根据电机 CAN ID 返回对应的力矩范围（小腿电机范围更大）。
// CAN ID 个位：1=hip，2=thigh，3=calf；例如 31=FR_calf。
RangeSpec RangeSpecForMotor(uint8_t motor_id, const RangeSpec& base = {});

// 29 位 CAN ID 拆分为 mode(5bit) + data16(16bit) + id8(8bit)。
struct CanIdFields {
  uint8_t mode{0};
  uint16_t data16{0};
  uint8_t id8{0};
};

// type1 控制命令：目标位置、速度、位置环/速度环增益与前馈力矩。
struct Type1Command {
  uint8_t motor_id{0};
  double q{0.0};
  double dq{0.0};
  double kp{0.0};
  double kd{0.0};
  double tau{0.0};
};

// type2 反馈：当前位置、速度、估计力矩与温度（温度放大 10 倍存储）。
struct Type2Feedback {
  uint8_t motor_id{0};
  double q{0.0};
  double dq{0.0};
  double tau{0.0};
  double temp_c{0.0};
};

// 浮点 -> 无符号整数的线性量化。
uint16_t float_to_uint(double x, double x_min, double x_max, int bits = 16);
// 无符号整数 -> 浮点的线性反量化。
double uint_to_float(uint16_t u, double x_min, double x_max, int bits = 16);

// 按 type1/type2 格式拼接 29 位 CAN ID。
uint32_t build_can_id(uint8_t mode, uint16_t data16, uint8_t id8);
// 拆分 29 位 CAN ID。
CanIdFields split_can_id(uint32_t can_id);

// 将 type1 命令编码为 (CAN_ID, 8 字节数据)；CAN_ID 中承载前馈力矩。
std::pair<uint32_t, std::array<uint8_t, 8>> encode_type1(const Type1Command& cmd, const RangeSpec& ranges = {});
// 解码 type2 反馈帧。
Type2Feedback decode_type2(uint32_t can_id, const std::array<uint8_t, 8>& data, const RangeSpec& ranges = {});

}  // 命名空间 serial_dds_gateway

