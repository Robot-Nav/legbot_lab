// 用途：RS02 电机 type1 控制命令编码与 type2 反馈解码。
// 说明：16 位无符号线性量化 q/dq/kp/kd/tau；type1 CAN ID 携带 mode=1 + 力矩 + 电机 ID；
//       type2 CAN ID 为 mode=2，数据域按 q/dq/tau/temp 顺序大端排列。

#include "protocol_codec.hpp"

#include <algorithm>
#include <cmath>

namespace serial_dds_gateway {

namespace {
double clampd(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

bool IsCalfMotorId(uint8_t motor_id) {
  // 电机 CAN ID 规则：腿号(1-4) * 10 + 关节号（1=髋，2=大腿，3=小腿）。
  // 例如 31=FR_calf，32=FL_calf，33=RR_calf，34=RL_calf。
  return (motor_id % 10) == 3;
}
}  // 命名空间

// 根据电机 ID 选择力矩范围：小腿电机减速比大，力矩范围更宽。
RangeSpec RangeSpecForMotor(uint8_t motor_id, const RangeSpec& base) {
  RangeSpec r = base;
  if (IsCalfMotorId(motor_id)) {
    r.tau_min = base.calf_tau_min;
    r.tau_max = base.calf_tau_max;
  }
  return r;
}

// 浮点物理量映射到 bits 位无符号整数：越界钳位，四舍五入。
uint16_t float_to_uint(double x, double x_min, double x_max, int bits) {
  const auto levels = (1u << bits) - 1u;
  x = clampd(x, x_min, x_max);
  const double span = x_max - x_min;
  const auto v = static_cast<uint32_t>(std::llround((x - x_min) * levels / span));
  return static_cast<uint16_t>(std::min<uint32_t>(v, levels));
}

// 无符号整数反映射为浮点物理量。
double uint_to_float(uint16_t u, double x_min, double x_max, int bits) {
  const auto levels = (1u << bits) - 1u;
  const auto v = std::min<uint32_t>(u, levels);
  return x_min + (x_max - x_min) * (static_cast<double>(v) / levels);
}

// 按 type1/type2 规则组装 29 位 CAN ID：mode(5bit) | data16(16bit) | id8(8bit)。
uint32_t build_can_id(uint8_t mode, uint16_t data16, uint8_t id8) {
  return (static_cast<uint32_t>(mode & 0x1F) << 24) | (static_cast<uint32_t>(data16) << 8) |
         static_cast<uint32_t>(id8);
}

// 拆分 29 位 CAN ID 到 mode/data16/id8 三个字段。
CanIdFields split_can_id(uint32_t can_id) {
  return CanIdFields{
      .mode = static_cast<uint8_t>((can_id >> 24) & 0x1F),
      .data16 = static_cast<uint16_t>((can_id >> 8) & 0xFFFF),
      .id8 = static_cast<uint8_t>(can_id & 0xFF),
  };
}

// 编码 type1 控制命令：q/dq/kp/kd 放入 8 字节数据域，力矩放入 CAN ID 的 data16 字段。
std::pair<uint32_t, std::array<uint8_t, 8>> encode_type1(const Type1Command& cmd, const RangeSpec& ranges) {
  const auto r = RangeSpecForMotor(cmd.motor_id, ranges);
  const uint16_t q_u16 = float_to_uint(cmd.q, r.q_min, r.q_max, 16);
  const uint16_t dq_u16 = float_to_uint(cmd.dq, r.dq_min, r.dq_max, 16);
  const uint16_t kp_u16 = float_to_uint(cmd.kp, r.kp_min, r.kp_max, 16);
  const uint16_t kd_u16 = float_to_uint(cmd.kd, r.kd_min, r.kd_max, 16);
  const uint16_t tau_u16 = float_to_uint(cmd.tau, r.tau_min, r.tau_max, 16);

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

// 解码 type2 反馈帧：验证 mode=2，按大端解析 q/dq/tau/temp。
Type2Feedback decode_type2(uint32_t can_id, const std::array<uint8_t, 8>& data, const RangeSpec& ranges) {
  const auto fields = split_can_id(can_id);
  if (fields.mode != 2) {
    throw std::runtime_error("decode_type2 called with non-type2 CAN ID");
  }
  const auto r = RangeSpecForMotor(fields.id8, ranges);
  const uint16_t q_u16 = static_cast<uint16_t>((data[0] << 8) | data[1]);
  const uint16_t dq_u16 = static_cast<uint16_t>((data[2] << 8) | data[3]);
  const uint16_t tau_u16 = static_cast<uint16_t>((data[4] << 8) | data[5]);
  const uint16_t temp_x10 = static_cast<uint16_t>((data[6] << 8) | data[7]);

  return Type2Feedback{
      .motor_id = fields.id8,
      .q = uint_to_float(q_u16, r.q_min, r.q_max, 16),
      .dq = uint_to_float(dq_u16, r.dq_min, r.dq_max, 16),
      .tau = uint_to_float(tau_u16, r.tau_min, r.tau_max, 16),
      .temp_c = static_cast<double>(temp_x10) / 10.0,
  };
}

}  // 命名空间 serial_dds_gateway

