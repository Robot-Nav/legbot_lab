// 灵足电机串口协议层实现：type1/2/3/4 帧的封装与解析。

#include "lingzu_motor_protocol.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace serial_dds_gateway {

namespace {

// 校验帧类型与 8 字节数据长度。
void RequireData8(const SerialFrame& frame, const char* context) {
  if (frame.frame_type != kLingzuCanStandardFrame && frame.frame_type != kLingzuCanExtendedFrame) {
    throw std::runtime_error(std::string(context) + ": unsupported CAN serial frame type");
  }
  if (frame.data.size() != 8) {
    throw std::runtime_error(std::string(context) + ": expected 8-byte data");
  }
}

// 大端 16 位读取。
uint16_t U16BE(const std::vector<uint8_t>& data, size_t offset) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[offset]) << 8) | data[offset + 1]);
}

// 标准帧的 tau 保存在 id_field 中；扩展帧 tau 未携带，返回 0。
double TauFromSerialFrame(const SerialFrame& frame, const RangeSpec& ranges) {
  if (frame.frame_type == kLingzuCanStandardFrame) {
    return uint_to_float(frame.id_field, ranges.tau_min, ranges.tau_max, 16);
  }
  return 0.0;
}

}  // namespace

// 从串口帧提取电机 CAN ID：标准帧取 master_id，扩展帧取 id_field 低 8 位。
uint8_t MotorIdFromSerialFrame(const SerialFrame& frame) {
  if (frame.frame_type == kLingzuCanStandardFrame) {
    return frame.master_id;
  }
  if (frame.frame_type == kLingzuCanExtendedFrame) {
    return LingzuMotorIdFromField(frame.id_field);
  }
  throw std::runtime_error("unsupported CAN serial frame type");
}

// type1 扩展帧封装（用于回环/测试）。
SerialFrame EncodeType1SerialFrame(uint8_t channel, uint8_t master_id, const Type1Command& cmd,
                                   const RangeSpec& ranges) {
  const auto encoded = encode_type1(cmd, ranges);
  return SerialFrame{
      .channel = channel,
      .frame_type = kLingzuCanExtendedFrame,
      .id_field = MakeLingzuIdField(cmd.motor_id),
      .master_id = master_id,
      .data = std::vector<uint8_t>(encoded.second.begin(), encoded.second.end()),
  };
}

// type1 标准帧封装（主机下发常用，tau 嵌入 id_field）。
SerialFrame EncodeType1StandardSerialFrame(uint8_t channel, const Type1Command& cmd, const RangeSpec& ranges) {
  const auto encoded = encode_type1(cmd, ranges);
  const auto fields = split_can_id(encoded.first);
  return SerialFrame{
      .channel = channel,
      .frame_type = kLingzuCanStandardFrame,
      .id_field = fields.data16,
      .master_id = cmd.motor_id,
      .data = std::vector<uint8_t>(encoded.second.begin(), encoded.second.end()),
  };
}

// type1 解码：数据区 q/dq/kp/kd；标准帧额外从 id_field 解码 tau。
Type1Command DecodeType1SerialFrame(const SerialFrame& frame, const RangeSpec& ranges) {
  RequireData8(frame, "DecodeType1SerialFrame");
  return Type1Command{
      .motor_id = MotorIdFromSerialFrame(frame),
      .q = uint_to_float(U16BE(frame.data, 0), ranges.q_min, ranges.q_max, 16),
      .dq = uint_to_float(U16BE(frame.data, 2), ranges.dq_min, ranges.dq_max, 16),
      .kp = uint_to_float(U16BE(frame.data, 4), ranges.kp_min, ranges.kp_max, 16),
      .kd = uint_to_float(U16BE(frame.data, 6), ranges.kd_min, ranges.kd_max, 16),
      .tau = TauFromSerialFrame(frame, ranges),
  };
}

// type3/4 模式帧：使能、失能、清故障。
SerialFrame BuildMotorModeFrame(uint8_t channel, uint8_t master_id, uint8_t motor_id, uint8_t mode_code,
                                uint8_t data0) {
  std::vector<uint8_t> data(8, 0);
  data[0] = data0;
  return SerialFrame{
      .channel = channel,
      .frame_type = mode_code,
      .id_field = static_cast<uint16_t>(master_id),
      .master_id = motor_id,
      .data = data,
  };
}

// type2 反馈帧解码：数据区 q/dq/tau/temp_x10。
Type2Feedback DecodeType2SerialFrame(const SerialFrame& frame, const RangeSpec& ranges) {
  RequireData8(frame, "DecodeType2SerialFrame");
  if (frame.frame_type != kLingzuCanExtendedFrame) {
    throw std::runtime_error("DecodeType2SerialFrame: feedback must be communication type 2 extended frame");
  }
  return Type2Feedback{
      .motor_id = MotorIdFromSerialFrame(frame),
      .q = uint_to_float(U16BE(frame.data, 0), ranges.q_min, ranges.q_max, 16),
      .dq = uint_to_float(U16BE(frame.data, 2), ranges.dq_min, ranges.dq_max, 16),
      .tau = uint_to_float(U16BE(frame.data, 4), ranges.tau_min, ranges.tau_max, 16),
      .temp_c = static_cast<double>(U16BE(frame.data, 6)) / 10.0,
  };
}

}  // namespace serial_dds_gateway
