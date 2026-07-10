#include "lingzu_motor_protocol.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace serial_dds_gateway {

namespace {

// 检查帧类型与数据长度，type1/type2 均需 8 字节数据。
void RequireData8(const SerialFrame& frame, const char* context) {
  if (frame.frame_type != kLingzuCanStandardFrame && frame.frame_type != kLingzuCanExtendedFrame) {
    throw std::runtime_error(std::string(context) + ": unsupported CAN serial frame type");
  }
  if (frame.data.size() != 8) {
    throw std::runtime_error(std::string(context) + ": expected 8-byte data");
  }
}

// 大端 16 位无符号整数读取。
uint16_t U16BE(const std::vector<uint8_t>& data, size_t offset) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[offset]) << 8) | data[offset + 1]);
}

// 标准帧的力矩存放在 id_field（16 位）中；扩展帧的力矩字段为 0。
double TauFromSerialFrame(const SerialFrame& frame, const RangeSpec& ranges) {
  if (frame.frame_type == kLingzuCanStandardFrame) {
    const uint8_t motor_id = frame.master_id;
    const auto r = RangeSpecForMotor(motor_id, ranges);
    return uint_to_float(frame.id_field, r.tau_min, r.tau_max, 16);
  }
  return 0.0;
}

}  // 命名空间

uint8_t MotorIdFromSerialFrame(const SerialFrame& frame) {
  if (frame.frame_type == kLingzuCanStandardFrame) {
    return frame.master_id;
  }
  if (frame.frame_type == kLingzuCanExtendedFrame) {
    return LingzuMotorIdFromField(frame.id_field);
  }
  throw std::runtime_error("unsupported CAN serial frame type");
}

// 构造扩展 type1 串口帧（旧版/调试）。
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

// 构造标准 type1 串口帧：CAN ID 放 master_id，力矩放 id_field。
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

Type1Command DecodeType1SerialFrame(const SerialFrame& frame, const RangeSpec& ranges) {
  RequireData8(frame, "DecodeType1SerialFrame");
  const uint8_t motor_id = MotorIdFromSerialFrame(frame);
  const auto r = RangeSpecForMotor(motor_id, ranges);
  return Type1Command{
      .motor_id = motor_id,
      .q = uint_to_float(U16BE(frame.data, 0), r.q_min, r.q_max, 16),
      .dq = uint_to_float(U16BE(frame.data, 2), r.dq_min, r.dq_max, 16),
      .kp = uint_to_float(U16BE(frame.data, 4), r.kp_min, r.kp_max, 16),
      .kd = uint_to_float(U16BE(frame.data, 6), r.kd_min, r.kd_max, 16),
      .tau = TauFromSerialFrame(frame, ranges),
  };
}

// 构造 type3/4 模式帧：帧类型即模式码，id_field 为主机 ID，master_id 为电机 ID。
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

Type2Feedback DecodeType2SerialFrame(const SerialFrame& frame, const RangeSpec& ranges) {
  RequireData8(frame, "DecodeType2SerialFrame");
  if (frame.frame_type != kLingzuCanExtendedFrame) {
    throw std::runtime_error("DecodeType2SerialFrame: feedback must be communication type 2 extended frame");
  }
  const uint8_t motor_id = MotorIdFromSerialFrame(frame);
  const auto r = RangeSpecForMotor(motor_id, ranges);
  return Type2Feedback{
      .motor_id = motor_id,
      .q = uint_to_float(U16BE(frame.data, 0), r.q_min, r.q_max, 16),
      .dq = uint_to_float(U16BE(frame.data, 2), r.dq_min, r.dq_max, 16),
      .tau = uint_to_float(U16BE(frame.data, 4), r.tau_min, r.tau_max, 16),
      .temp_c = static_cast<double>(U16BE(frame.data, 6)) / 10.0,
  };
}

}  // 命名空间 serial_dds_gateway
