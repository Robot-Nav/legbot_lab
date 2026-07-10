#pragma once

// 灵足电机串口协议层：type1 控制帧、type2 反馈帧、type3/4 模式帧的封装与解析。

#include "protocol_codec.hpp"
#include "serial_framer.hpp"

#include <cstdint>

namespace serial_dds_gateway {

// 封装 type1 扩展帧（反馈通道常用）与标准帧（主机下发常用）。
SerialFrame EncodeType1SerialFrame(uint8_t channel, uint8_t master_id, const Type1Command& cmd,
                                   const RangeSpec& ranges = {});
SerialFrame EncodeType1StandardSerialFrame(uint8_t channel, const Type1Command& cmd, const RangeSpec& ranges = {});
Type1Command DecodeType1SerialFrame(const SerialFrame& frame, const RangeSpec& ranges = {});

// 构造 type3/4 模式帧。
SerialFrame BuildMotorModeFrame(uint8_t channel, uint8_t master_id, uint8_t motor_id, uint8_t mode_code,
                                uint8_t data0 = 0);
inline SerialFrame BuildMotorEnableFrame(uint8_t channel, uint8_t master_id, uint8_t motor_id) {
  return BuildMotorModeFrame(channel, master_id, motor_id, kLingzuMotorEnableCode);
}
inline SerialFrame BuildMotorDisableFrame(uint8_t channel, uint8_t master_id, uint8_t motor_id) {
  return BuildMotorModeFrame(channel, master_id, motor_id, kLingzuMotorDisableCode);
}
inline SerialFrame BuildMotorClearFaultFrame(uint8_t channel, uint8_t master_id, uint8_t motor_id) {
  return BuildMotorModeFrame(channel, master_id, motor_id, kLingzuMotorDisableCode, 1);
}

// 解析 type2 反馈帧，并提取帧中的电机 CAN ID。
Type2Feedback DecodeType2SerialFrame(const SerialFrame& frame, const RangeSpec& ranges = {});
uint8_t MotorIdFromSerialFrame(const SerialFrame& frame);

}  // namespace serial_dds_gateway
