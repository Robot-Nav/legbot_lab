#pragma once

#include "protocol_codec.hpp"
#include "serial_framer.hpp"

#include <cstdint>

namespace serial_dds_gateway {

// 将 type1 控制命令封装为灵足扩展串口帧（旧版/调试用）。
SerialFrame EncodeType1SerialFrame(uint8_t channel, uint8_t master_id, const Type1Command& cmd,
                                   const RangeSpec& ranges = {});
// 将 type1 控制命令封装为标准串口帧（网关下发实际使用）。
SerialFrame EncodeType1StandardSerialFrame(uint8_t channel, const Type1Command& cmd, const RangeSpec& ranges = {});
// 解析串口帧中的 type1 控制命令。
Type1Command DecodeType1SerialFrame(const SerialFrame& frame, const RangeSpec& ranges = {});

// 构造 type3/type4 电机模式帧（使能、失能、清错）。
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

// 解析 type2 反馈帧。
Type2Feedback DecodeType2SerialFrame(const SerialFrame& frame, const RangeSpec& ranges = {});
// 从串口帧中提取目标电机 CAN ID（标准帧在 master_id，扩展帧在 id_field 低 8 位）。
uint8_t MotorIdFromSerialFrame(const SerialFrame& frame);

}  // 命名空间 serial_dds_gateway
