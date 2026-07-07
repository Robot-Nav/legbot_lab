#pragma once

#include "protocol_codec.hpp"
#include "serial_framer.hpp"

#include <cstdint>

namespace serial_dds_gateway {

SerialFrame EncodeType1SerialFrame(uint8_t channel, uint8_t master_id, const Type1Command& cmd,
                                   const RangeSpec& ranges = {});
SerialFrame EncodeType1StandardSerialFrame(uint8_t channel, const Type1Command& cmd, const RangeSpec& ranges = {});
Type1Command DecodeType1SerialFrame(const SerialFrame& frame, const RangeSpec& ranges = {});

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

Type2Feedback DecodeType2SerialFrame(const SerialFrame& frame, const RangeSpec& ranges = {});
uint8_t MotorIdFromSerialFrame(const SerialFrame& frame);

}  // namespace serial_dds_gateway
