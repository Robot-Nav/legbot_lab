#include "protocol_codec.hpp"

#include <array>
#include <iomanip>
#include <iostream>

using namespace serial_dds_gateway;

int main() {
  RangeSpec ranges;

  Type1Command cmd{
      .motor_id = 0x01,
      .q = 1.20,
      .dq = 0.80,
      .kp = 40.0,
      .kd = 1.0,
      .tau = 2.5,
  };

  auto [can_id_tx, data_tx] = encode_type1(cmd, ranges);
  std::cout << "[TYPE1 TX]\n";
  std::cout << "can_id = 0x" << std::hex << std::uppercase << can_id_tx << std::dec << "\n";
  std::cout << "data   = ";
  for (auto b : data_tx) {
    std::cout << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
  }
  std::cout << std::dec << "\n";

  // Build synthetic type2 frame and decode.
  const auto q_u16 = float_to_uint(1.15, ranges.q_min, ranges.q_max, 16);
  const auto dq_u16 = float_to_uint(0.75, ranges.dq_min, ranges.dq_max, 16);
  const auto tau_u16 = float_to_uint(2.20, ranges.tau_min, ranges.tau_max, 16);
  const uint16_t temp_x10 = 365;
  std::array<uint8_t, 8> rx_data = {
      static_cast<uint8_t>(q_u16 >> 8),   static_cast<uint8_t>(q_u16 & 0xFF), static_cast<uint8_t>(dq_u16 >> 8),
      static_cast<uint8_t>(dq_u16 & 0xFF), static_cast<uint8_t>(tau_u16 >> 8), static_cast<uint8_t>(tau_u16 & 0xFF),
      static_cast<uint8_t>(temp_x10 >> 8), static_cast<uint8_t>(temp_x10 & 0xFF),
  };
  const uint32_t can_id_rx = build_can_id(2, 0x00FD, cmd.motor_id);
  auto fb = decode_type2(can_id_rx, rx_data, ranges);

  std::cout << "[TYPE2 RX]\n";
  std::cout << "can_id = 0x" << std::hex << std::uppercase << can_id_rx << std::dec << "\n";
  std::cout << "decoded: motor_id=" << static_cast<int>(fb.motor_id) << ", q=" << fb.q << ", dq=" << fb.dq
            << ", tau=" << fb.tau << ", temp=" << fb.temp_c << "C\n";

  return 0;
}

