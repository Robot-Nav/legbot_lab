#include "joint_motor_bias.hpp"
#include "lingzu_motor_protocol.hpp"
#include "lingzu_serial.hpp"
#include "motor_map.hpp"
#include "serial_framer.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace serial_dds_gateway;

namespace {

bool BytesEq(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  return a == b;
}

void PrintHex(const std::vector<uint8_t>& v) {
  for (auto b : v) {
    std::cout << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(b) << ' ';
  }
  std::cout << std::dec << '\n';
}

}  // namespace

int main() {
  int failures = 0;

  // Golden capture from Lingzu RS02 USB-CAN serial protocol:
  // ET + channel + frame_type + id_field + master_id + dlc + data + CRLF.
  const std::vector<uint8_t> kGolden = {
      0x45, 0x54, 0x01, 0x02, 0x00, 0x20, 0xFD, 0x08, 0xA3, 0x5B, 0x7F,
      0xAC, 0x7F, 0xFF, 0x01, 0x22, 0x0D, 0x0A,
  };
  const std::vector<uint8_t> kStandardZero = {
      0x45, 0x54, 0x00, 0x01, 0x00, 0x00, 0x20, 0x08, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x0D, 0x0A,
  };
  const std::vector<uint8_t> kEnable = {
      0x45, 0x54, 0x00, 0x03, 0x00, 0xFD, 0x7F, 0x08, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x0D, 0x0A,
  };
  const std::vector<uint8_t> kDisable = {
      0x45, 0x54, 0x00, 0x04, 0x00, 0xFD, 0x7F, 0x08, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x0D, 0x0A,
  };
  const std::vector<uint8_t> kClearFault = {
      0x45, 0x54, 0x00, 0x04, 0x00, 0xFD, 0x7F, 0x08, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x0D, 0x0A,
  };

  // 1) Parse golden buffer as USB-CAN fields, not as a 32-bit CAN ID.
  auto buf = kGolden;
  const auto parsed = SerialFramer::ParseBuffer(buf);
  if (parsed.size() != 1) {
    std::cerr << "[FAIL] parse frame count " << parsed.size() << '\n';
    ++failures;
  } else {
    const auto& p = parsed.front();
    if (p.channel != 1 || p.frame_type != kLingzuCanExtendedFrame || p.id_field != 0x0020 || p.master_id != 0xFD ||
        p.data.size() != 8) {
      std::cerr << "[FAIL] parsed fields\n";
      ++failures;
    } else {
      std::cout << "[PASS] parse golden -> CH1 motor=32(0x20) master=0xFD\n";
    }
  }

  // 2) Decode payload as type1 motion-control command.
  const auto cmd = DecodeType1SerialFrame(parsed.front());
  auto close = [](double a, double b, double eps) { return std::fabs(a - b) <= eps; };
  if (cmd.motor_id != 0x20 || !close(cmd.q, 3.4713, 0.001) || !close(cmd.dq, -0.1121, 0.001) ||
      !close(cmd.kp, 249.996, 0.01) || !close(cmd.kd, 0.0221, 0.001)) {
    std::cerr << "[FAIL] type1 decode values\n";
    ++failures;
  } else {
    std::cout << "[PASS] type1 payload decodes q/dq/kp/kd\n";
  }

  // 3) Re-encode the decoded command and require byte-for-byte equality.
  const auto reencoded_frame = EncodeType1SerialFrame(1, 0xFD, cmd);
  const auto encoded = SerialFramer::EncodeBytes(reencoded_frame);
  if (!BytesEq(encoded, kGolden)) {
    std::cerr << "[FAIL] type1 re-encode mismatch\nexpected: ";
    PrintHex(kGolden);
    std::cerr << "got:      ";
    PrintHex(encoded);
    ++failures;
  } else {
    std::cout << "[PASS] type1 re-encodes golden bytes\n";
  }

  // 4) Feedback decoding uses the same serial envelope but type2 data semantics.
  const auto fb = DecodeType2SerialFrame(parsed.front());
  if (fb.motor_id != 0x20 || !close(fb.q, 3.4713, 0.001) || !close(fb.dq, -0.1121, 0.001) ||
      !close(fb.tau, -0.0003, 0.001) || !close(fb.temp_c, 29.0, 0.01)) {
    std::cerr << "[FAIL] type2 feedback decode values\n";
    ++failures;
  } else {
    std::cout << "[PASS] type2 payload decodes q/dq/tau/temp\n";
  }

  auto standard_buf = kStandardZero;
  const auto standard_parsed = SerialFramer::ParseBuffer(standard_buf);
  if (standard_parsed.size() != 1) {
    std::cerr << "[FAIL] standard frame parse count " << standard_parsed.size() << '\n';
    ++failures;
  } else {
    const auto& p = standard_parsed.front();
    if (p.channel != 0 || p.frame_type != kLingzuCanStandardFrame || p.id_field != 0x0000 || p.master_id != 0x20 ||
        p.data.size() != 8 || MotorIdFromSerialFrame(p) != 0x20) {
      std::cerr << "[FAIL] standard frame fields\n";
      ++failures;
    } else {
      std::cout << "[PASS] parse standard frame -> CH0 CAN ID=32(0x20)\n";
    }

    const auto standard_cmd = DecodeType1SerialFrame(p);
    if (!close(standard_cmd.tau, -17.0, 0.001)) {
      std::cerr << "[FAIL] standard frame torque field decode\n";
      ++failures;
    } else {
      std::cout << "[PASS] standard frame Byte4~5 decodes torque\n";
    }

    const auto encoded_standard = SerialFramer::EncodeBytes(EncodeType1StandardSerialFrame(0, standard_cmd));
    if (!BytesEq(encoded_standard, kStandardZero)) {
      std::cerr << "[FAIL] standard frame re-encode mismatch\nexpected: ";
      PrintHex(kStandardZero);
      std::cerr << "got:      ";
      PrintHex(encoded_standard);
      ++failures;
    } else {
      std::cout << "[PASS] standard frame re-encodes golden bytes\n";
    }

    Type1Command zero_tau_cmd{
        .motor_id = 0x20, .q = 0.0, .dq = 0.0, .kp = 0.0, .kd = 0.0, .tau = 0.0};
    const auto zero_tau_frame = EncodeType1StandardSerialFrame(0, zero_tau_cmd);
    if (zero_tau_frame.id_field != 0x8000 || zero_tau_frame.master_id != 0x20) {
      std::cerr << "[FAIL] physical zero torque standard frame field\n";
      ++failures;
    } else {
      std::cout << "[PASS] physical tau=0 encodes Byte4~5 as 0x8000\n";
    }
  }

  const auto enable_frame = BuildMotorEnableFrame(0, 0xFD, 0x7F);
  const auto encoded_enable = SerialFramer::EncodeBytes(enable_frame);
  if (!BytesEq(encoded_enable, kEnable)) {
    std::cerr << "[FAIL] enable frame mismatch\nexpected: ";
    PrintHex(kEnable);
    std::cerr << "got:      ";
    PrintHex(encoded_enable);
    ++failures;
  } else {
    std::cout << "[PASS] type3 enable frame encodes mode/master/motor ID\n";
  }

  const auto disable_frame = BuildMotorDisableFrame(0, 0xFD, 0x7F);
  const auto encoded_disable = SerialFramer::EncodeBytes(disable_frame);
  if (!BytesEq(encoded_disable, kDisable)) {
    std::cerr << "[FAIL] disable frame mismatch\nexpected: ";
    PrintHex(kDisable);
    std::cerr << "got:      ";
    PrintHex(encoded_disable);
    ++failures;
  } else {
    std::cout << "[PASS] type4 disable frame encodes zero data\n";
  }

  const auto clear_fault_frame = BuildMotorClearFaultFrame(0, 0xFD, 0x7F);
  const auto encoded_clear_fault = SerialFramer::EncodeBytes(clear_fault_frame);
  if (!BytesEq(encoded_clear_fault, kClearFault)) {
    std::cerr << "[FAIL] clear fault frame mismatch\nexpected: ";
    PrintHex(kClearFault);
    std::cerr << "got:      ";
    PrintHex(encoded_clear_fault);
    ++failures;
  } else {
    std::cout << "[PASS] type4 clear fault frame sets Byte0=1\n";
  }

  for (auto joint_name : kJointOrder) {
    const auto motor_it = kJointToCanId.find(joint_name);
    if (motor_it == kJointToCanId.end()) {
      std::cerr << "[FAIL] missing motor map for " << joint_name << "\n";
      ++failures;
      continue;
    }
    const auto motor_id = motor_it->second;
    const Type1Command joint_cmd{
        .motor_id = motor_id, .q = -1.5, .dq = 0.0, .kp = 10.0, .kd = 1.0, .tau = -3.0};
    const auto type1 = EncodeType1StandardSerialFrame(0, joint_cmd);
    if (type1.frame_type != kLingzuCanStandardFrame || type1.master_id != motor_id || type1.data.size() != 8) {
      std::cerr << "[FAIL] 12-motor type1 frame for " << joint_name << "\n";
      ++failures;
    }

    SerialFrame feedback{
        .channel = 1,
        .frame_type = kLingzuCanExtendedFrame,
        .id_field = MakeLingzuIdField(motor_id),
        .master_id = 0xFD,
        .data = type1.data,
    };
    const auto decoded = DecodeType2SerialFrame(feedback);
    if (decoded.motor_id != motor_id) {
      std::cerr << "[FAIL] 12-motor feedback ID for " << joint_name << "\n";
      ++failures;
    }
  }
  if (failures == 0) {
    std::cout << "[PASS] 12-motor type1/feedback ID mapping\n";
  }

  const std::unordered_map<uint8_t, MotorSerialBus> expected_bus = {
      {11, MotorSerialBus::A}, {21, MotorSerialBus::A}, {31, MotorSerialBus::A},
      {13, MotorSerialBus::A}, {23, MotorSerialBus::A}, {33, MotorSerialBus::A},
      {12, MotorSerialBus::B}, {22, MotorSerialBus::B}, {32, MotorSerialBus::B},
      {14, MotorSerialBus::B}, {24, MotorSerialBus::B}, {34, MotorSerialBus::B},
  };
  for (const auto& [motor_id, bus] : expected_bus) {
    if (MotorSerialBusForCanId(motor_id) != bus) {
      std::cerr << "[FAIL] motor serial bus mapping for " << static_cast<int>(motor_id) << "\n";
      ++failures;
    }
  }
  if (failures == 0) {
    std::cout << "[PASS] two motor serial bus split maps A=FR/RR and B=FL/RL\n";
  }

  // Fatu log session: motor raw q -> sign*scale -> bias -> model q
  constexpr std::array<float, 12> kLogMotorRawQ = {
      0.1398f, 1.2518f, -2.3295f, -0.1448f, -1.1719f, 2.3261f,
      -0.1206f, 1.1963f, -2.4293f, 0.1325f, -1.1472f, 2.3672f,
  };
  constexpr std::array<float, 12> kLogJointAfterSignGear = {
      0.1398f, 1.2518f, -1.1648f, -0.1448f, 1.1719f, -1.1631f,
      -0.1206f, 1.1963f, -1.2147f, 0.1325f, 1.1472f, -1.1836f,
  };
  const auto& bias = kFatuProneBiasFromLog;
  for (size_t i = 0; i < kJointOrder.size(); ++i) {
    const float joint = MotorToJointFromRaw(i, kLogMotorRawQ[i]);
    if (!close(joint, kLogJointAfterSignGear[i], 0.0005)) {
      std::cerr << "[FAIL] joint mapping sign+gear idx " << i << " got " << joint
                << " want " << kLogJointAfterSignGear[i] << "\n";
      ++failures;
    }
    const float q_model = JointToModelWithBias(i, kLogMotorRawQ[i], bias);
    const float q_model_ref = kLogJointAfterSignGear[i] - bias[i];
    if (!close(q_model, q_model_ref, 0.0005)) {
      std::cerr << "[FAIL] joint bias round-trip idx " << i << "\n";
      ++failures;
    }
    const float q_motor_back = ModelToMotorFromBias(i, q_model, bias);
    if (!close(q_motor_back, kLogMotorRawQ[i], 0.0005)) {
      std::cerr << "[FAIL] model->motor idx " << i << " got " << q_motor_back
                << " want " << kLogMotorRawQ[i] << "\n";
      ++failures;
    }
  }
  if (failures == 0) {
    std::cout << "[PASS] fatu joint map (log session sign+gear/bias round-trip)\n";
  }

  if (failures == 0) {
    std::cout << "\nAll lingzu frame checks passed.\n";
    return 0;
  }
  std::cerr << "\n" << failures << " check(s) failed.\n";
  return 1;
}
