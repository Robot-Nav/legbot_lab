#include "imu_framer.hpp"
#include "imu_gyro_filter.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace serial_dds_gateway;

namespace {

// 小端 float32 写入辅助函数。
void PushFloatLe(std::vector<uint8_t>& out, float value) {
  uint8_t bytes[4]{};
  std::memcpy(bytes, &value, sizeof(value));
  out.insert(out.end(), bytes, bytes + sizeof(bytes));
}

bool Close(double a, double b, double eps = 1e-5) { return std::fabs(a - b) <= eps; }

// 按 IMU 线序构造一帧：yaw, pitch, roll, gz, gy, gx。
std::vector<uint8_t> BuildFrame(float yaw, float pitch, float roll, float gz, float gy, float gx) {
  std::vector<uint8_t> frame = {0xEB, 0x90, 0xA5, 0xFF};
  PushFloatLe(frame, yaw);
  PushFloatLe(frame, pitch);
  PushFloatLe(frame, roll);
  PushFloatLe(frame, gz);
  PushFloatLe(frame, gy);
  PushFloatLe(frame, gx);
  const auto crc = ImuFramer::Crc16Modbus(frame);
  frame.push_back(static_cast<uint8_t>(crc & 0xFF));
  frame.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));
  frame.push_back(0x80);
  frame.push_back(0x7F);
  return frame;
}

}  // 命名空间

int main() {
  int failures = 0;

  // 测试帧解析：线序为 gz, gy, gx，解析后应分别对应 gx, gy, gz。
  auto rx = BuildFrame(0.1F, -0.2F, 0.3F, 3.0F, 2.0F, 1.0F);
  const auto samples = ImuFramer::ParseBuffer(rx);
  if (samples.size() != 1 || !rx.empty()) {
    std::cerr << "[FAIL] IMU frame parse count/buffer\n";
    ++failures;
  } else {
    const auto& s = samples.front();
    if (!Close(s.yaw, 0.1, 1e-6) || !Close(s.pitch, -0.2, 1e-6) || !Close(s.roll, 0.3, 1e-6) ||
        !Close(s.gx, 1.0) || !Close(s.gy, 2.0) || !Close(s.gz, 3.0)) {
      std::cerr << "[FAIL] IMU decoded float values\n";
      ++failures;
    } else {
      std::cout << "[PASS] IMU frame parses 6 little-endian float32 values\n";
    }
  }

  auto zero_q = EulerYprToQuaternion(0.0, 0.0, 0.0);
  if (!Close(zero_q.w, 1.0) || !Close(zero_q.x, 0.0) || !Close(zero_q.y, 0.0) || !Close(zero_q.z, 0.0)) {
    std::cerr << "[FAIL] zero Euler quaternion\n";
    ++failures;
  } else {
    std::cout << "[PASS] zero Euler angles produce identity quaternion\n";
  }

  auto yaw_q = EulerYprToQuaternion(M_PI / 2.0, 0.0, 0.0);
  const double root_half = std::sqrt(0.5);
  if (!Close(yaw_q.w, root_half) || !Close(yaw_q.x, 0.0) || !Close(yaw_q.y, 0.0) ||
      !Close(yaw_q.z, root_half)) {
    std::cerr << "[FAIL] yaw 90deg quaternion\n";
    ++failures;
  } else {
    std::cout << "[PASS] yaw/pitch/roll use ZYX quaternion convention\n";
  }

  // 测试 CRC 错误帧应被拒绝。
  auto bad_crc = BuildFrame(0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
  bad_crc[10] ^= 0xFF;
  const auto bad_samples = ImuFramer::ParseBuffer(bad_crc);
  if (!bad_samples.empty()) {
    std::cerr << "[FAIL] bad CRC frame accepted\n";
    ++failures;
  } else {
    std::cout << "[PASS] IMU parser rejects bad CRC frames\n";
  }

  // 测试陀螺零偏标定：10 个相同样本后零偏应等于该值，再次 Apply 输出经死区后为 0。
  {
    ImuGyroFilter filter;
    filter.Configure(true, 0.0, 0.02, 10);
    for (int i = 0; i < 10; ++i) {
      filter.Apply(0.1F, -0.05F, 0.08F);
    }
    if (!filter.calibration_done()) {
      std::cerr << "[FAIL] gyro filter calibration did not finish\n";
      ++failures;
    } else {
      const auto bias = filter.bias();
      if (!Close(bias[0], 0.1) || !Close(bias[1], -0.05) || !Close(bias[2], 0.08)) {
        std::cerr << "[FAIL] gyro filter bias estimate\n";
        ++failures;
      } else {
        const auto out = filter.Apply(0.1F, -0.05F, 0.08F);
        if (!Close(out[0], 0.0) || !Close(out[1], 0.0) || !Close(out[2], 0.0)) {
          std::cerr << "[FAIL] gyro filter bias-subtracted output\n";
          ++failures;
        } else {
          std::cout << "[PASS] IMU gyro bias calibration and deadzone\n";
        }
      }
    }
  }

  if (failures == 0) {
    std::cout << "\nAll IMU frame checks passed.\n";
    return 0;
  }
  std::cerr << "\n" << failures << " IMU check(s) failed.\n";
  return 1;
}
