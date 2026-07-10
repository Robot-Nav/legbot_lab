#pragma once

// IMU 串口帧解析：EB 90 A5 FF 头，6 个 float32 little-endian + CRC16-Modbus + 80 7F 尾。

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace serial_dds_gateway {

// IMU 采样值：串口顺序为 yaw/pitch/roll/gz/gy/gx。
struct ImuSample {
  double yaw{0.0};
  double pitch{0.0};
  double roll{0.0};
  double gx{0.0};
  double gy{0.0};
  double gz{0.0};
};

struct Quaternion {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

// ZYX 欧拉角（yaw-pitch-roll）转四元数。
Quaternion EulerYprToQuaternion(double yaw, double pitch, double roll);

class ImuFramer {
 public:
  ImuFramer(std::string port, int baudrate);
  ~ImuFramer();

  ImuFramer(const ImuFramer&) = delete;
  ImuFramer& operator=(const ImuFramer&) = delete;

  bool IsOpen() const { return fd_ >= 0; }
  void Close();
  std::vector<ImuSample> ReadAvailableSamples();

  static uint16_t Crc16Modbus(const std::vector<uint8_t>& data);
  static std::vector<ImuSample> ParseBuffer(std::vector<uint8_t>& rx_buf);

 private:
  int fd_{-1};
  std::vector<uint8_t> rx_buf_;  // 接收字节缓冲区
};

}  // namespace serial_dds_gateway
