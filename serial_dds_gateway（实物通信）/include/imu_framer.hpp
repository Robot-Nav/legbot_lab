#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace serial_dds_gateway {

// IMU 串口帧载荷（去掉帧头后）：yaw、pitch、roll、gz、gy、gx，均为 float32 小端。
struct ImuSample {
  double yaw{0.0};
  double pitch{0.0};
  double roll{0.0};
  double gx{0.0};
  double gy{0.0};
  double gz{0.0};
};

// 四元数，w,x,y,z 顺序。
struct Quaternion {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

// 欧拉角 yaw/pitch/roll（ZYX 顺序）转四元数。
Quaternion EulerYprToQuaternion(double yaw, double pitch, double roll);

// IMU 串口成帧器：打开串口、读取原始字节、按 EB 90 A5 FF ... 80 7F 格式拆帧并校验 CRC16-Modbus。
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
  std::vector<uint8_t> rx_buf_;
};

}  // 命名空间 serial_dds_gateway
