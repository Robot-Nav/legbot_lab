#pragma once

#include "lingzu_serial.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace serial_dds_gateway {

// 灵足 USB-CAN 串口帧语义结构。
struct SerialFrame {
  uint8_t channel{0};                         // 通道字节
  uint8_t frame_type{kLingzuCanExtendedFrame}; // 帧类型：01 标准 / 02 扩展 / 03 使能 / 04 失能
  uint16_t id_field{0};                       // 扩展帧：ID/控制字段；标准帧：承载 16 位数据。
  uint8_t master_id{0xFD};                    // 扩展帧：主机/源地址；标准帧：CAN ID 字节。
  std::vector<uint8_t> data;                  // 有效载荷，0..8 字节
};

// 串口成帧器：负责打开串口、按 header/tail 打包/拆包、写失败时自动重开。
class SerialFramer {
 public:
  SerialFramer(std::string port, int baudrate, std::array<uint8_t, 2> header = kLingzuUsbHeader,
               std::array<uint8_t, 2> tail = kLingzuUsbTail);
  ~SerialFramer();

  SerialFramer(const SerialFramer&) = delete;
  SerialFramer& operator=(const SerialFramer&) = delete;

  bool IsOpen() const;
  const std::string& port() const { return port_; }
  void Close();
  // 写帧；重试耗尽后返回 false，网关侧仍继续运行。
  bool WriteFrame(const SerialFrame& frame);
  std::vector<SerialFrame> ReadAvailableFrames();

  // 无需打开串口即可使用的编解码接口（单元测试用）。
  static std::vector<uint8_t> EncodeBytes(const SerialFrame& frame,
                                          std::array<uint8_t, 2> header = kLingzuUsbHeader,
                                          std::array<uint8_t, 2> tail = kLingzuUsbTail);
  static std::vector<SerialFrame> ParseBuffer(std::vector<uint8_t>& rx_buf,
                                              std::array<uint8_t, 2> header = kLingzuUsbHeader,
                                              std::array<uint8_t, 2> tail = kLingzuUsbTail);

 private:
  bool OpenPortLocked();
  bool ReopenPortLocked();
  bool WritePayloadLocked(const std::vector<uint8_t>& payload);
  std::vector<SerialFrame> ReadAvailableFramesLocked();

  std::string port_;
  int baudrate_{0};
  int fd_{-1};
  std::array<uint8_t, 2> header_{};
  std::array<uint8_t, 2> tail_{};
  std::vector<uint8_t> rx_buf_;
  mutable std::mutex io_mutex_;

  std::vector<SerialFrame> ExtractFrames();
};

}  // 命名空间 serial_dds_gateway
