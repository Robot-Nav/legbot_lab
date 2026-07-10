#pragma once

// 串口帧封装/解析器：支持灵足 USB-CAN 的 45 54 头、0D 0A 尾格式，以及串口收发重连。

#include "lingzu_serial.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace serial_dds_gateway {

// 灵足串口帧结构：按 USB-CAN 格式组织，不含帧头帧尾。
struct SerialFrame {
  uint8_t channel{0};              // 通道字节
  uint8_t frame_type{kLingzuCanExtendedFrame};  // 通信类型：01 标准帧 / 02 扩展帧 / 03 使能 / 04 失能
  uint16_t id_field{0};            // 扩展帧：控制域；标准帧：固定为 0
  uint8_t master_id{0xFD};         // 扩展帧：主/源 CAN ID；标准帧：CAN ID 字节
  std::vector<uint8_t> data;       // 数据域，长度 0..8
};

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
  // 写单帧；失败后返回 false，网关层继续运行。
  bool WriteFrame(const SerialFrame& frame);
  // 非阻塞读取并拆帧。
  std::vector<SerialFrame> ReadAvailableFrames();

  // 不打开串口，仅对帧做字节编解码（供单元测试）。
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
  std::vector<uint8_t> rx_buf_;  // 接收字节缓冲区
  mutable std::mutex io_mutex_;

  std::vector<SerialFrame> ExtractFrames();
};

}  // namespace serial_dds_gateway
