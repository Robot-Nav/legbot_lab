#pragma once

#include "lingzu_serial.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace serial_dds_gateway {

struct SerialFrame {
  uint8_t channel{0};
  uint8_t frame_type{kLingzuCanExtendedFrame};
  uint16_t id_field{0};  // Extended frame ID/control field; standard frame keeps this at 0.
  uint8_t master_id{0xFD};  // Extended: master/source ID. Standard: CAN ID byte per RS02 USB-CAN format.
  std::vector<uint8_t> data;  // 0..8
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
  // Returns false after retries exhausted (gateway keeps running).
  bool WriteFrame(const SerialFrame& frame);
  std::vector<SerialFrame> ReadAvailableFrames();

  // Encode/decode without opening a port (unit tests).
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

}  // namespace serial_dds_gateway
