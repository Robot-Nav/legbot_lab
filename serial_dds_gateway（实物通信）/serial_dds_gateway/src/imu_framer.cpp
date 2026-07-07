#include "imu_framer.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cmath>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace serial_dds_gateway {

namespace {

constexpr std::array<uint8_t, 4> kImuHeader = {0xEB, 0x90, 0xA5, 0xFF};
constexpr std::array<uint8_t, 2> kImuTail = {0x80, 0x7F};
constexpr size_t kImuChannels = 6;
constexpr size_t kBytesPerFloat = 4;
constexpr size_t kImuDataBytes = kImuChannels * kBytesPerFloat;
constexpr size_t kImuFrameLen = kImuHeader.size() + kImuDataBytes + 2 + kImuTail.size();

speed_t ToTermiosBaud(int baudrate) {
  switch (baudrate) {
    case 115200:
      return B115200;
    case 230400:
      return B230400;
    case 460800:
      return B460800;
    case 921600:
      return B921600;
#ifdef B2000000
    case 2000000:
      return B2000000;
#endif
    default:
      throw std::runtime_error("Unsupported IMU baudrate for termios");
  }
}

float ReadFloatLe(const std::vector<uint8_t>& bytes, size_t offset) {
  uint32_t raw = static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                 (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                 (static_cast<uint32_t>(bytes[offset + 3]) << 24);
  float value = 0.0F;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

bool HasHeaderAt(const std::vector<uint8_t>& bytes, size_t offset) {
  if (offset + kImuHeader.size() > bytes.size()) {
    return false;
  }
  for (size_t i = 0; i < kImuHeader.size(); ++i) {
    if (bytes[offset + i] != kImuHeader[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

Quaternion EulerYprToQuaternion(double yaw, double pitch, double roll) {
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);

  Quaternion q;
  q.w = cy * cp * cr + sy * sp * sr;
  q.x = cy * cp * sr - sy * sp * cr;
  q.y = sy * cp * sr + cy * sp * cr;
  q.z = sy * cp * cr - cy * sp * sr;
  return q;
}

ImuFramer::ImuFramer(std::string port, int baudrate) {
  fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    throw std::runtime_error("open IMU serial failed: " + std::string(std::strerror(errno)));
  }

  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    Close();
    throw std::runtime_error("IMU tcgetattr failed");
  }

  cfmakeraw(&tty);
  const auto baud = ToTermiosBaud(baudrate);
  cfsetispeed(&tty, baud);
  cfsetospeed(&tty, baud);

  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CRTSCTS;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    Close();
    throw std::runtime_error("IMU tcsetattr failed");
  }
}

ImuFramer::~ImuFramer() { Close(); }

void ImuFramer::Close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

std::vector<ImuSample> ImuFramer::ReadAvailableSamples() {
  uint8_t tmp[1024];
  while (true) {
    const auto n = ::read(fd_, tmp, sizeof(tmp));
    if (n > 0) {
      rx_buf_.insert(rx_buf_.end(), tmp, tmp + n);
    } else {
      break;
    }
  }
  return ParseBuffer(rx_buf_);
}

uint16_t ImuFramer::Crc16Modbus(const std::vector<uint8_t>& data) {
  uint16_t crc = 0xFFFF;
  for (const auto byte : data) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      if ((crc & 0x0001) != 0) {
        crc = static_cast<uint16_t>((crc >> 1) ^ 0xA001);
      } else {
        crc = static_cast<uint16_t>(crc >> 1);
      }
    }
  }
  return crc;
}

std::vector<ImuSample> ImuFramer::ParseBuffer(std::vector<uint8_t>& rx_buf) {
  std::vector<ImuSample> out;

  while (rx_buf.size() >= kImuHeader.size()) {
    size_t start = rx_buf.size();
    for (size_t i = 0; i + kImuHeader.size() <= rx_buf.size(); ++i) {
      if (HasHeaderAt(rx_buf, i)) {
        start = i;
        break;
      }
    }

    if (start == rx_buf.size()) {
      rx_buf.clear();
      break;
    }
    if (start != 0) {
      rx_buf.erase(rx_buf.begin(), rx_buf.begin() + static_cast<std::ptrdiff_t>(start));
    }
    if (rx_buf.size() < kImuFrameLen) {
      break;
    }

    if (rx_buf[kImuFrameLen - 2] != kImuTail[0] || rx_buf[kImuFrameLen - 1] != kImuTail[1]) {
      rx_buf.erase(rx_buf.begin());
      continue;
    }

    const size_t crc_offset = kImuHeader.size() + kImuDataBytes;
    const uint16_t received_crc =
        static_cast<uint16_t>(rx_buf[crc_offset] | (static_cast<uint16_t>(rx_buf[crc_offset + 1]) << 8));
    const std::vector<uint8_t> crc_data(rx_buf.begin(), rx_buf.begin() + static_cast<std::ptrdiff_t>(crc_offset));
    if (received_crc != Crc16Modbus(crc_data)) {
      rx_buf.erase(rx_buf.begin());
      continue;
    }

    const size_t data_offset = kImuHeader.size();
    // Wire order: yaw, pitch, roll, gz, gy, gx (float32 LE each).
    ImuSample sample;
    sample.yaw = ReadFloatLe(rx_buf, data_offset + 0 * kBytesPerFloat);
    sample.pitch = ReadFloatLe(rx_buf, data_offset + 1 * kBytesPerFloat);
    sample.roll = ReadFloatLe(rx_buf, data_offset + 2 * kBytesPerFloat);
    sample.gz = ReadFloatLe(rx_buf, data_offset + 3 * kBytesPerFloat);
    sample.gy = ReadFloatLe(rx_buf, data_offset + 4 * kBytesPerFloat);
    sample.gx = ReadFloatLe(rx_buf, data_offset + 5 * kBytesPerFloat);
    out.push_back(sample);
    rx_buf.erase(rx_buf.begin(), rx_buf.begin() + static_cast<std::ptrdiff_t>(kImuFrameLen));
  }

  return out;
}

}  // namespace serial_dds_gateway
