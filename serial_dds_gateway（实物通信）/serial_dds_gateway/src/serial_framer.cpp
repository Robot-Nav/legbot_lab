#include "serial_framer.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace serial_dds_gateway {

namespace {
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
      throw std::runtime_error("Unsupported baudrate for termios");
  }
}

constexpr int kMaxWriteAttempts = 5;
constexpr auto kEagainBackoff = std::chrono::microseconds(200);
constexpr auto kRetryBackoff = std::chrono::milliseconds(2);
constexpr auto kReopenDelay = std::chrono::milliseconds(50);

bool IsRecoverableIoError(int err) {
  return err == EIO || err == ENODEV || err == EBADF;
}
}  // namespace

SerialFramer::SerialFramer(std::string port, int baudrate, std::array<uint8_t, 2> header, std::array<uint8_t, 2> tail)
    : port_(std::move(port)), baudrate_(baudrate), header_(header), tail_(tail) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  if (!OpenPortLocked()) {
    throw std::runtime_error("open serial failed: " + port_ + ": " + std::string(std::strerror(errno)));
  }
}

SerialFramer::~SerialFramer() { Close(); }

bool SerialFramer::IsOpen() const {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return fd_ >= 0;
}

void SerialFramer::Close() {
  std::lock_guard<std::mutex> lock(io_mutex_);
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialFramer::OpenPortLocked() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }

  fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    return false;
  }

  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  cfmakeraw(&tty);
  const auto baud = ToTermiosBaud(baudrate_);
  cfsetispeed(&tty, baud);
  cfsetospeed(&tty, baud);

  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CRTSCTS;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  return true;
}

bool SerialFramer::ReopenPortLocked() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  std::this_thread::sleep_for(kReopenDelay);
  if (!OpenPortLocked()) {
    return false;
  }
  std::cerr << "[INFO] serial reopened on " << port_ << "\n";
  return true;
}

std::vector<uint8_t> SerialFramer::EncodeBytes(const SerialFrame& frame, std::array<uint8_t, 2> header,
                                              std::array<uint8_t, 2> tail) {
  if (frame.data.size() > 8) {
    throw std::runtime_error("serial frame data > 8 bytes");
  }
  std::vector<uint8_t> payload;
  payload.reserve(2 + 1 + 4 + 1 + frame.data.size() + 2);
  payload.push_back(header[0]);
  payload.push_back(header[1]);
  payload.push_back(frame.channel);
  payload.push_back(frame.frame_type);
  payload.push_back(static_cast<uint8_t>((frame.id_field >> 8) & 0xFF));
  payload.push_back(static_cast<uint8_t>(frame.id_field & 0xFF));
  payload.push_back(frame.master_id);
  payload.push_back(static_cast<uint8_t>(frame.data.size()));
  payload.insert(payload.end(), frame.data.begin(), frame.data.end());
  payload.push_back(tail[0]);
  payload.push_back(tail[1]);
  return payload;
}

bool SerialFramer::WritePayloadLocked(const std::vector<uint8_t>& payload) {
  if (payload.empty()) {
    return true;
  }

  size_t off = 0;
  int last_errno = 0;

  for (int attempt = 0; attempt < kMaxWriteAttempts; ++attempt) {
    if (fd_ < 0 && !ReopenPortLocked()) {
      break;
    }

    while (off < payload.size()) {
      const ssize_t n = ::write(fd_, payload.data() + off, payload.size() - off);
      if (n > 0) {
        off += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        std::this_thread::sleep_for(kEagainBackoff);
        continue;
      }
      if (n < 0) {
        last_errno = errno;
        if (IsRecoverableIoError(last_errno)) {
          ReopenPortLocked();
        }
      }
      break;
    }

    if (off >= payload.size()) {
      return true;
    }

    if (attempt + 1 < kMaxWriteAttempts) {
      std::this_thread::sleep_for(kRetryBackoff);
    }
  }

  std::cerr << "[WARN] serial write failed on " << port_;
  if (last_errno != 0) {
    std::cerr << ": " << std::strerror(last_errno);
  }
  std::cerr << " (retried " << kMaxWriteAttempts << " times)\n";
  return false;
}

bool SerialFramer::WriteFrame(const SerialFrame& frame) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return WritePayloadLocked(EncodeBytes(frame, header_, tail_));
}

std::vector<SerialFrame> SerialFramer::ReadAvailableFramesLocked() {
  if (fd_ < 0 && !ReopenPortLocked()) {
    return ExtractFrames();
  }

  uint8_t tmp[1024];
  while (true) {
    const ssize_t n = ::read(fd_, tmp, sizeof(tmp));
    if (n > 0) {
      rx_buf_.insert(rx_buf_.end(), tmp, tmp + n);
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      break;
    }
    if (n < 0 && IsRecoverableIoError(errno)) {
      ReopenPortLocked();
    }
    break;
  }
  return ExtractFrames();
}

std::vector<SerialFrame> SerialFramer::ReadAvailableFrames() {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return ReadAvailableFramesLocked();
}

std::vector<SerialFrame> SerialFramer::ParseBuffer(std::vector<uint8_t>& rx_buf, std::array<uint8_t, 2> header,
                                                   std::array<uint8_t, 2> tail) {
  std::vector<SerialFrame> out;
  constexpr size_t kMin = 2 + 1 + 4 + 1 + 2;

  while (rx_buf.size() >= kMin) {
    auto start = rx_buf.end();
    for (auto it = rx_buf.begin(); it + 1 < rx_buf.end(); ++it) {
      if ((*it == header[0]) && (*(it + 1) == header[1])) {
        start = it;
        break;
      }
    }
    if (start == rx_buf.end()) {
      rx_buf.clear();
      break;
    }
    if (start != rx_buf.begin()) {
      rx_buf.erase(rx_buf.begin(), start);
      if (rx_buf.size() < kMin) break;
    }

    const uint8_t dlc = rx_buf[7];
    if (dlc > 8) {
      rx_buf.erase(rx_buf.begin());
      continue;
    }
    const size_t frame_len = 2 + 1 + 4 + 1 + dlc + 2;
    if (rx_buf.size() < frame_len) break;

    if (rx_buf[frame_len - 2] != tail[0] || rx_buf[frame_len - 1] != tail[1]) {
      rx_buf.erase(rx_buf.begin());
      continue;
    }

    SerialFrame f;
    f.channel = rx_buf[2];
    f.frame_type = rx_buf[3];
    f.id_field = static_cast<uint16_t>((static_cast<uint16_t>(rx_buf[4]) << 8) | rx_buf[5]);
    f.master_id = rx_buf[6];
    f.data.assign(rx_buf.begin() + 8, rx_buf.begin() + 8 + dlc);
    out.push_back(std::move(f));
    rx_buf.erase(rx_buf.begin(), rx_buf.begin() + frame_len);
  }
  return out;
}

std::vector<SerialFrame> SerialFramer::ExtractFrames() { return ParseBuffer(rx_buf_, header_, tail_); }

}  // namespace serial_dds_gateway
