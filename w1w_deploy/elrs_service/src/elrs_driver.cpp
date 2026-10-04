#include "elrs/elrs_driver.h"

#include <asm/termbits.h>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" int tcflush(int fd, int queue_selector);

namespace elrs_serial
{
namespace
{
constexpr uint8_t kRcChannelsPacked = 0x16;
constexpr uint8_t kRcFrameLength = 24;
constexpr std::size_t kRcFrameSize = 26;

bool is_crsf_address(uint8_t address)
{
    switch (address)
    {
    case 0x00:
    case 0x10:
    case 0x80:
    case 0xC0:
    case 0xC2:
    case 0xC4:
    case 0xC8:
    case 0xCA:
    case 0xCC:
    case 0xEA:
    case 0xEC:
    case 0xEE:
    case 0xEF:
        return true;
    default:
        return false;
    }
}

uint8_t crc8(const uint8_t* bytes, std::size_t len)
{
    uint8_t crc = 0;
    for (std::size_t i = 0; i < len; ++i)
    {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            crc = (crc & 0x80U) ? static_cast<uint8_t>((crc << 1U) ^ 0xD5U)
                                : static_cast<uint8_t>(crc << 1U);
        }
    }
    return crc;
}

bool valid_crc(const uint8_t* frame, std::size_t len)
{
    return len >= 4 && frame[len - 1] == crc8(frame + 2, len - 3);
}
} // namespace

bool parse_crsf_rc_frame(const uint8_t* frame, std::size_t len, ELRS_Data& data)
{
    if (frame == nullptr || len != kRcFrameSize || !is_crsf_address(frame[0]) ||
        frame[1] != kRcFrameLength || frame[2] != kRcChannelsPacked || !valid_crc(frame, len))
    {
        return false;
    }

    const uint8_t* rc = frame + 3;
    for (int channel = 0; channel < 16; ++channel)
    {
        const int bit_offset = channel * 11;
        const int byte_offset = bit_offset / 8;
        const int shift = bit_offset % 8;
        const uint32_t packed = static_cast<uint32_t>(rc[byte_offset]) |
                                (static_cast<uint32_t>(rc[byte_offset + 1]) << 8U) |
                                (byte_offset + 2 < 22
                                     ? static_cast<uint32_t>(rc[byte_offset + 2]) << 16U
                                     : 0U);
        data.channels[channel] = static_cast<uint16_t>((packed >> shift) & 0x07FFU);
    }
    data.frame_id = frame[2];
    return true;
}

std::vector<uint8_t> build_crsf_frame(
    uint8_t frame_type, const uint8_t* payload, std::size_t payload_size)
{
    if ((payload == nullptr && payload_size != 0) || payload_size > 60)
    {
        return {};
    }
    std::vector<uint8_t> frame(payload_size + 4);
    frame[0] = 0xC8; // CRSF flight-controller sync/address
    frame[1] = static_cast<uint8_t>(payload_size + 2); // type + payload + CRC
    frame[2] = frame_type;
    if (payload_size != 0)
    {
        std::memcpy(frame.data() + 3, payload, payload_size);
    }
    frame.back() = crc8(frame.data() + 2, payload_size + 1);
    return frame;
}

ElrsReceiver::ElrsReceiver(const std::string& device, int baudrate, Callback callback)
    : device_(device), baudrate_(baudrate), callback_(std::move(callback)), fd_(-1), running_(true)
{
    fd_ = open(device_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0)
    {
        throw std::runtime_error("failed to open ELRS serial device: " + device_);
    }

    struct termios2 tty{};
    if (ioctl(fd_, TCGETS2, &tty) != 0)
    {
        close(fd_);
        throw std::runtime_error("ELRS TCGETS2 failed");
    }
    tty.c_cflag &= ~CBAUD;
    tty.c_cflag |= BOTHER;
    tty.c_ispeed = static_cast<speed_t>(baudrate_);
    tty.c_ospeed = static_cast<speed_t>(baudrate_);
    tty.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS);
    tty.c_cflag |= CS8 | CREAD | CLOCAL;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHONL | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | IGNBRK | BRKINT | PARMRK |
                     ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~(OPOST | ONLCR);
    tty.c_cc[VTIME] = 0;
    tty.c_cc[VMIN] = 0;
    if (ioctl(fd_, TCSETS2, &tty) != 0)
    {
        close(fd_);
        throw std::runtime_error("ELRS TCSETS2 failed");
    }
    tcflush(fd_, 2); // TCIOFLUSH
    read_thread_ = std::thread(&ElrsReceiver::read_thread_func, this);
}

ElrsReceiver::~ElrsReceiver()
{
    stop();
}

void ElrsReceiver::stop()
{
    if (running_.exchange(false))
    {
        if (read_thread_.joinable())
        {
            read_thread_.join();
        }
        if (fd_ >= 0)
        {
            close(fd_);
            fd_ = -1;
        }
    }
}

bool ElrsReceiver::write_crsf_frame(
    uint8_t frame_type, const uint8_t* payload, std::size_t payload_size)
{
    const auto frame = build_crsf_frame(frame_type, payload, payload_size);
    if (frame.empty()) return false;
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (fd_ < 0) return false;
    return write(fd_, frame.data(), frame.size()) == static_cast<ssize_t>(frame.size());
}

void ElrsReceiver::read_thread_func()
{
    uint8_t buffer[512]{};
    std::size_t buffered = 0;
    while (running_.load())
    {
        const ssize_t count = read(fd_, buffer + buffered, sizeof(buffer) - buffered);
        if (count > 0)
        {
            buffered += static_cast<std::size_t>(count);
        }
        else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            break;
        }
        else if (count <= 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::size_t consumed = 0;
        while (buffered - consumed >= 2)
        {
            const std::size_t body_len = buffer[consumed + 1];
            const std::size_t frame_len = body_len + 2;
            if (!is_crsf_address(buffer[consumed]) || body_len < 2 || body_len > 62)
            {
                ++consumed;
                continue;
            }
            if (buffered - consumed < frame_len)
            {
                break;
            }
            if (!valid_crc(buffer + consumed, frame_len))
            {
                ++consumed;
                continue;
            }
            if (buffer[consumed + 2] == kRcChannelsPacked)
            {
                ELRS_Data data{};
                if (parse_crsf_rc_frame(buffer + consumed, frame_len, data) && callback_)
                {
                    callback_(data);
                }
            }
            consumed += frame_len;
        }
        if (consumed > 0)
        {
            buffered -= consumed;
            std::memmove(buffer, buffer + consumed, buffered);
        }
        if (buffered > sizeof(buffer) * 3 / 4)
        {
            const std::size_t discard = buffered / 2;
            buffered -= discard;
            std::memmove(buffer, buffer + discard, buffered);
        }
    }
}

} // namespace elrs_serial
