#include "motor_control/w190_canfd.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

constexpr uint32_t kHostCanId = 0x00;
constexpr std::size_t kJointCommandSize = 10;
constexpr std::size_t kJointCount = 6;

uint64_t steady_now_ms()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint16_t float_to_uint(float value, float minimum, float maximum)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument("W190 command contains a non-finite value");
    }
    value = std::clamp(value, minimum, maximum);
    if (value == 0.0F && minimum < 0.0F && maximum > 0.0F)
    {
        return 0x8000;
    }
    return static_cast<uint16_t>(std::lround(
        (value - minimum) * 65535.0F / (maximum - minimum)));
}

float uint_to_float(uint16_t raw, float minimum, float maximum)
{
    if (raw == 0x8000 && minimum < 0.0F && maximum > 0.0F)
    {
        return 0.0F;
    }
    return minimum + static_cast<float>(raw) * (maximum - minimum) / 65535.0F;
}

void put_u16(std::array<uint8_t, 64>& payload, std::size_t offset, uint16_t value)
{
    payload[offset] = static_cast<uint8_t>(value & 0xFF);
    payload[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
}

uint16_t get_u16(const uint8_t* data, std::size_t offset)
{
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1]) << 8);
}

std::array<uint8_t, kJointCommandSize> encode_motion(const W190Command& command)
{
    const std::array<uint16_t, 5> values = {
        float_to_uint(command.position, -12.5F, 12.5F),
        float_to_uint(command.velocity, -150.0F, 150.0F),
        float_to_uint(command.kp, 0.0F, 500.0F),
        float_to_uint(command.kd, 0.0F, 5.0F),
        float_to_uint(command.torque, -50.0F, 50.0F),
    };
    std::array<uint8_t, kJointCommandSize> data{};
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        data[i * 2] = static_cast<uint8_t>(values[i] & 0xFF);
        data[i * 2 + 1] = static_cast<uint8_t>((values[i] >> 8) & 0xFF);
    }
    return data;
}

void copy_joint_command(std::array<uint8_t, 64>& payload,
                        std::size_t motor_id,
                        const std::array<uint8_t, kJointCommandSize>& command)
{
    const std::size_t offset = 2 + (motor_id - 1) * kJointCommandSize;
    std::copy(command.begin(), command.end(), payload.begin() + offset);
}

} // namespace

W190CanFd::W190CanFd(const std::string& interface_name, uint16_t msg_id)
    : msg_id_(msg_id)
{
    socket_fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0)
    {
        throw std::runtime_error(std::string("Failed to open CAN FD socket: ") + std::strerror(errno));
    }

    int enable_fd = 1;
    if (setsockopt(socket_fd_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable_fd, sizeof(enable_fd)) < 0)
    {
        ::close(socket_fd_);
        throw std::runtime_error(std::string("Failed to enable CAN FD frames: ") + std::strerror(errno));
    }

    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, interface_name.c_str(), IFNAMSIZ - 1);
    if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0)
    {
        ::close(socket_fd_);
        throw std::runtime_error(std::string("Failed to resolve CAN FD interface: ") + std::strerror(errno));
    }

    sockaddr_can address{};
    address.can_family = AF_CAN;
    address.can_ifindex = ifr.ifr_ifindex;
    if (bind(socket_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0)
    {
        ::close(socket_fd_);
        throw std::runtime_error(std::string("Failed to bind CAN FD socket: ") + std::strerror(errno));
    }

    const int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags >= 0)
    {
        fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
    }
}

W190CanFd::~W190CanFd()
{
    if (socket_fd_ >= 0)
    {
        ::close(socket_fd_);
    }
}

void W190CanFd::sendPayload(const std::array<uint8_t, 64>& payload)
{
    struct canfd_frame frame{};
    frame.can_id = kHostCanId;
    frame.len = 64;
    frame.flags = CANFD_BRS;
    std::memcpy(frame.data, payload.data(), payload.size());

    const ssize_t written = ::write(socket_fd_, &frame, sizeof(frame));
    if (written == static_cast<ssize_t>(sizeof(frame)))
    {
        return;
    }
    if (written < 0 && (errno == ENOBUFS || errno == EAGAIN || errno == EWOULDBLOCK))
    {
        throw std::runtime_error("CAN FD transmit queue is full (no bus ACK)");
    }
    throw std::runtime_error(std::string("CAN FD write failed: ") + std::strerror(errno));
}

void W190CanFd::sendSpecial(uint8_t command_code)
{
    std::array<uint8_t, 64> payload{};
    put_u16(payload, 0, msg_id_);
    const auto neutral = encode_motion(W190Command{});
    for (std::size_t id = 1; id <= kJointCount; ++id)
    {
        copy_joint_command(payload, id, neutral);
    }
    std::array<uint8_t, kJointCommandSize> special{};
    special.fill(0xFF);
    special.back() = command_code;
    copy_joint_command(payload, 1, special);
    copy_joint_command(payload, 2, special);
    sendPayload(payload);
}

void W190CanFd::enable()
{
    sendSpecial(0xFC);
}

void W190CanFd::disable()
{
    sendSpecial(0xFD);
}

void W190CanFd::sendMotion(const std::array<W190Command, 2>& commands)
{
    std::array<uint8_t, 64> payload{};
    put_u16(payload, 0, msg_id_);
    const auto neutral = encode_motion(W190Command{});
    for (std::size_t id = 1; id <= kJointCount; ++id)
    {
        copy_joint_command(payload, id, neutral);
    }
    copy_joint_command(payload, 1, encode_motion(commands[0]));
    copy_joint_command(payload, 2, encode_motion(commands[1]));
    sendPayload(payload);
}

bool W190CanFd::receive(W190Feedback& feedback, std::chrono::milliseconds timeout)
{
    struct pollfd poll_fd{};
    poll_fd.fd = socket_fd_;
    poll_fd.events = POLLIN;
    const int result = ::poll(&poll_fd, 1, static_cast<int>(timeout.count()));
    if (result <= 0)
    {
        if (result < 0 && errno != EINTR)
        {
            throw std::runtime_error(std::string("CAN FD poll failed: ") + std::strerror(errno));
        }
        return false;
    }

    struct canfd_frame frame{};
    const ssize_t bytes = ::read(socket_fd_, &frame, sizeof(frame));
    if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    {
        return false;
    }
    if (bytes != static_cast<ssize_t>(sizeof(frame)) || frame.len != 12)
    {
        return false;
    }

    const uint32_t can_id = frame.can_id & CAN_SFF_MASK;
    if (can_id < 1 || can_id > 2 || (frame.can_id & CAN_EFF_FLAG) != 0)
    {
        return false;
    }

    feedback.motor_id = static_cast<uint8_t>(can_id);
    feedback.msg_id = get_u16(frame.data, 0);
    feedback.position = uint_to_float(get_u16(frame.data, 2), -12.5F, 12.5F);
    feedback.velocity = uint_to_float(get_u16(frame.data, 4), -150.0F, 150.0F);
    feedback.torque = uint_to_float(get_u16(frame.data, 6), -50.0F, 50.0F);
    feedback.temperature = static_cast<float>(frame.data[8]) - 40.0F;
    feedback.status = frame.data[9];
    feedback.voltage = static_cast<float>(frame.data[10]);
    feedback.last_rx_ms = steady_now_ms();
    return true;
}
