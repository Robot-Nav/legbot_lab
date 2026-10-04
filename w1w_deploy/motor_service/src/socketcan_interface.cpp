#include "motor_control/socketcan_interface.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>

namespace {
uint32_t build_can_id(const CanFrame& frame)
{
    uint32_t can_id = frame.id & CAN_EFF_MASK;
    if (frame.is_extended)
    {
        can_id |= CAN_EFF_FLAG;
    }
    if (frame.is_rtr)
    {
        can_id |= CAN_RTR_FLAG;
    }
    return can_id;
}

void populate_frame_from_can(const struct can_frame& can_frame, CanFrame& frame)
{
    frame.is_extended = (can_frame.can_id & CAN_EFF_FLAG) != 0;
    frame.is_rtr = (can_frame.can_id & CAN_RTR_FLAG) != 0;
    if (frame.is_extended)
    {
        frame.id = can_frame.can_id & CAN_EFF_MASK;
    }
    else
    {
        frame.id = can_frame.can_id & CAN_SFF_MASK;
    }
    frame.dlc = can_frame.len;
    std::memcpy(frame.data.data(), can_frame.data, frame.dlc);
}
}

SocketCanInterface::SocketCanInterface(const std::string& interface_name)
    : socket_fd_(-1)
{
    socket_fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0)
    {
        throw std::runtime_error(std::string("Failed to open CAN socket: ") + std::strerror(errno));
    }

    struct ifreq ifr;
    std::memset(&ifr, 0, sizeof(ifr));
    std::strncpy(ifr.ifr_name, interface_name.c_str(), IFNAMSIZ - 1);
    if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0)
    {
        ::close(socket_fd_);
        throw std::runtime_error(std::string("Failed to resolve interface index: ") + std::strerror(errno));
    }

    sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        ::close(socket_fd_);
        throw std::runtime_error(std::string("Failed to bind CAN socket: ") + std::strerror(errno));
    }

    int enable_recv_own_msgs = 0;
    setsockopt(socket_fd_, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &enable_recv_own_msgs, sizeof(enable_recv_own_msgs));

    // 设为非阻塞：发送缓冲满时 write() 返回 EAGAIN/ENOBUFS 而非阻塞，
    // 配合 send() 的有限重试实现"单条总线故障不阻塞主循环"。
    const int fl = fcntl(socket_fd_, F_GETFL, 0);
    if (fl >= 0)
    {
        fcntl(socket_fd_, F_SETFL, fl | O_NONBLOCK);
    }
}

SocketCanInterface::~SocketCanInterface()
{
    if (socket_fd_ >= 0)
    {
        ::close(socket_fd_);
    }
}

void SocketCanInterface::send(const CanFrame& frame) const
{
    struct can_frame can_frame{};
    can_frame.can_id = build_can_id(frame);
    can_frame.len = frame.dlc;
    std::memcpy(can_frame.data, frame.data.data(), can_frame.len);

    // 有限重试后丢弃：当某条总线彻底故障(无 ACK)导致发送缓冲被占满时，
    // 不能像以前那样 while(true) 无限重试——否则会卡死整个主控制循环，
    // 连带其他正常总线也无法发送。这里最多等待约 2ms，仍不可写则丢弃本帧，
    // 保证单条总线故障不会阻塞其他电机的控制。
    constexpr int kMaxRetries = 2;
    for (int attempt = 0; attempt <= kMaxRetries; ++attempt)
    {
        const ssize_t bytes = ::write(socket_fd_, &can_frame, sizeof(can_frame));
        if (bytes >= 0)
        {
            return;
        }

        if (errno == ENOBUFS || errno == EAGAIN || errno == EWOULDBLOCK)
        {
            if (attempt == kMaxRetries)
            {
                // 缓冲持续满（总线可能已断开），丢弃本帧，不阻塞主循环。
                return;
            }
            struct pollfd pfd{};
            pfd.fd = socket_fd_;
            pfd.events = POLLOUT;
            ::poll(&pfd, 1, 1);  // 最多等 1ms
            continue;
        }

        throw std::runtime_error(std::string("CAN write failed: ") + std::strerror(errno));
    }
}

bool SocketCanInterface::receive(CanFrame& frame, std::chrono::milliseconds timeout)
{
    struct pollfd pfd{};
    pfd.fd = socket_fd_;
    pfd.events = POLLIN;

    const int poll_result = ::poll(&pfd, 1, static_cast<int>(timeout.count()));
    if (poll_result < 0)
    {
        if (errno == EINTR)
        {
            return false;
        }
        throw std::runtime_error(std::string("CAN poll failed: ") + std::strerror(errno));
    }
    if (poll_result == 0)
    {
        return false;
    }

    struct can_frame can_frame{};
    const ssize_t bytes = ::read(socket_fd_, &can_frame, sizeof(can_frame));
    if (bytes < 0)
    {
        if (errno == EWOULDBLOCK || errno == EAGAIN)
        {
            return false;
        }
        throw std::runtime_error(std::string("CAN read failed: ") + std::strerror(errno));
    }
    if (bytes < static_cast<ssize_t>(sizeof(struct can_frame)))
    {
        return false;
    }

    populate_frame_from_can(can_frame, frame);
    return true;
}
