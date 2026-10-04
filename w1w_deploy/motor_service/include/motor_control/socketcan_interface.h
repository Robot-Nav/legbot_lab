#ifndef MOTOR_CONTROL_SOCKETCAN_INTERFACE_H
#define MOTOR_CONTROL_SOCKETCAN_INTERFACE_H

#include <chrono>
#include <string>

#include "motor_control/can_frame.h"

class SocketCanInterface {
public:
    explicit SocketCanInterface(const std::string& interface_name);
    ~SocketCanInterface();

    void send(const CanFrame& frame) const;
    bool receive(CanFrame& frame, std::chrono::milliseconds timeout);

private:
    int socket_fd_;
};

#endif
