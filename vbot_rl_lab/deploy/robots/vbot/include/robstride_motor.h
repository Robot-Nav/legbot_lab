#pragma once

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <map>
#include <mutex>
#include <net/if.h>
#include <optional>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define Set_mode 'j'
#define Set_parameter 'p'

#define move_control_mode 0
#define Communication_Type_MotionControl 0x01
#define Communication_Type_MotorRequest 0x02
#define Communication_Type_MotorEnable 0x03
#define Communication_Type_MotorStop 0x04
#define Communication_Type_SetPosZero 0x06
#define Communication_Type_GetSingleParameter 0x11
#define Communication_Type_SetSingleParameter 0x12

enum class ActuatorType {
    ROBSTRIDE_00 = 0,
    ROBSTRIDE_01 = 1,
    ROBSTRIDE_02 = 2,
    ROBSTRIDE_03 = 3,
    ROBSTRIDE_04 = 4,
    ROBSTRIDE_05 = 5,
    ROBSTRIDE_06 = 6
};

struct ActuatorOperation {
    double position;
    double velocity;
    double torque;
    double kp;
    double kd;
};

static const std::map<ActuatorType, ActuatorOperation>
    ACTUATOR_OPERATION_MAPPING = {
        {ActuatorType::ROBSTRIDE_00, {4 * M_PI, 50, 17, 500.0, 5.0}},
        {ActuatorType::ROBSTRIDE_01, {4 * M_PI, 44, 17, 500.0, 5.0}},
        {ActuatorType::ROBSTRIDE_02, {4 * M_PI, 44, 17, 500.0, 5.0}},
        {ActuatorType::ROBSTRIDE_03, {4 * M_PI, 50, 60, 5000.0, 100.0}},
        {ActuatorType::ROBSTRIDE_04, {4 * M_PI, 15, 120, 5000.0, 100.0}},
        {ActuatorType::ROBSTRIDE_05, {4 * M_PI, 33, 17, 500.0, 5.0}},
        {ActuatorType::ROBSTRIDE_06, {4 * M_PI, 20, 36, 5000.0, 100.0}},
};

class RobStrideMotor {
public:
    RobStrideMotor(const std::string &can_interface, uint8_t master_id,
                   uint8_t motor_id, int actuator_type)
        : iface(can_interface), master_id(master_id), motor_id(motor_id),
          actuator_type(actuator_type) {
        init_socket();
    }

    ~RobStrideMotor() {
        if (socket_fd >= 0)
            close(socket_fd);
    }

    void enable_motor() {
        struct can_frame frame{};
        frame.can_id =
            (Communication_Type_MotorEnable << 24) | (master_id << 8) | motor_id;
        frame.can_id |= CAN_EFF_FLAG;
        frame.can_dlc = 8;
        memset(frame.data, 0, 8);
        write(socket_fd, &frame, sizeof(frame));
        receive_status_frame();
    }

    void disable_motor(uint8_t clear_error = 0) {
        struct can_frame frame{};
        frame.can_id =
            (Communication_Type_MotorStop << 24) | (master_id << 8) | motor_id;
        frame.can_id |= CAN_EFF_FLAG;
        frame.can_dlc = 8;
        memset(frame.data, 0, 8);
        frame.data[0] = clear_error;
        write(socket_fd, &frame, sizeof(frame));
        receive_status_frame();
    }

    void set_mode(uint8_t mode) {
        set_parameter(0x7005, (float)mode, Set_mode);
    }

    void send_motion_command(float torque, float position_rad,
                             float velocity_rad_s, float kp, float kd) {
        struct can_frame frame{};
        auto &op = ACTUATOR_OPERATION_MAPPING.at(static_cast<ActuatorType>(actuator_type));
        frame.can_id =
            (Communication_Type_MotionControl << 24) |
            (float_to_uint(torque, -op.torque, op.torque, 16) << 8) |
            motor_id;
        frame.can_id |= CAN_EFF_FLAG;
        frame.can_dlc = 8;

        uint16_t pos = float_to_uint(position_rad, -op.position, op.position, 16);
        uint16_t vel = float_to_uint(velocity_rad_s, -op.velocity, op.velocity, 16);
        uint16_t kp_u = float_to_uint(kp, 0.0f, (float)op.kp, 16);
        uint16_t kd_u = float_to_uint(kd, 0.0f, (float)op.kd, 16);

        frame.data[0] = (pos >> 8);
        frame.data[1] = pos;
        frame.data[2] = (vel >> 8);
        frame.data[3] = vel;
        frame.data[4] = (kp_u >> 8);
        frame.data[5] = kp_u;
        frame.data[6] = (kd_u >> 8);
        frame.data[7] = kd_u;

        write(socket_fd, &frame, sizeof(frame));
        receive_status_frame();
    }

    float get_position() const { return position_; }
    float get_velocity() const { return velocity_; }
    float get_torque() const { return torque_; }

private:
    void init_socket() {
        socket_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (socket_fd < 0) {
            perror("socket");
            return;
        }

        struct ifreq ifr{};
        std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ);
        if (ioctl(socket_fd, SIOCGIFINDEX, &ifr) < 0) {
            perror("ioctl");
            return;
        }

        struct sockaddr_can addr{};
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;

        if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("bind");
            return;
        }

        struct can_filter rfilter[1];
        rfilter[0].can_id = (motor_id << 8) | CAN_EFF_FLAG;
        rfilter[0].can_mask = (0xFF << 8) | CAN_EFF_FLAG;

        if (setsockopt(socket_fd, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter,
                       sizeof(rfilter)) < 0) {
            perror("setsockopt filter");
            return;
        }
    }

    uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
        if (x < x_min) x = x_min;
        if (x > x_max) x = x_max;
        float span = x_max - x_min;
        float offset = x - x_min;
        return static_cast<uint16_t>((offset * ((1 << bits) - 1)) / span);
    }

    float uint_to_float(uint16_t x_int, float x_min, float x_max, int bits) {
        float span = x_max - x_min;
        return ((float)x_int) * span / ((1 << bits) - 1) + x_min;
    }

    void set_parameter(uint16_t index, float value, char value_mode) {
        struct can_frame frame{};
        frame.can_id =
            Communication_Type_SetSingleParameter << 24 | master_id << 8 | motor_id;
        frame.can_id |= CAN_EFF_FLAG;
        frame.can_dlc = 0x08;

        frame.data[0] = index;
        frame.data[1] = index >> 8;
        frame.data[2] = 0x00;
        frame.data[3] = 0x00;

        if (value_mode == 'p') {
            memcpy(&frame.data[4], &value, 4);
        } else if (value_mode == 'j') {
            frame.data[4] = (uint8_t)value;
            frame.data[5] = 0x00;
            frame.data[6] = 0x00;
            frame.data[7] = 0x00;
        }

        write(socket_fd, &frame, sizeof(frame));
        receive_status_frame();
    }

    void receive_status_frame() {
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 5000;
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        struct can_frame frame;
        memset(&frame, 0, sizeof(frame));
        ssize_t nbytes = recv(socket_fd, &frame, sizeof(struct can_frame), 0);

        if (nbytes <= 0) return;
        if (!(frame.can_id & CAN_EFF_FLAG)) return;

        uint32_t can_id = frame.can_id & CAN_EFF_MASK;
        uint8_t communication_type = (can_id >> 24) & 0x1F;

        if (communication_type == Communication_Type_MotorRequest) {
            if (frame.can_dlc < 8) return;

            uint16_t position_u16 = (frame.data[0] << 8) | frame.data[1];
            uint16_t velocity_u16 = (frame.data[2] << 8) | frame.data[3];
            uint16_t torque_i16 = (frame.data[4] << 8) | frame.data[5];

            auto &op = ACTUATOR_OPERATION_MAPPING.at(static_cast<ActuatorType>(actuator_type));
            position_ = ((static_cast<float>(position_u16) / 32767.0f) - 1.0f) * op.position;
            velocity_ = ((static_cast<float>(velocity_u16) / 32767.0f) - 1.0f) * op.velocity;
            torque_ = ((static_cast<float>(torque_i16) / 32767.0f) - 1.0f) * op.torque;
        }
    }

    std::string iface;
    uint8_t master_id;
    uint8_t motor_id;
    int socket_fd = -1;
    int actuator_type;

    float position_ = 0.0;
    float velocity_ = 0.0;
    float torque_ = 0.0;
};
