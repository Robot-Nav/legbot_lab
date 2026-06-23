#pragma once

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <termios.h>
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
    RobStrideMotor(int serial_fd, uint8_t channel, uint8_t master_id,
                   uint8_t motor_id, int actuator_type)
        : serial_fd(serial_fd), channel(channel), master_id(master_id),
          motor_id(motor_id), actuator_type(actuator_type) {}

    ~RobStrideMotor() = default;

    void enable_motor() {
        uint32_t arb_id = (Communication_Type_MotorEnable << 24) | (master_id << 8) | motor_id;
        uint8_t data[8] = {0};
        send_can_frame(arb_id, data, 8);
        receive_status_frame();
    }

    void disable_motor(uint8_t clear_error = 0) {
        uint32_t arb_id = (Communication_Type_MotorStop << 24) | (master_id << 8) | motor_id;
        uint8_t data[8] = {0};
        data[0] = clear_error;
        send_can_frame(arb_id, data, 8);
        receive_status_frame();
    }

    void set_mode(uint8_t mode) {
        set_parameter(0x7005, (float)mode, Set_mode);
    }

    void send_motion_command(float torque, float position_rad,
                             float velocity_rad_s, float kp, float kd) {
        auto &op = ACTUATOR_OPERATION_MAPPING.at(static_cast<ActuatorType>(actuator_type));

        uint32_t arb_id =
            (Communication_Type_MotionControl << 24) |
            (float_to_uint(torque, -op.torque, op.torque, 16) << 8) |
            motor_id;

        uint16_t pos = float_to_uint(position_rad, -op.position, op.position, 16);
        uint16_t vel = float_to_uint(velocity_rad_s, -op.velocity, op.velocity, 16);
        uint16_t kp_u = float_to_uint(kp, 0.0f, (float)op.kp, 16);
        uint16_t kd_u = float_to_uint(kd, 0.0f, (float)op.kd, 16);

        uint8_t data[8];
        data[0] = (pos >> 8);
        data[1] = pos;
        data[2] = (vel >> 8);
        data[3] = vel;
        data[4] = (kp_u >> 8);
        data[5] = kp_u;
        data[6] = (kd_u >> 8);
        data[7] = kd_u;

        send_can_frame(arb_id, data, 8);
        receive_status_frame();
    }

    void set_zero_position() {
        uint32_t arb_id = (Communication_Type_SetPosZero << 24) | (master_id << 8) | motor_id;
        uint8_t data[8] = {1, 0, 0, 0, 0, 0, 0, 0};
        send_can_frame(arb_id, data, 8);
        receive_status_frame();
    }

    float get_position() const { return position_; }
    float get_velocity() const { return velocity_; }
    float get_torque() const { return torque_; }

private:
    void send_can_frame(uint32_t arbitration_id, const uint8_t *data, uint8_t dlc) {
        tcflush(serial_fd, TCIFLUSH);

        uint8_t frame[18];
        frame[0] = 0x45;
        frame[1] = 0x54;
        frame[2] = channel;
        frame[3] = (arbitration_id >> 24) & 0xFF;
        frame[4] = (arbitration_id >> 16) & 0xFF;
        frame[5] = (arbitration_id >> 8) & 0xFF;
        frame[6] = arbitration_id & 0xFF;
        frame[7] = dlc;
        memcpy(&frame[8], data, dlc);
        frame[8 + dlc] = 0x0D;
        frame[9 + dlc] = 0x0A;

        write(serial_fd, frame, 10 + dlc);
        tcdrain(serial_fd);
    }

    bool receive_status_frame() {
        std::vector<uint8_t> rx_buffer;
        auto start = std::chrono::steady_clock::now();
        const int timeout_ms = 10;

        while (true) {
            auto now = std::chrono::steady_clock::now();
            int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed_ms > timeout_ms) return false;

            uint8_t buf[256];
            int n = read(serial_fd, buf, sizeof(buf));
            if (n > 0) {
                rx_buffer.insert(rx_buffer.end(), buf, buf + n);
            }

            while (rx_buffer.size() >= 10) {
                if (rx_buffer[0] != 0x45 || rx_buffer[1] != 0x54) {
                    rx_buffer.erase(rx_buffer.begin());
                    continue;
                }

                uint8_t dlc = rx_buffer[7];
                size_t frame_size = 10 + dlc;

                if (rx_buffer.size() < frame_size) break;

                if (rx_buffer[frame_size - 2] != 0x0D || rx_buffer[frame_size - 1] != 0x0A) {
                    rx_buffer.erase(rx_buffer.begin());
                    continue;
                }

                uint32_t can_id = ((uint32_t)rx_buffer[3] << 24) |
                                  ((uint32_t)rx_buffer[4] << 16) |
                                  ((uint32_t)rx_buffer[5] << 8) |
                                  (uint32_t)rx_buffer[6];
                can_id &= 0x1FFFFFFF;

                uint8_t comm_type = (can_id >> 24) & 0x1F;
                uint8_t resp_motor_id = (can_id >> 8) & 0xFF;

                std::vector<uint8_t> data(rx_buffer.begin() + 8,
                                          rx_buffer.begin() + 8 + dlc);
                rx_buffer.erase(rx_buffer.begin(),
                                rx_buffer.begin() + frame_size);

                if (comm_type == Communication_Type_MotorRequest &&
                    resp_motor_id == motor_id && dlc >= 6) {
                    uint16_t position_u16 = (data[0] << 8) | data[1];
                    uint16_t velocity_u16 = (data[2] << 8) | data[3];
                    uint16_t torque_i16 = (data[4] << 8) | data[5];

                    auto &op = ACTUATOR_OPERATION_MAPPING.at(
                        static_cast<ActuatorType>(actuator_type));
                    position_ = ((static_cast<float>(position_u16) / 32767.0f) - 1.0f) * op.position;
                    velocity_ = ((static_cast<float>(velocity_u16) / 32767.0f) - 1.0f) * op.velocity;
                    torque_ = ((static_cast<float>(torque_i16) / 32767.0f) - 1.0f) * op.torque;
                    return true;
                }
            }

            usleep(200);
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
        uint32_t arb_id = (Communication_Type_SetSingleParameter << 24) | (master_id << 8) | motor_id;
        uint8_t data[8] = {0};
        data[0] = index;
        data[1] = index >> 8;
        data[2] = 0x00;
        data[3] = 0x00;

        if (value_mode == 'p') {
            memcpy(&data[4], &value, 4);
        } else if (value_mode == 'j') {
            data[4] = (uint8_t)value;
        }

        send_can_frame(arb_id, data, 8);
        receive_status_frame();
    }

    int serial_fd;
    uint8_t channel;
    uint8_t master_id;
    uint8_t motor_id;
    int actuator_type;

    float position_ = 0.0;
    float velocity_ = 0.0;
    float torque_ = 0.0;
};
