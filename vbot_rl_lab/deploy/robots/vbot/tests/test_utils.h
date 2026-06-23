#pragma once
/// 测试工具头文件 — 从 robstride_motor.h 和 Types.h 中提取纯函数，
/// 使其可在不依赖硬件的情况下进行单元测试。

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>

// ─── 浮点数 <-> 无符号整数转换（来自 robstride_motor.h）─────────────────

inline uint16_t float_to_uint(float x, float x_min, float x_max, int bits = 16) {
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    float span = x_max - x_min;
    float offset = x - x_min;
    return static_cast<uint16_t>((offset * ((1 << bits) - 1)) / span);
}

inline float uint_to_float(uint16_t x_int, float x_min, float x_max, int bits = 16) {
    float span = x_max - x_min;
    return static_cast<float>(x_int) * span / static_cast<float>((1 << bits) - 1) + x_min;
}

// ─── CRC16-Modbus 校验（来自 Types.h Sim2RealInterface）──────────────────

inline uint16_t crc16_modbus(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

inline uint16_t crc16_modbus(const std::vector<uint8_t> &data) {
    return crc16_modbus(data.data(), data.size());
}

// ─── 欧拉角转四元数（来自 Types.h Sim2RealInterface）────────────────────

inline void euler_to_quaternion(float yaw, float pitch, float roll, float q[4]) {
    float cy = std::cos(yaw * 0.5f);
    float sy = std::sin(yaw * 0.5f);
    float cp = std::cos(pitch * 0.5f);
    float sp = std::sin(pitch * 0.5f);
    float cr = std::cos(roll * 0.5f);
    float sr = std::sin(roll * 0.5f);

    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}

// ─── CAN 帧编码辅助函数（来自 robstride_motor.h）────────────────────────

// 通信类型常量
constexpr uint8_t COMM_MOTION_CONTROL   = 0x01;
constexpr uint8_t COMM_MOTOR_REQUEST    = 0x02;
constexpr uint8_t COMM_MOTOR_ENABLE     = 0x03;
constexpr uint8_t COMM_MOTOR_STOP       = 0x04;
constexpr uint8_t COMM_SET_POS_ZERO     = 0x06;
constexpr uint8_t COMM_SET_PARAM        = 0x12;

// 帧分隔符
constexpr uint8_t FRAME_HEADER_0 = 0x45;
constexpr uint8_t FRAME_HEADER_1 = 0x54;
constexpr uint8_t FRAME_TRAILER_0 = 0x0D;
constexpr uint8_t FRAME_TRAILER_1 = 0x0A;

/// 构建 CAN-over-serial 帧，与 send_can_frame() 格式一致。
inline std::vector<uint8_t> build_can_frame(
    uint8_t channel, uint32_t arb_id, const uint8_t *data, uint8_t dlc)
{
    std::vector<uint8_t> frame(10 + dlc);
    frame[0] = FRAME_HEADER_0;
    frame[1] = FRAME_HEADER_1;
    frame[2] = channel;
    frame[3] = (arb_id >> 24) & 0xFF;
    frame[4] = (arb_id >> 16) & 0xFF;
    frame[5] = (arb_id >> 8) & 0xFF;
    frame[6] = arb_id & 0xFF;
    frame[7] = dlc;
    if (dlc > 0) std::memcpy(&frame[8], data, dlc);
    frame[8 + dlc] = FRAME_TRAILER_0;
    frame[9 + dlc] = FRAME_TRAILER_1;
    return frame;
}

/// 构建运动控制指令帧（与 send_motion_command 格式一致）。
inline std::vector<uint8_t> build_motion_command_frame(
    uint8_t channel, uint8_t motor_id, int actuator_type,
    float torque, float position, float velocity, float kp, float kd)
{
    // 执行器限制（来自 ACTUATOR_OPERATION_MAPPING）
    struct ActuatorLimits { double pos, vel, tor, kp_max, kd_max; };
    static const ActuatorLimits limits[] = {
        {4*M_PI, 50, 17,  500,   5},
        {4*M_PI, 44, 17,  500,   5},
        {4*M_PI, 44, 17,  500,   5},
        {4*M_PI, 50, 60,  5000, 100},
        {4*M_PI, 15, 120, 5000, 100},
        {4*M_PI, 33, 17,  500,   5},
        {4*M_PI, 20, 36,  5000, 100},
    };
    auto &op = limits[actuator_type];

    uint16_t torque_u16 = float_to_uint(
        torque, static_cast<float>(-op.tor), static_cast<float>(op.tor), 16);
    uint32_t arb_id = (COMM_MOTION_CONTROL << 24) | (torque_u16 << 8) | motor_id;

    uint16_t pos_u = float_to_uint(position, -op.pos, op.pos, 16);
    uint16_t vel_u = float_to_uint(velocity, -op.vel, op.vel, 16);
    uint16_t kp_u  = float_to_uint(kp, 0.0f, static_cast<float>(op.kp_max), 16);
    uint16_t kd_u  = float_to_uint(kd, 0.0f, static_cast<float>(op.kd_max), 16);

    uint8_t data[8];
    data[0] = (pos_u >> 8) & 0xFF;
    data[1] = pos_u & 0xFF;
    data[2] = (vel_u >> 8) & 0xFF;
    data[3] = vel_u & 0xFF;
    data[4] = (kp_u >> 8) & 0xFF;
    data[5] = kp_u & 0xFF;
    data[6] = (kd_u >> 8) & 0xFF;
    data[7] = kd_u & 0xFF;

    return build_can_frame(channel, arb_id, data, 8);
}

/// 构建电机使能帧。
inline std::vector<uint8_t> build_enable_frame(
    uint8_t channel, uint8_t master_id, uint8_t motor_id)
{
    uint32_t arb_id = (COMM_MOTOR_ENABLE << 24) | (master_id << 8) | motor_id;
    uint8_t zero[8] = {0};
    return build_can_frame(channel, arb_id, zero, 8);
}

/// 构建电机关闭帧。
inline std::vector<uint8_t> build_disable_frame(
    uint8_t channel, uint8_t master_id, uint8_t motor_id, uint8_t clear_error = 0)
{
    uint32_t arb_id = (COMM_MOTOR_STOP << 24) | (master_id << 8) | motor_id;
    uint8_t data[8] = {0};
    data[0] = clear_error;
    return build_can_frame(channel, arb_id, data, 8);
}

/// 构建设置零位帧。
inline std::vector<uint8_t> build_zero_pos_frame(
    uint8_t channel, uint8_t master_id, uint8_t motor_id)
{
    uint32_t arb_id = (COMM_SET_POS_ZERO << 24) | (master_id << 8) | motor_id;
    uint8_t data[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    return build_can_frame(channel, arb_id, data, 8);
}

/// 构建设置模式帧（参数 0x7005）。
inline std::vector<uint8_t> build_set_mode_frame(
    uint8_t channel, uint8_t master_id, uint8_t motor_id, uint8_t mode)
{
    uint32_t arb_id = (COMM_SET_PARAM << 24) | (master_id << 8) | motor_id;
    uint8_t data[8] = {0};
    data[0] = 0x05;  // 索引低字节
    data[1] = 0x70;  // 索引高字节
    data[2] = 0x00;
    data[3] = 0x00;
    data[4] = mode;
    return build_can_frame(channel, arb_id, data, 8);
}

/// 解析状态响应帧。有效则返回 true 并填充输出参数。
inline bool parse_status_frame(
    const std::vector<uint8_t> &raw, int actuator_type,
    uint8_t &out_motor_id, float &out_position, float &out_velocity, float &out_torque)
{
    if (raw.size() < 10) return false;

    // 查找帧头
    size_t idx = 0;
    while (idx <= raw.size() - 10) {
        if (raw[idx] == FRAME_HEADER_0 && raw[idx + 1] == FRAME_HEADER_1)
            break;
        idx++;
    }
    if (idx > raw.size() - 10) return false;

    uint8_t dlc = raw[idx + 7];
    size_t frame_size = 10 + dlc;
    if (raw.size() - idx < frame_size) return false;

    // 验证帧尾
    if (raw[idx + frame_size - 2] != FRAME_TRAILER_0 ||
        raw[idx + frame_size - 1] != FRAME_TRAILER_1)
        return false;

    uint32_t can_id = ((uint32_t)raw[idx + 3] << 24) |
                      ((uint32_t)raw[idx + 4] << 16) |
                      ((uint32_t)raw[idx + 5] << 8) |
                      (uint32_t)raw[idx + 6];
    can_id &= 0x1FFFFFFF;

    uint8_t comm_type = (can_id >> 24) & 0x1F;
    out_motor_id = (can_id >> 8) & 0xFF;

    if (comm_type != COMM_MOTOR_REQUEST || dlc < 6)
        return false;

    auto d = &raw[idx + 8];
    uint16_t pos_u16 = (d[0] << 8) | d[1];
    uint16_t vel_u16 = (d[2] << 8) | d[3];
    uint16_t tor_u16 = (d[4] << 8) | d[5];

    struct ActuatorLimits { double pos, vel, tor, kp_max, kd_max; };
    static const ActuatorLimits limits[] = {
        {4*M_PI, 50, 17,  500,   5},
        {4*M_PI, 44, 17,  500,   5},
        {4*M_PI, 44, 17,  500,   5},
        {4*M_PI, 50, 60,  5000, 100},
        {4*M_PI, 15, 120, 5000, 100},
        {4*M_PI, 33, 17,  500,   5},
        {4*M_PI, 20, 36,  5000, 100},
    };
    auto &op = limits[actuator_type];

    out_position = uint_to_float(pos_u16, -op.pos, op.pos);
    out_velocity = uint_to_float(vel_u16, -op.vel, op.vel);
    out_torque   = uint_to_float(tor_u16, -op.tor, op.tor);
    return true;
}

// ─── IMU 帧辅助函数 ──────────────────────────────────────────────────────

constexpr uint8_t IMU_HEADER[4] = {0xEB, 0x90, 0xA5, 0xFF};
constexpr uint8_t IMU_FOOTER[2] = {0x80, 0x7F};

/// 构建完整的 IMU 数据帧（32 字节），用于测试。
inline std::vector<uint8_t> build_imu_frame(
    float yaw, float pitch, float roll, float wx, float wy, float wz)
{
    std::vector<uint8_t> frame(32);
    frame[0] = IMU_HEADER[0];
    frame[1] = IMU_HEADER[1];
    frame[2] = IMU_HEADER[2];
    frame[3] = IMU_HEADER[3];

    float data[6] = {yaw, pitch, roll, wx, wy, wz};
    std::memcpy(&frame[4], data, 24);

    uint16_t crc = crc16_modbus(frame.data(), 28);
    frame[28] = crc & 0xFF;
    frame[29] = (crc >> 8) & 0xFF;

    frame[30] = IMU_FOOTER[0];
    frame[31] = IMU_FOOTER[1];
    return frame;
}
