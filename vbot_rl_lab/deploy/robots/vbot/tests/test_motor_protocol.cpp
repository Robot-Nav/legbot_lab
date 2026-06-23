/**
 * 电机 CAN 协议编码/解码单元测试。
 *
 * 测试覆盖：
 *   - float_to_uint / uint_to_float 转换（含边界情况）
 *   - CAN 帧编码（帧头、仲裁 ID、DLC、帧尾）
 *   - 运动指令数据打包（位置、速度、kp、kd 字节布局）
 *   - 电机使能/关闭/归零指令编码
 *   - 状态帧解析与解码
 *   - 执行器类型限制与往返精度
 *   - 仲裁 ID 中电机 ID 的编码
 *
 * 编译：
 *   g++ -std=c++17 -Wall -o test_motor_protocol test_motor_protocol.cpp -lgtest -lgtest_main -lpthread
 */

#include <gtest/gtest.h>
#include "test_utils.h"
#include <cmath>
#include <limits>

// ═══════════════════════════════════════════════════════════════════════════
// float_to_uint / uint_to_float 转换测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(FloatUintConversion, MinValue) {
    // x = x_min → 0
    EXPECT_EQ(float_to_uint(-10.0f, -10.0f, 10.0f), 0);
    EXPECT_EQ(float_to_uint(-17.0f, -17.0f, 17.0f), 0);
}

TEST(FloatUintConversion, MaxValue) {
    // x = x_max → 0xFFFF (65535)
    EXPECT_EQ(float_to_uint(10.0f, -10.0f, 10.0f), 65535);
    EXPECT_EQ(float_to_uint(17.0f, -17.0f, 17.0f), 65535);
}

TEST(FloatUintConversion, MidValue) {
    // x = 中点 → ~32767
    uint16_t result = float_to_uint(0.0f, -10.0f, 10.0f);
    EXPECT_NEAR(result, 32767, 1);
}

TEST(FloatUintConversion, ClampingBelow) {
    // x < x_min → 钳位到 x_min → 0
    EXPECT_EQ(float_to_uint(-999.0f, -10.0f, 10.0f), 0);
}

TEST(FloatUintConversion, ClampingAbove) {
    // x > x_max → 钳位到 x_max → 0xFFFF
    EXPECT_EQ(float_to_uint(999.0f, -10.0f, 10.0f), 65535);
}

TEST(FloatUintConversion, RoundTripPosition) {
    // 往返测试：float → uint → float 应能恢复原始值
    const float pos_range = 4.0f * M_PIf32;
    float test_vals[] = {-pos_range, -1.0f, 0.0f, 1.0f, pos_range, 2.5f, -3.14f};

    for (float orig : test_vals) {
        uint16_t encoded = float_to_uint(orig, -pos_range, pos_range);
        float decoded = uint_to_float(encoded, -pos_range, pos_range);
        // 16 位精度：误差 < 量程 / 65536
        float tolerance = (2.0f * pos_range) / 65535.0f * 2.0f;
        EXPECT_NEAR(orig, decoded, tolerance)
            << "值 " << orig << " 的往返转换失败";
    }
}

TEST(FloatUintConversion, RoundTripVelocity) {
    const float vel_range = 50.0f;
    float test_vals[] = {-vel_range, -10.0f, 0.0f, 10.0f, vel_range};

    for (float orig : test_vals) {
        uint16_t encoded = float_to_uint(orig, -vel_range, vel_range);
        float decoded = uint_to_float(encoded, -vel_range, vel_range);
        float tolerance = (2.0f * vel_range) / 65535.0f * 2.0f;
        EXPECT_NEAR(orig, decoded, tolerance)
            << "速度值 " << orig << " 的往返转换失败";
    }
}

TEST(FloatUintConversion, ZeroInput) {
    // 对称范围下的零值 → 中点
    uint16_t result = float_to_uint(0.0f, -17.0f, 17.0f);
    EXPECT_NEAR(result, 32767, 2);
    float back = uint_to_float(result, -17.0f, 17.0f);
    EXPECT_NEAR(back, 0.0f, 0.001f);
}

TEST(FloatUintConversion, KpKdNonNegative) {
    // kp/kd 范围为 [0, max]，不允许负值
    EXPECT_EQ(float_to_uint(0.0f, 0.0f, 500.0f), 0);
    EXPECT_EQ(float_to_uint(500.0f, 0.0f, 500.0f), 65535);
    EXPECT_EQ(float_to_uint(-1.0f, 0.0f, 500.0f), 0);  // 被钳位
}

// ═══════════════════════════════════════════════════════════════════════════
// CAN 帧编码测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(CanFrameEncoding, BasicFrameStructure) {
    uint8_t data[8] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11};
    auto frame = build_can_frame(0x01, 0x12345678, data, 8);

    // 总大小：10（帧头+元数据）+ DLC = 18
    EXPECT_EQ(frame.size(), 18u);

    // 帧头
    EXPECT_EQ(frame[0], 0x45);
    EXPECT_EQ(frame[1], 0x54);

    // 通道
    EXPECT_EQ(frame[2], 0x01);

    // 仲裁 ID（4 字节，大端）
    EXPECT_EQ(frame[3], 0x12);
    EXPECT_EQ(frame[4], 0x34);
    EXPECT_EQ(frame[5], 0x56);
    EXPECT_EQ(frame[6], 0x78);

    // DLC
    EXPECT_EQ(frame[7], 8);

    // 数据负载
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(frame[8 + i], data[i]) << "第 " << i << " 字节数据不匹配";
    }

    // 帧尾
    EXPECT_EQ(frame[16], 0x0D);
    EXPECT_EQ(frame[17], 0x0A);
}

TEST(CanFrameEncoding, EmptyDataFrame) {
    auto frame = build_can_frame(0x00, 0x00000000, nullptr, 0);

    EXPECT_EQ(frame.size(), 10u);
    EXPECT_EQ(frame[0], 0x45);
    EXPECT_EQ(frame[1], 0x54);
    EXPECT_EQ(frame[7], 0);        // DLC = 0
    EXPECT_EQ(frame[8], 0x0D);     // 帧尾紧接帧头+元数据
    EXPECT_EQ(frame[9], 0x0A);
}

TEST(CanFrameEncoding, ChannelEncoding) {
    // 通道应位于字节位置 2
    for (uint8_t ch : {0x00, 0x01, 0x02, 0xFF}) {
        auto frame = build_can_frame(ch, 0, nullptr, 0);
        EXPECT_EQ(frame[2], ch) << "通道 " << (int)ch << " 编码不匹配";
    }
}

TEST(CanFrameEncoding, DlCEncoding) {
    for (uint8_t dlc : {0, 1, 2, 4, 6, 8}) {
        std::vector<uint8_t> data(dlc, 0x42);
        auto frame = build_can_frame(0, 0, data.data(), dlc);
        EXPECT_EQ(frame[7], dlc) << "DLC=" << (int)dlc << " 编码不匹配";
        EXPECT_EQ(frame.size(), 10u + dlc) << "DLC=" << (int)dlc << " 帧大小不匹配";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 运动指令编码测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(MotionCommand, ArbitrationIdStructure) {
    // arb_id = (COMM_MOTION_CONTROL << 24) | (torque_uint << 8) | motor_id
    auto frame = build_motion_command_frame(0x00, 0x21, 2,
                                            0.0f, 0.0f, 0.0f, 20.0f, 2.0f);

    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;

    uint8_t comm_type = (can_id >> 24) & 0x1F;
    uint8_t motor_id = can_id & 0xFF;

    EXPECT_EQ(comm_type, COMM_MOTION_CONTROL);
    EXPECT_EQ(motor_id, 0x21);
}

TEST(MotionCommand, MotorIdEncoding) {
    // 验证各电机 ID 正确编码在仲裁 ID 中
    for (uint8_t mid : {0x11, 0x21, 0x31, 0x12, 0x22, 0x32,
                         0x13, 0x23, 0x33, 0x14, 0x24, 0x34}) {
        auto frame = build_motion_command_frame(0x00, mid, 2,
                                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                          (frame[5] << 8) | frame[6];
        can_id &= 0x1FFFFFFF;
        EXPECT_EQ(can_id & 0xFF, mid) << "电机 ID 0x" << std::hex << (int)mid << " 不匹配";
    }
}

TEST(MotionCommand, DataByteLayout) {
    // 目标位置=0 rad, kp=20, kd=2, 执行器类型 2
    auto frame = build_motion_command_frame(0x00, 0x11, 2,
                                            0.0f, 0.0f, 0.0f, 20.0f, 2.0f);

    // 提取数据字节
    uint16_t pos_u16 = (frame[8] << 8) | frame[9];
    uint16_t vel_u16 = (frame[10] << 8) | frame[11];
    uint16_t kp_u16  = (frame[12] << 8) | frame[13];
    uint16_t kd_u16  = (frame[14] << 8) | frame[15];

    // 位置 0，对称范围 → ~中点
    EXPECT_NEAR(pos_u16, 32767, 2);
    // 速度 0 → ~中点
    EXPECT_NEAR(vel_u16, 32767, 2);
    // kp=20，范围 500 → 20/500 * 65535
    uint16_t expected_kp = static_cast<uint16_t>(20.0f / 500.0f * 65535.0f);
    EXPECT_NEAR(kp_u16, expected_kp, 2);
    // kd=2，范围 5 → 2/5 * 65535
    uint16_t expected_kd = static_cast<uint16_t>(2.0f / 5.0f * 65535.0f);
    EXPECT_NEAR(kd_u16, expected_kd, 2);
}

TEST(MotionCommand, PositionEncodingExtremes) {
    // 执行器类型 2：位置范围 = ±4π ≈ ±12.566
    const float pos_range = 4.0f * M_PIf32;

    // 最小位置 → 0
    auto f_min = build_motion_command_frame(0x00, 0x11, 2,
                                             0.0f, -pos_range, 0.0f, 0.0f, 0.0f);
    uint16_t pos_min = (f_min[8] << 8) | f_min[9];
    EXPECT_EQ(pos_min, 0);

    // 最大位置 → 0xFFFF
    auto f_max = build_motion_command_frame(0x00, 0x11, 2,
                                             0.0f, pos_range, 0.0f, 0.0f, 0.0f);
    uint16_t pos_max = (f_max[8] << 8) | f_max[9];
    EXPECT_EQ(pos_max, 65535);
}

TEST(MotionCommand, TorqueInArbId) {
    // 正力矩测试（电机 0x11）
    auto f_pos = build_motion_command_frame(0x00, 0x11, 2,
                                             8.5f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint32_t can_id = (f_pos[3] << 24) | (f_pos[4] << 16) |
                      (f_pos[5] << 8) | f_pos[6];
    can_id &= 0x1FFFFFFF;
    uint16_t torque_field = (can_id >> 8) & 0xFFFF;

    // 8.5 Nm，范围 ±17 Nm
    uint16_t expected = float_to_uint(8.5f, -17.0f, 17.0f);
    EXPECT_NEAR(torque_field, expected, 3);

    // 零力矩
    auto f_zero = build_motion_command_frame(0x00, 0x11, 2,
                                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    can_id = (f_zero[3] << 24) | (f_zero[4] << 16) |
             (f_zero[5] << 8) | f_zero[6];
    can_id &= 0x1FFFFFFF;
    torque_field = (can_id >> 8) & 0xFFFF;
    EXPECT_NEAR(torque_field, 32767, 2);  // 中点
}

// ═══════════════════════════════════════════════════════════════════════════
// 电机指令类型测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(MotorCommands, EnableFrame) {
    auto frame = build_enable_frame(0x00, 0xFF, 0x21);
    EXPECT_EQ(frame.size(), 18u);

    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;

    uint8_t comm_type = (can_id >> 24) & 0x1F;
    uint8_t master = (can_id >> 8) & 0xFF;
    uint8_t motor = can_id & 0xFF;

    EXPECT_EQ(comm_type, COMM_MOTOR_ENABLE);
    EXPECT_EQ(master, 0xFF);
    EXPECT_EQ(motor, 0x21);

    // 数据应全部为零
    for (int i = 8; i < 16; i++) {
        EXPECT_EQ(frame[i], 0) << "第 " << i << " 字节非零";
    }
}

TEST(MotorCommands, DisableFrame) {
    auto frame = build_disable_frame(0x01, 0xFF, 0x33, 0x01);

    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;

    EXPECT_EQ((can_id >> 24) & 0x1F, (uint32_t)COMM_MOTOR_STOP);
    EXPECT_EQ(can_id & 0xFF, 0x33);

    // clear_error 标志应在 data[0] 中
    EXPECT_EQ(frame[8], 0x01);
}

TEST(MotorCommands, SetZeroPositionFrame) {
    auto frame = build_zero_pos_frame(0x00, 0xFF, 0x11);

    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;

    EXPECT_EQ((can_id >> 24) & 0x1F, (uint32_t)COMM_SET_POS_ZERO);
    EXPECT_EQ(frame[8], 1);  // data[0] = 1 表示设置零位
}

TEST(MotorCommands, SetModeFrame) {
    auto frame = build_set_mode_frame(0x00, 0xFF, 0x11, 0);  // mode 0 = 运动控制

    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;

    EXPECT_EQ((can_id >> 24) & 0x1F, (uint32_t)COMM_SET_PARAM);
    // 索引 0x7005 在 data[0:1]（小端序）
    EXPECT_EQ(frame[8], 0x05);  // 索引低字节
    EXPECT_EQ(frame[9], 0x70);  // 索引高字节
    EXPECT_EQ(frame[12], 0);    // 模式值
}

// ═══════════════════════════════════════════════════════════════════════════
// 状态帧解析测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(StatusFrameParsing, ValidFrame) {
    // 手动构建状态响应帧
    // comm_type = MOTOR_REQUEST, motor_id = 0x21, data = 中点值
    uint32_t arb_id = (COMM_MOTOR_REQUEST << 24) | (0xFF << 8) | 0x21;
    uint8_t data[8] = {
        0x80, 0x00,  // 位置 = 32768 → ~0 rad
        0x80, 0x00,  // 速度 = 32768 → ~0 rad/s
        0x80, 0x00,  // 力矩 = 32768 → ~0 Nm
        0x00, 0x00,
    };

    auto frame = build_can_frame(0x00, arb_id, data, 8);
    // 在有效帧前插入垃圾数据
    std::vector<uint8_t> raw = {0x00, 0x11, 0x22};
    raw.insert(raw.end(), frame.begin(), frame.end());

    uint8_t motor_id;
    float pos, vel, tor;
    bool ok = parse_status_frame(raw, 2, motor_id, pos, vel, tor);

    EXPECT_TRUE(ok);
    EXPECT_EQ(motor_id, 0x21);
    EXPECT_NEAR(pos, 0.0f, 0.005f);
    EXPECT_NEAR(vel, 0.0f, 0.005f);
    EXPECT_NEAR(tor, 0.0f, 0.005f);
}

TEST(StatusFrameParsing, RejectsNonMotorRequest) {
    // 通信类型错误（运动控制而非电机请求）
    uint32_t arb_id = (COMM_MOTION_CONTROL << 24) | (0xFF << 8) | 0x21;
    uint8_t data[8] = {0};
    auto frame = build_can_frame(0x00, arb_id, data, 8);

    uint8_t motor_id;
    float pos, vel, tor;
    bool ok = parse_status_frame(frame, 2, motor_id, pos, vel, tor);

    EXPECT_FALSE(ok);
}

TEST(StatusFrameParsing, RejectsBadHeader) {
    // 损坏帧头
    uint32_t arb_id = (COMM_MOTOR_REQUEST << 24) | (0xFF << 8) | 0x21;
    uint8_t data[8] = {0};
    auto frame = build_can_frame(0x00, arb_id, data, 8);
    frame[0] = 0x00;  // 损坏帧头

    uint8_t motor_id;
    float pos, vel, tor;
    bool ok = parse_status_frame(frame, 2, motor_id, pos, vel, tor);

    EXPECT_FALSE(ok);
}

TEST(StatusFrameParsing, RejectsBadTrailer) {
    uint32_t arb_id = (COMM_MOTOR_REQUEST << 24) | (0xFF << 8) | 0x21;
    uint8_t data[8] = {0};
    auto frame = build_can_frame(0x00, arb_id, data, 8);
    frame[frame.size() - 2] = 0x00;  // 损坏帧尾

    uint8_t motor_id;
    float pos, vel, tor;
    bool ok = parse_status_frame(frame, 2, motor_id, pos, vel, tor);

    EXPECT_FALSE(ok);
}

TEST(StatusFrameParsing, ShortDlc) {
    // DLC < 6 应解析失败
    uint32_t arb_id = (COMM_MOTOR_REQUEST << 24) | (0xFF << 8) | 0x21;
    uint8_t data[4] = {0, 0, 0, 0};
    auto frame = build_can_frame(0x00, arb_id, data, 4);

    uint8_t motor_id;
    float pos, vel, tor;
    bool ok = parse_status_frame(frame, 2, motor_id, pos, vel, tor);

    EXPECT_FALSE(ok);
}

TEST(StatusFrameParsing, InputTooShort) {
    std::vector<uint8_t> short_data = {0x45, 0x54};  // 仅帧头，不完整

    uint8_t motor_id;
    float pos, vel, tor;
    bool ok = parse_status_frame(short_data, 2, motor_id, pos, vel, tor);

    EXPECT_FALSE(ok);
}

// ═══════════════════════════════════════════════════════════════════════════
// 执行器类型限制测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(ActuatorLimits, Type2PositionRange) {
    // ROBSTRIDE_02：位置 = ±4π，速度 = ±44，力矩 = ±17
    auto frame = build_motion_command_frame(0x00, 0x11, 2,
                                            0.0f, 4.0f * M_PIf32, 0.0f, 0.0f, 0.0f);
    uint16_t pos_u16 = (frame[8] << 8) | frame[9];
    EXPECT_EQ(pos_u16, 65535);  // 最大位置 → 最大 uint
}

TEST(ActuatorLimits, Type3TorqueRange) {
    // ROBSTRIDE_03：力矩 = ±60 Nm
    auto frame = build_motion_command_frame(0x00, 0x11, 3,
                                            60.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;
    uint16_t torque_field = (can_id >> 8) & 0xFFFF;
    EXPECT_EQ(torque_field, 65535);
}

TEST(ActuatorLimits, KpKdLimits) {
    // 类型 3：kp_max=5000, kd_max=100
    auto frame = build_motion_command_frame(0x00, 0x11, 3,
                                            0.0f, 0.0f, 0.0f, 5000.0f, 100.0f);
    uint16_t kp_u16 = (frame[12] << 8) | frame[13];
    uint16_t kd_u16 = (frame[14] << 8) | frame[15];
    EXPECT_EQ(kp_u16, 65535);
    EXPECT_EQ(kd_u16, 65535);
}

// ═══════════════════════════════════════════════════════════════════════════
// 边界情况
// ═══════════════════════════════════════════════════════════════════════════

TEST(EdgeCases, NaNHandling) {
    // NaN 应被钳位（实现通过比较进行钳位）
    float nan = std::numeric_limits<float>::quiet_NaN();
    uint16_t result = float_to_uint(nan, -10.0f, 10.0f);
    // NaN 比较始终为 false → 落入转换逻辑
    // 不验证具体输出（NaN 行为未定义），仅确认不崩溃
    SUCCEED();
}

TEST(EdgeCases, InfHandling) {
    float inf = std::numeric_limits<float>::infinity();
    // +inf 应被钳位到最大值
    EXPECT_EQ(float_to_uint(inf, -10.0f, 10.0f), 65535);
    // -inf 应被钳位到最小值
    EXPECT_EQ(float_to_uint(-inf, -10.0f, 10.0f), 0);
}

TEST(EdgeCases, ZeroSpanRange) {
    // 退化范围（min == max）
    uint16_t result = float_to_uint(5.0f, 5.0f, 5.0f);
    // span = 0 → 除零错误。行为依赖具体实现，仅验证不崩溃。
    SUCCEED();
}

TEST(EdgeCases, MaxMotorId) {
    // 电机 ID 0xFF
    auto frame = build_motion_command_frame(0x00, 0xFF, 2,
                                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint32_t can_id = (frame[3] << 24) | (frame[4] << 16) |
                      (frame[5] << 8) | frame[6];
    can_id &= 0x1FFFFFFF;
    EXPECT_EQ(can_id & 0xFF, 0xFF);
}

TEST(EdgeCases, MaxChannel) {
    auto frame = build_can_frame(0xFF, 0, nullptr, 0);
    EXPECT_EQ(frame[2], 0xFF);
}

// ═══════════════════════════════════════════════════════════════════════════
// 协议往返测试：编码指令 → 解码模拟状态
// ═══════════════════════════════════════════════════════════════════════════

TEST(ProtocolRoundTrip, EncodeDecodeConsistency) {
    // 模拟：编码运动指令，然后创建对应的模拟状态响应，验证来回正确
    const float target_pos = 0.8f;
    const float target_vel = 1.5f;
    const float target_torque = 3.0f;
    const float target_kp = 40.0f;
    const float target_kd = 4.0f;
    const int actuator_type = 2;

    // 构建运动指令
    auto cmd_frame = build_motion_command_frame(
        0x00, 0x21, actuator_type, target_torque, target_pos, target_vel,
        target_kp, target_kd);

    // 指令帧本身不可直接作为状态帧解码（通信类型不同），
    // 需要单独构建包含相同编码值的状态帧
    uint16_t pos_u16 = (cmd_frame[8] << 8) | cmd_frame[9];
    uint16_t vel_u16 = (cmd_frame[10] << 8) | cmd_frame[11];

    // 创建包含相同编码值的状态响应帧
    uint32_t status_arb_id = (COMM_MOTOR_REQUEST << 24) | (0xFF << 8) | 0x21;
    uint8_t status_data[8] = {
        cmd_frame[8], cmd_frame[9],    // 相同位置值
        cmd_frame[10], cmd_frame[11],  // 相同速度值
        0x80, 0x01,                     // 某力矩值
        0x00, 0x00,
    };
    auto status_frame = build_can_frame(0x00, status_arb_id, status_data, 8);

    uint8_t decoded_id;
    float decoded_pos, decoded_vel, decoded_torque;
    bool ok = parse_status_frame(status_frame, actuator_type,
                                  decoded_id, decoded_pos, decoded_vel, decoded_torque);

    ASSERT_TRUE(ok);
    EXPECT_EQ(decoded_id, 0x21);
    // 位置应正确往返
    float pos_tolerance = (8.0f * M_PIf32) / 65535.0f * 2.0f;
    EXPECT_NEAR(decoded_pos, target_pos, pos_tolerance);
    // 速度应正确往返
    float vel_tolerance = 88.0f / 65535.0f * 2.0f;
    EXPECT_NEAR(decoded_vel, target_vel, vel_tolerance);
}
