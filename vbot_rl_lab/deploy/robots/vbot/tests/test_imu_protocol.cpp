/**
 * IMU 串口协议单元测试。
 *
 * 测试覆盖：
 *   - CRC16-Modbus 校验计算（已知测试向量、边界情况）
 *   - 欧拉角到四元数转换（已知角度、往返测试）
 *   - IMU 帧构建（帧头、数据打包、CRC、帧尾）
 *   - IMU 帧解析（有效帧、损坏帧、缓冲区管理）
 *   - 帧拒绝：帧头错误、帧尾错误、CRC 不匹配
 *   - 多帧缓冲区处理
 *   - 编解码往返数据完整性
 *
 * 编译：
 *   g++ -std=c++17 -Wall -o test_imu_protocol test_imu_protocol.cpp -lgtest -lgtest_main -lpthread
 */

#include <gtest/gtest.h>
#include "test_utils.h"
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// CRC16-Modbus 测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(Crc16Modbus, KnownTestVectors) {
    // 标准 CRC-16/MODBUS 校验值："123456789" → 0x4B37
    {
        uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
        EXPECT_EQ(crc16_modbus(data, 9), 0x4B37);
    }
    {
        uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
        EXPECT_EQ(crc16_modbus(data, 4), 0x2BA1);
    }
    {
        uint8_t data[] = {0x00, 0x00, 0x00, 0x00};
        EXPECT_EQ(crc16_modbus(data, 4), 0x2400);
    }
    {
        uint8_t data[] = {0xFF, 0xFF, 0xFF, 0xFF};
        EXPECT_EQ(crc16_modbus(data, 4), 0xB001);
    }
}

TEST(Crc16Modbus, EmptyInput) {
    // 空数据 CRC = 0xFFFF（初始值）
    EXPECT_EQ(crc16_modbus(nullptr, 0), 0xFFFF);
}

TEST(Crc16Modbus, SingleByte) {
    uint8_t data[] = {0x00};
    EXPECT_EQ(crc16_modbus(data, 1), 0x40BF);
}

TEST(Crc16Modbus, AllZeros) {
    std::vector<uint8_t> data(28, 0x00);
    uint16_t crc = crc16_modbus(data);
    EXPECT_EQ(crc, 0xEAA8);
}

TEST(Crc16Modbus, Deterministic) {
    // 相同输入必须产生相同的 CRC 值
    std::vector<uint8_t> data = {0xEB, 0x90, 0xA5, 0xFF,
                                  0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00};

    uint16_t crc1 = crc16_modbus(data);
    uint16_t crc2 = crc16_modbus(data);
    EXPECT_EQ(crc1, crc2) << "CRC 必须是确定性的";
}

TEST(Crc16Modbus, SingleBitChangeDetected) {
    // 翻转一个比特位必须产生不同的 CRC
    std::vector<uint8_t> data(28, 0xA5);
    uint16_t crc1 = crc16_modbus(data);
    data[10] ^= 0x01;  // 翻转一个比特位
    uint16_t crc2 = crc16_modbus(data);
    EXPECT_NE(crc1, crc2) << "单比特翻转必须改变 CRC 值";
}

TEST(Crc16Modbus, ByteOrderChangeDetected) {
    // 字节顺序改变必须产生不同的 CRC
    std::vector<uint8_t> data1 = {0x01, 0x02, 0x03, 0x04};
    std::vector<uint8_t> data2 = {0x04, 0x03, 0x02, 0x01};
    EXPECT_NE(crc16_modbus(data1), crc16_modbus(data2))
        << "字节顺序改变必须改变 CRC 值";
}

// ═══════════════════════════════════════════════════════════════════════════
// 欧拉角转四元数转换测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(EulerToQuaternion, Identity) {
    // 零旋转 → 单位四元数
    float q[4];
    euler_to_quaternion(0.0f, 0.0f, 0.0f, q);
    EXPECT_NEAR(q[0], 1.0f, 1e-6f);  // w
    EXPECT_NEAR(q[1], 0.0f, 1e-6f);  // x
    EXPECT_NEAR(q[2], 0.0f, 1e-6f);  // y
    EXPECT_NEAR(q[3], 0.0f, 1e-6f);  // z
}

TEST(EulerToQuaternion, UnitNorm) {
    // 四元数应始终具有单位模长
    float test_angles[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.5f, -0.3f, 0.7f},
        {-2.0f, 1.2f, -0.5f},
        {3.14159f, 0.0f, 0.0f},
        {0.0f, 1.57f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {0.1f, -0.2f, 0.3f},
    };

    for (auto &angles : test_angles) {
        float q[4];
        euler_to_quaternion(angles[0], angles[1], angles[2], q);
        float norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        EXPECT_NEAR(norm, 1.0f, 1e-5f)
            << "非单位四元数，ypr=("
            << angles[0] << ", " << angles[1] << ", " << angles[2] << ")";
    }
}

TEST(EulerToQuaternion, PureYaw) {
    // 偏航 π/2 → 四元数 (cos(π/4), 0, 0, sin(π/4))
    float q[4];
    euler_to_quaternion(M_PIf32 / 2.0f, 0.0f, 0.0f, q);

    float expected_w = std::cos(M_PIf32 / 4.0f);
    float expected_z = std::sin(M_PIf32 / 4.0f);

    EXPECT_NEAR(q[0], expected_w, 1e-5f);
    EXPECT_NEAR(q[1], 0.0f, 1e-5f);
    EXPECT_NEAR(q[2], 0.0f, 1e-5f);
    EXPECT_NEAR(q[3], expected_z, 1e-5f);
}

TEST(EulerToQuaternion, PurePitch) {
    // 俯仰 π/4
    float q[4];
    euler_to_quaternion(0.0f, M_PIf32 / 4.0f, 0.0f, q);

    float expected_w = std::cos(M_PIf32 / 8.0f);
    float expected_y = std::sin(M_PIf32 / 8.0f);

    EXPECT_NEAR(q[0], expected_w, 1e-5f);
    EXPECT_NEAR(q[1], 0.0f, 1e-5f);
    EXPECT_NEAR(q[2], expected_y, 1e-5f);
    EXPECT_NEAR(q[3], 0.0f, 1e-5f);
}

TEST(EulerToQuaternion, PureRoll) {
    // 横滚 π/4
    float q[4];
    euler_to_quaternion(0.0f, 0.0f, M_PIf32 / 4.0f, q);

    float expected_w = std::cos(M_PIf32 / 8.0f);
    float expected_x = std::sin(M_PIf32 / 8.0f);

    EXPECT_NEAR(q[0], expected_w, 1e-5f);
    EXPECT_NEAR(q[1], expected_x, 1e-5f);
    EXPECT_NEAR(q[2], 0.0f, 1e-5f);
    EXPECT_NEAR(q[3], 0.0f, 1e-5f);
}

TEST(EulerToQuaternion, YawPlusPi) {
    // 偏航角和偏航角+2π 应得到相同的四元数（允许符号相反）
    float q1[4], q2[4];
    euler_to_quaternion(1.0f, 0.3f, -0.2f, q1);
    euler_to_quaternion(1.0f + 2.0f * M_PIf32, 0.3f, -0.2f, q2);

    // 应相同或互为相反数（代表相同旋转）
    bool same_or_negated =
        (std::abs(q1[0] - q2[0]) < 1e-5f &&
         std::abs(q1[1] - q2[1]) < 1e-5f &&
         std::abs(q1[2] - q2[2]) < 1e-5f &&
         std::abs(q1[3] - q2[3]) < 1e-5f) ||
        (std::abs(q1[0] + q2[0]) < 1e-5f &&
         std::abs(q1[1] + q2[1]) < 1e-5f &&
         std::abs(q1[2] + q2[2]) < 1e-5f &&
         std::abs(q1[3] + q2[3]) < 1e-5f);
    EXPECT_TRUE(same_or_negated) << "偏航角+2π 应产生相同旋转";
}

// 辅助函数：四元数转欧拉角，用于往返测试
inline void quaternion_to_euler(const float q[4], float &yaw, float &pitch, float &roll) {
    float sinr_cosp = 2.0f * (q[0] * q[1] + q[2] * q[3]);
    float cosr_cosp = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (q[0] * q[2] - q[3] * q[1]);
    if (std::abs(sinp) >= 1.0f)
        pitch = std::copysign(M_PIf32 / 2.0f, sinp);
    else
        pitch = std::asin(sinp);

    float siny_cosp = 2.0f * (q[0] * q[3] + q[1] * q[2]);
    float cosy_cosp = 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

TEST(EulerToQuaternion, RoundTrip) {
    float test_angles[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f},
        {0.0f, 0.8f, 0.0f},
        {0.0f, 0.0f, -0.5f},
        {1.0f, -0.3f, 0.7f},
        {-2.0f, 0.4f, -0.2f},
        {0.5f, 0.5f, 0.5f},
    };

    for (auto &angles : test_angles) {
        float q[4];
        euler_to_quaternion(angles[0], angles[1], angles[2], q);

        float y2, p2, r2;
        quaternion_to_euler(q, y2, p2, r2);

        auto diff = [](float a, float b) {
            float d = std::abs(a - b);
            d = std::fmod(d, 2.0f * M_PIf32);
            if (d > M_PIf32) d = 2.0f * M_PIf32 - d;
            return d;
        };

        EXPECT_LT(diff(angles[0], y2), 1e-4f)
            << "偏航角往返失败：" << angles[0] << " -> " << y2;
        EXPECT_LT(diff(angles[1], p2), 1e-4f)
            << "俯仰角往返失败：" << angles[1] << " -> " << p2;
        EXPECT_LT(diff(angles[2], r2), 1e-4f)
            << "横滚角往返失败：" << angles[2] << " -> " << r2;
    }
}

TEST(EulerToQuaternion, NearGimbalLock) {
    // 俯仰角接近 ±π/2（万向节死锁）— 不应崩溃
    float q[4];
    euler_to_quaternion(0.5f, M_PIf32 / 2.0f - 0.01f, 0.3f, q);
    float norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    EXPECT_NEAR(norm, 1.0f, 1e-5f);

    euler_to_quaternion(0.5f, -M_PIf32 / 2.0f + 0.01f, 0.3f, q);
    norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(EulerToQuaternion, LargeAngles) {
    // 超出 ±π 的角度仍应产生有效单位四元数
    float q[4];
    euler_to_quaternion(10.0f, 5.0f, -8.0f, q);
    float norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

// ═══════════════════════════════════════════════════════════════════════════
// IMU 帧构建测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(ImuFrameConstruction, FrameSize) {
    auto frame = build_imu_frame(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(frame.size(), 32u);
}

TEST(ImuFrameConstruction, HeaderPresent) {
    auto frame = build_imu_frame(1.0f, 0.5f, -0.3f, 0.1f, 0.2f, 0.3f);
    EXPECT_EQ(frame[0], 0xEB);
    EXPECT_EQ(frame[1], 0x90);
    EXPECT_EQ(frame[2], 0xA5);
    EXPECT_EQ(frame[3], 0xFF);
}

TEST(ImuFrameConstruction, FooterPresent) {
    auto frame = build_imu_frame(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(frame[30], 0x80);
    EXPECT_EQ(frame[31], 0x7F);
}

TEST(ImuFrameConstruction, DataEncoding) {
    float yaw = 1.5f, pitch = -0.3f, roll = 0.7f;
    float wx = 0.1f, wy = -0.2f, wz = 0.05f;

    auto frame = build_imu_frame(yaw, pitch, roll, wx, wy, wz);

    // 解码 float 数据回来验证
    float decoded[6];
    std::memcpy(decoded, &frame[4], 24);

    EXPECT_FLOAT_EQ(decoded[0], yaw);
    EXPECT_FLOAT_EQ(decoded[1], pitch);
    EXPECT_FLOAT_EQ(decoded[2], roll);
    EXPECT_FLOAT_EQ(decoded[3], wx);
    EXPECT_FLOAT_EQ(decoded[4], wy);
    EXPECT_FLOAT_EQ(decoded[5], wz);
}

TEST(ImuFrameConstruction, CrcValid) {
    auto frame = build_imu_frame(0.5f, 0.2f, -0.1f, 0.0f, 0.0f, 0.0f);

    uint16_t stored_crc = frame[28] | (frame[29] << 8);
    uint16_t computed_crc = crc16_modbus(frame.data(), 28);

    EXPECT_EQ(stored_crc, computed_crc)
        << "构建的帧必须具有有效 CRC";
}

// ═══════════════════════════════════════════════════════════════════════════
// IMU 帧解析（模拟 read_imu() 逻辑）
// ═══════════════════════════════════════════════════════════════════════════

// 最小化解析器，匹配 Types.h 中 read_imu() 算法
class ImuParser {
public:
    std::vector<uint8_t> buffer;
    size_t frames_parsed = 0;
    size_t crc_errors = 0;
    size_t header_rejects = 0;
    size_t footer_rejects = 0;

    struct ParsedFrame {
        float yaw, pitch, roll, wx, wy, wz;
        float quat[4];
    };
    std::deque<ParsedFrame> parsed_frames;

    void feed(const uint8_t *data, size_t len) {
        buffer.insert(buffer.end(), data, data + len);

        // 缓冲区超过 512 字节时裁剪
        if (buffer.size() > 512) {
            buffer.erase(buffer.begin(),
                         buffer.begin() + (buffer.size() - 512));
        }

        while (buffer.size() >= 32) {
            // 检查帧头
            if (buffer[0] != 0xEB || buffer[1] != 0x90 ||
                buffer[2] != 0xA5 || buffer[3] != 0xFF) {
                header_rejects++;
                buffer.erase(buffer.begin());
                continue;
            }

            // 检查帧尾
            if (buffer[30] != 0x80 || buffer[31] != 0x7F) {
                footer_rejects++;
                buffer.erase(buffer.begin());
                continue;
            }

            // 检查 CRC
            uint16_t received_crc = buffer[29] << 8 | buffer[28];
            uint16_t calc_crc = crc16_modbus(buffer.data(), 28);
            if (received_crc != calc_crc) {
                crc_errors++;
                buffer.erase(buffer.begin());
                continue;
            }

            // 解析数据
            ParsedFrame pf;
            float raw[6];
            std::memcpy(raw, &buffer[4], 24);
            pf.yaw = raw[0];
            pf.pitch = raw[1];
            pf.roll = raw[2];
            pf.wx = raw[3];
            pf.wy = raw[4];
            pf.wz = raw[5];

            euler_to_quaternion(pf.yaw, pf.pitch, pf.roll, pf.quat);

            parsed_frames.push_back(pf);
            frames_parsed++;
            buffer.erase(buffer.begin(), buffer.begin() + 32);
        }
    }
};

TEST(ImuParser, ParseSingleValidFrame) {
    ImuParser parser;
    auto frame = build_imu_frame(1.0f, 0.5f, -0.3f, 0.1f, 0.2f, 0.3f);

    parser.feed(frame.data(), frame.size());

    EXPECT_EQ(parser.frames_parsed, 1u);
    EXPECT_EQ(parser.crc_errors, 0u);
    EXPECT_EQ(parser.header_rejects, 0u);
    EXPECT_EQ(parser.footer_rejects, 0u);

    ASSERT_EQ(parser.parsed_frames.size(), 1u);
    auto &pf = parser.parsed_frames.front();
    EXPECT_FLOAT_EQ(pf.yaw, 1.0f);
    EXPECT_FLOAT_EQ(pf.pitch, 0.5f);
    EXPECT_FLOAT_EQ(pf.roll, -0.3f);
    EXPECT_FLOAT_EQ(pf.wx, 0.1f);
    EXPECT_FLOAT_EQ(pf.wy, 0.2f);
    EXPECT_FLOAT_EQ(pf.wz, 0.3f);
}

TEST(ImuParser, RejectBadHeader) {
    ImuParser parser;
    auto frame = build_imu_frame(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 损坏帧头
    frame[0] = 0x00;
    parser.feed(frame.data(), frame.size());

    EXPECT_EQ(parser.frames_parsed, 0u);
    EXPECT_GT(parser.header_rejects, 0u);
}

TEST(ImuParser, RejectBadFooter) {
    ImuParser parser;
    auto frame = build_imu_frame(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 损坏帧尾
    frame[30] = 0x00;
    parser.feed(frame.data(), frame.size());

    EXPECT_EQ(parser.frames_parsed, 0u);
}

TEST(ImuParser, RejectBadCrc) {
    ImuParser parser;
    auto frame = build_imu_frame(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 翻转 CRC 字节
    frame[28] ^= 0xFF;
    parser.feed(frame.data(), frame.size());

    EXPECT_EQ(parser.frames_parsed, 0u);
    EXPECT_GT(parser.crc_errors, 0u);
}

TEST(ImuParser, ParseMultipleFrames) {
    ImuParser parser;
    auto frame = build_imu_frame(0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 一次性送入 5 帧
    std::vector<uint8_t> multi;
    for (int i = 0; i < 5; i++) {
        auto f = build_imu_frame(0.1f * i, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        multi.insert(multi.end(), f.begin(), f.end());
    }

    parser.feed(multi.data(), multi.size());

    EXPECT_EQ(parser.frames_parsed, 5u);
    EXPECT_EQ(parser.parsed_frames.size(), 5u);
}

TEST(ImuParser, SkipGarbageBeforeFrame) {
    ImuParser parser;
    auto frame = build_imu_frame(0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 在有效帧前添加垃圾字节
    std::vector<uint8_t> with_garbage = {0x00, 0x11, 0x22, 0x33, 0x44};
    with_garbage.insert(with_garbage.end(), frame.begin(), frame.end());

    parser.feed(with_garbage.data(), with_garbage.size());

    EXPECT_EQ(parser.frames_parsed, 1u);
    EXPECT_GE(parser.header_rejects, 5u);
}

TEST(ImuParser, PartialFrameAccumulation) {
    ImuParser parser;
    auto frame = build_imu_frame(0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 先送入前 10 字节（不完整帧）
    parser.feed(frame.data(), 10);
    EXPECT_EQ(parser.frames_parsed, 0u);
    EXPECT_EQ(parser.buffer.size(), 10u);

    // 再送入剩余字节
    parser.feed(frame.data() + 10, 22);
    EXPECT_EQ(parser.frames_parsed, 1u);
    EXPECT_EQ(parser.buffer.size(), 0u);
}

TEST(ImuParser, BufferOverflowTrimming) {
    ImuParser parser;

    // 送入 600 字节垃圾数据（超过 512 缓冲区限制）
    std::vector<uint8_t> garbage(600, 0x00);
    parser.feed(garbage.data(), garbage.size());

    // 缓冲区应被裁剪到 512
    EXPECT_LE(parser.buffer.size(), 512u);
}

TEST(ImuParser, MixedValidAndInvalid) {
    ImuParser parser;
    auto valid = build_imu_frame(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    std::vector<uint8_t> mixed;
    // 3 垃圾字节, 有效帧, 5 垃圾字节, 有效帧, 2 垃圾字节
    mixed.insert(mixed.end(), {0x00, 0x00, 0x00});
    mixed.insert(mixed.end(), valid.begin(), valid.end());
    mixed.insert(mixed.end(), {0xFF, 0xFF, 0xFF, 0xFF, 0xFF});
    mixed.insert(mixed.end(), valid.begin(), valid.end());
    mixed.insert(mixed.end(), {0xAA, 0xAA});

    parser.feed(mixed.data(), mixed.size());

    EXPECT_EQ(parser.frames_parsed, 2u);
}

TEST(ImuParser, FrameHeaderAtNonZeroOffset) {
    // 帧头字节 0xEB, 0x90, 0xA5, 0xFF 出现在非零偏移位置
    // 验证解析器能正确拒绝
    ImuParser parser;

    std::vector<uint8_t> tricky(32, 0x00);
    tricky[0] = 0x00;  // 非帧头
    tricky[1] = 0xEB;  // 伪帧头起始
    tricky[2] = 0x90;

    parser.feed(tricky.data(), tricky.size());
    // 解析器应逐字节拒绝直到帧边界对齐
    // 预期不会崩溃
    SUCCEED();
}

// ═══════════════════════════════════════════════════════════════════════════
// 数据完整性测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(ImuDataIntegrity, RpyOrderMatchesTypesH) {
    // 验证 rpy[0]=roll, rpy[1]=pitch, rpy[2]=yaw（匹配 Types.h 第 363-365 行）
    float yaw = 1.5f, pitch = 0.3f, roll = -0.2f;
    auto frame = build_imu_frame(yaw, pitch, roll, 0.0f, 0.0f, 0.0f);

    float decoded[6];
    std::memcpy(decoded, &frame[4], 24);

    // Types.h: imu_state_cache.rpy[0] = roll; rpy[1] = pitch; rpy[2] = yaw;
    EXPECT_FLOAT_EQ(decoded[2], roll);   // rpy[0] = data[2] = roll
    EXPECT_FLOAT_EQ(decoded[1], pitch);  // rpy[1] = data[1] = pitch
    EXPECT_FLOAT_EQ(decoded[0], yaw);    // rpy[2] = data[0] = yaw
}

TEST(ImuDataIntegrity, QuaternionMatchesEulerToQuaternion) {
    float yaw = 1.2f, pitch = -0.4f, roll = 0.6f;
    auto frame = build_imu_frame(yaw, pitch, roll, 0.0f, 0.0f, 0.0f);

    float expected_q[4];
    euler_to_quaternion(yaw, pitch, roll, expected_q);

    ImuParser parser;
    parser.feed(frame.data(), frame.size());

    ASSERT_EQ(parser.parsed_frames.size(), 1u);
    auto &pf = parser.parsed_frames.front();

    EXPECT_NEAR(pf.quat[0], expected_q[0], 1e-5f);
    EXPECT_NEAR(pf.quat[1], expected_q[1], 1e-5f);
    EXPECT_NEAR(pf.quat[2], expected_q[2], 1e-5f);
    EXPECT_NEAR(pf.quat[3], expected_q[3], 1e-5f);
}

// ═══════════════════════════════════════════════════════════════════════════
// 压力测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(ImuStress, ThousandFrames) {
    ImuParser parser;
    auto frame = build_imu_frame(0.1f, -0.1f, 0.05f, 0.01f, -0.01f, 0.0f);

    for (int i = 0; i < 1000; i++) {
        parser.feed(frame.data(), frame.size());
    }

    EXPECT_EQ(parser.frames_parsed, 1000u);
    EXPECT_EQ(parser.crc_errors, 0u);
}
