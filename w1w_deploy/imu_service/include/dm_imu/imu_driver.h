#ifndef DM_IMU_IMU_DRIVER_H_
#define DM_IMU_IMU_DRIVER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <termios.h>

#include "dm_imu/bsp_crc.h"

namespace dmbot_serial
{

#pragma pack(push, 1)
struct IMU_Receive_Frame
{
    uint8_t FrameHeader1;
    uint8_t flag1;
    uint8_t slave_id1;
    uint8_t reg_acc;
    uint32_t accx_u32;
    uint32_t accy_u32;
    uint32_t accz_u32;
    uint16_t crc1;
    uint8_t FrameEnd1;

    uint8_t FrameHeader2;
    uint8_t flag2;
    uint8_t slave_id2;
    uint8_t reg_gyro;
    uint32_t gyrox_u32;
    uint32_t gyroy_u32;
    uint32_t gyroz_u32;
    uint16_t crc2;
    uint8_t FrameEnd2;

    uint8_t FrameHeader3;
    uint8_t flag3;
    uint8_t slave_id3;
    uint8_t reg_euler;
    uint32_t roll_u32;
    uint32_t pitch_u32;
    uint32_t yaw_u32;
    uint16_t crc3;
    uint8_t FrameEnd3;
};
#pragma pack(pop)

struct IMU_Data
{
    float accx = 0.0F;
    float accy = 0.0F;
    float accz = 0.0F;
    float gyrox = 0.0F;
    float gyroy = 0.0F;
    float gyroz = 0.0F;
    float roll = 0.0F;
    float pitch = 0.0F;
    float yaw = 0.0F;
};

class DmImu
{
public:
    using DataCallback = std::function<void(const IMU_Data&)>;

    DmImu(std::string port = "/dev/ttyACM1", int baud = 921600, DataCallback callback = {});
    ~DmImu();

    DmImu(const DmImu&) = delete;
    DmImu& operator=(const DmImu&) = delete;

    void setCallback(DataCallback callback);
    bool getLatest(IMU_Data& out) const;
    void stop();
    bool running() const { return running_.load(); }

private:
    void start();
    void readerLoop();
    void initSerial();
    bool readBytes(uint8_t* buffer, std::size_t length, int timeout_ms);
    bool writeBytes(const uint8_t* buffer, std::size_t length);
    speed_t resolveBaud(int baud) const;

    void enter_setting_mode();
    void turn_on_accel();
    void turn_on_gyro();
    void turn_on_euler();
    void turn_off_quat();
    void set_output_1000HZ();
    void set_output_usb();
    void save_imu_para();
    void exit_setting_mode();
    void restart_imu();

    std::string imu_serial_port_;
    int imu_serial_baud_;
    int serial_fd_ = -1;
    std::thread reader_thread_;
    std::atomic_bool running_{false};
    std::atomic_bool stop_requested_{false};

    DataCallback callback_;

    mutable std::mutex data_mutex_;
    IMU_Data latest_data_{};
    bool has_data_ = false;

    IMU_Receive_Frame receive_frame_{};
};

} // namespace dmbot_serial

#endif // DM_IMU_IMU_DRIVER_H_


