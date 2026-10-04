#include "dm_imu/imu_driver.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <string>

#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

namespace dmbot_serial
{
namespace
{
constexpr float kDegToRad = 3.14159265358979323846F / 180.0F;
constexpr uint8_t kUsbFrameHeader = 0x55;
constexpr uint8_t kUsbFrameFlag = 0xAA;
constexpr uint8_t kFrameTypeAccel = 0x01;
constexpr uint8_t kFrameTypeGyro = 0x02;
constexpr uint8_t kFrameTypeEuler = 0x03;
constexpr uint8_t kFrameTypeQuat = 0x04;
constexpr uint8_t kFrameEnd = 0x0A;
constexpr std::size_t kShortFrameSize = 19U;
constexpr std::size_t kQuatFrameSize = 23U;

float to_float(uint32_t value)
{
    float out = 0.0F;
    std::memcpy(&out, &value, sizeof(float));
    return out;
}

uint16_t load_u16_le(const uint8_t* data)
{
    return static_cast<uint16_t>(data[0]) |
           (static_cast<uint16_t>(data[1]) << 8);
}

float load_float_le(const uint8_t* data)
{
    uint32_t raw = static_cast<uint32_t>(data[0]) |
                   (static_cast<uint32_t>(data[1]) << 8) |
                   (static_cast<uint32_t>(data[2]) << 16) |
                   (static_cast<uint32_t>(data[3]) << 24);
    return to_float(raw);
}

void sleep_for_ms(int ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
} // namespace

DmImu::DmImu(std::string port, int baud, DataCallback callback)
    : imu_serial_port_(std::move(port)),
      imu_serial_baud_(baud),
      callback_(std::move(callback))
{
    start();
}

DmImu::~DmImu()
{
    stop();
}

void DmImu::setCallback(DataCallback callback)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    callback_ = std::move(callback);
}

bool DmImu::getLatest(IMU_Data& out) const
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!has_data_)
    {
        return false;
    }
    out = latest_data_;
    return true;
}

void DmImu::stop()
{
    stop_requested_.store(true);
    if (reader_thread_.joinable())
    {
        reader_thread_.join();
    }
    running_.store(false);
    if (serial_fd_ >= 0)
    {
        ::close(serial_fd_);
        serial_fd_ = -1;
    }
}

void DmImu::start()
{
    stop_requested_.store(false);
    initSerial();

    // USB CDC IMUs can reboot when the ACM device is opened. Give the
    // device time to finish booting before sending configuration commands.
    sleep_for_ms(500);

    enter_setting_mode();
    sleep_for_ms(100);

    set_output_usb();
    sleep_for_ms(50);

    turn_on_accel();
    sleep_for_ms(50);

    turn_on_gyro();
    sleep_for_ms(50);

    turn_on_euler();
    sleep_for_ms(50);

    turn_off_quat();
    sleep_for_ms(50);

    set_output_1000HZ();
    sleep_for_ms(50);

    save_imu_para();
    sleep_for_ms(100);

    exit_setting_mode();
    sleep_for_ms(200);

    // Some DM-IMU-L1 units keep streaming configuration/status frames until
    // the USB CDC port is reopened after leaving setting mode.
    if (serial_fd_ >= 0)
    {
        ::close(serial_fd_);
        serial_fd_ = -1;
    }

    sleep_for_ms(300);
    initSerial();
    sleep_for_ms(300);

    if (tcflush(serial_fd_, TCIFLUSH) != 0)
    {
        std::cerr << "Warning: tcflush(TCIFLUSH) failed for " << imu_serial_port_
                  << ": " << std::strerror(errno) << std::endl;
    }

    running_.store(true);
    reader_thread_ = std::thread(&DmImu::readerLoop, this);
    std::cout << "IMU initialized on " << imu_serial_port_ << " at " << imu_serial_baud_ << " baud" << std::endl;
}

void DmImu::readerLoop()
{
    int read_error_count = 0;
    int unexpected_frame_count = 0;
    IMU_Data pending_data{};
    bool have_acc = false;
    bool have_gyro = false;
    bool have_euler = false;
    auto note_read_error = [&read_error_count]()
    {
        ++read_error_count;
        if (read_error_count > 1200)
        {
            std::cerr << "Failed to synchronize with supported IMU USB data frames" << std::endl;
            read_error_count = 0;
        }
    };

    while (!stop_requested_.load())
    {
        if (serial_fd_ < 0)
        {
            std::cerr << "IMU serial port closed unexpectedly" << std::endl;
            sleep_for_ms(100);
            continue;
        }

        uint8_t frame[kQuatFrameSize] = {};
        std::size_t matched = 0;
        while (matched < 4 && !stop_requested_.load())
        {
            uint8_t byte = 0;
            if (!readBytes(&byte, 1, 100))
            {
                note_read_error();
                continue;
            }

            if (matched == 0)
            {
                if (byte == kUsbFrameHeader)
                {
                    frame[0] = byte;
                    matched = 1;
                }
                continue;
            }

            if (matched == 1)
            {
                if (byte == kUsbFrameFlag)
                {
                    frame[1] = byte;
                    matched = 2;
                }
                else if (byte == kUsbFrameHeader)
                {
                    frame[0] = byte;
                    matched = 1;
                }
                else
                {
                    matched = 0;
                }
                continue;
            }

            if (matched == 2)
            {
                frame[2] = byte;
                matched = 3;
                continue;
            }

            frame[3] = byte;
            matched = 4;
        }

        if (stop_requested_.load())
        {
            break;
        }

        if (matched < 4)
        {
            continue;
        }

        const uint8_t frame_type = frame[3];
        std::size_t frame_size = 0;
        if (frame_type == kFrameTypeAccel ||
            frame_type == kFrameTypeGyro ||
            frame_type == kFrameTypeEuler)
        {
            frame_size = kShortFrameSize;
        }
        else if (frame_type == kFrameTypeQuat)
        {
            frame_size = kQuatFrameSize;
        }
        else
        {
            ++unexpected_frame_count;
            if (unexpected_frame_count == 1 || (unexpected_frame_count % 200) == 0)
            {
                std::cerr << "Received unexpected IMU USB frame type 0x"
                          << std::hex << static_cast<int>(frame_type)
                          << std::dec
                          << "; expected data frame types 0x01-0x04" << std::endl;
            }
            continue;
        }

        if (!readBytes(frame + 4, frame_size - 4, 100))
        {
            note_read_error();
            continue;
        }

        if (frame[frame_size - 1] != kFrameEnd)
        {
            note_read_error();
            continue;
        }

        const uint16_t received_crc = load_u16_le(frame + frame_size - 3);
        const uint16_t expected_crc = Get_CRC16(frame, frame_size - 3);
        if (received_crc != expected_crc)
        {
            note_read_error();
            continue;
        }

        if (frame_type == kFrameTypeAccel)
        {
            pending_data.accx = load_float_le(frame + 4);
            pending_data.accy = load_float_le(frame + 8);
            pending_data.accz = load_float_le(frame + 12);
            have_acc = true;
        }
        else if (frame_type == kFrameTypeGyro)
        {
            pending_data.gyrox = load_float_le(frame + 4);
            pending_data.gyroy = load_float_le(frame + 8);
            pending_data.gyroz = load_float_le(frame + 12);
            have_gyro = true;
        }
        else if (frame_type == kFrameTypeEuler)
        {
            pending_data.roll = load_float_le(frame + 4) * kDegToRad;
            pending_data.pitch = load_float_le(frame + 8) * kDegToRad;
            pending_data.yaw = load_float_le(frame + 12) * kDegToRad;
            have_euler = true;
        }

        read_error_count = 0;
        unexpected_frame_count = 0;

        if (frame_type == kFrameTypeEuler && have_acc && have_gyro && have_euler)
        {
            DataCallback callback_copy;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_data_ = pending_data;
                has_data_ = true;
                callback_copy = callback_;
            }
            if (callback_copy)
            {
                callback_copy(pending_data);
            }

            have_euler = false;
        }
    }

    running_.store(false);
}

void DmImu::initSerial()
{
    if (serial_fd_ >= 0)
    {
        ::close(serial_fd_);
        serial_fd_ = -1;
    }

    serial_fd_ = ::open(imu_serial_port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (serial_fd_ < 0)
    {
        throw std::runtime_error("Unable to open IMU serial port " + imu_serial_port_ + ": " + std::strerror(errno));
    }

    termios tty{};
    if (tcgetattr(serial_fd_, &tty) != 0)
    {
        const int err = errno;
        ::close(serial_fd_);
        serial_fd_ = -1;
        throw std::runtime_error("tcgetattr failed: " + std::string(std::strerror(err)));
    }

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);

    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_lflag = 0;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1; // 0.1s timeout on read

    const speed_t speed = resolveBaud(imu_serial_baud_);
    if (cfsetispeed(&tty, speed) != 0 || cfsetospeed(&tty, speed) != 0)
    {
        const int err = errno;
        ::close(serial_fd_);
        serial_fd_ = -1;
        throw std::runtime_error("Failed to set IMU baud rate: " + std::string(std::strerror(err)));
    }

    if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0)
    {
        const int err = errno;
        ::close(serial_fd_);
        serial_fd_ = -1;
        throw std::runtime_error("tcsetattr failed: " + std::string(std::strerror(err)));
    }

    if (tcflush(serial_fd_, TCIOFLUSH) != 0)
    {
        std::cerr << "Warning: tcflush failed for " << imu_serial_port_ << ": " << std::strerror(errno) << std::endl;
    }
}

void DmImu::enter_setting_mode()
{
    const uint8_t txbuf[4] = {0xAA, 0x06, 0x01, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (enter_setting_mode)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::turn_on_accel()
{
    const uint8_t txbuf[4] = {0xAA, 0x01, 0x14, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (turn_on_accel)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::turn_on_gyro()
{
    const uint8_t txbuf[4] = {0xAA, 0x01, 0x15, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (turn_on_gyro)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::turn_on_euler()
{
    const uint8_t txbuf[4] = {0xAA, 0x01, 0x16, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (turn_on_euler)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::turn_off_quat()
{
    const uint8_t txbuf[4] = {0xAA, 0x01, 0x07, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (turn_off_quat)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::set_output_1000HZ()
{
    const uint8_t txbuf[5] = {0xAA, 0x02, 0x01, 0x00, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (set_output_1000HZ)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::set_output_usb()
{
    const uint8_t txbuf[4] = {0xAA, 0x0A, 0x00, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (set_output_usb)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::save_imu_para()
{
    const uint8_t txbuf[4] = {0xAA, 0x03, 0x01, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (save_imu_para)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::exit_setting_mode()
{
    const uint8_t txbuf[4] = {0xAA, 0x06, 0x00, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (exit_setting_mode)");
        }
        sleep_for_ms(10);
    }
}

void DmImu::restart_imu()
{
    const uint8_t txbuf[4] = {0xAA, 0x00, 0x00, 0x0D};
    for (int i = 0; i < 5; ++i)
    {
        if (!writeBytes(txbuf, sizeof(txbuf)))
        {
            throw std::runtime_error("Failed to send IMU command (restart_imu)");
        }
        sleep_for_ms(10);
    }
}

speed_t DmImu::resolveBaud(int baud) const
{
#ifdef B921600
    if (baud == 921600)
    {
        return B921600;
    }
#endif
#ifdef B1000000
    if (baud == 1000000)
    {
        return B1000000;
    }
#endif
#ifdef B1152000
    if (baud == 1152000)
    {
        return B1152000;
    }
#endif
#ifdef B1500000
    if (baud == 1500000)
    {
        return B1500000;
    }
#endif
#ifdef B2000000
    if (baud == 2000000)
    {
        return B2000000;
    }
#endif

    switch (baud)
    {
    case 9600:
        return B9600;
    case 19200:
        return B19200;
    case 38400:
        return B38400;
    case 57600:
        return B57600;
    case 115200:
        return B115200;
    case 230400:
        return B230400;
    case 460800:
        return B460800;
    default:
        throw std::runtime_error("Unsupported baud rate: " + std::to_string(baud));
    }
}

bool DmImu::readBytes(uint8_t* buffer, std::size_t length, int timeout_ms)
{
    if (serial_fd_ < 0)
    {
        return false;
    }

    std::size_t total = 0;
    const auto deadline = timeout_ms < 0
                               ? std::chrono::steady_clock::time_point::max()
                               : std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (total < length && !stop_requested_.load())
    {
        int poll_timeout = -1;
        if (timeout_ms >= 0)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
            {
                return false;
            }
            poll_timeout = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        }

        pollfd pfd{};
        pfd.fd = serial_fd_;
        pfd.events = POLLIN;

        const int poll_result = ::poll(&pfd, 1, poll_timeout);
        if (poll_result < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        if (poll_result == 0)
        {
            return false;
        }

        const ssize_t n = ::read(serial_fd_, buffer + total, length - total);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        if (n == 0)
        {
            continue;
        }

        total += static_cast<std::size_t>(n);
    }

    return total == length;
}

bool DmImu::writeBytes(const uint8_t* buffer, std::size_t length)
{
    if (serial_fd_ < 0)
    {
        return false;
    }

    std::size_t total = 0;
    while (total < length)
    {
        const ssize_t n = ::write(serial_fd_, buffer + total, length - total);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        total += static_cast<std::size_t>(n);
    }

    if (tcdrain(serial_fd_) != 0)
    {
        return false;
    }

    return true;
}

} // namespace dmbot_serial


