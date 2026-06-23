#pragma once

#include <unitree/dds_wrapper/robots/go2/go2.h>
#include "robstride_motor.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <termios.h>
#include <cmath>
#include <algorithm>

#define HEIGHT_SCAN_PORT 19876
#define HEIGHT_SCAN_SIZE 187

struct MotorCmd {
    float q = 0.0f;
    float dq = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float tau = 0.0f;
};

struct MotorState {
    float q = 0.0f;
    float dq = 0.0f;
    float tau_est = 0.0f;
};

struct IMUState {
    float quaternion[4] = {1, 0, 0, 0};
    float gyroscope[3] = {0, 0, 0};
    float accelerometer[3] = {0, 0, 0};
    float rpy[3] = {0, 0, 0};
};

struct MotorSerialConfig {
    std::string serial_port;
    uint8_t channel;
    int baudrate;
    std::vector<uint8_t> motor_ids;
};

struct IMUConfig {
    std::string serial_port;
    int baudrate;
};

using LowCmdPublisher = unitree::robot::go2::publisher::LowCmd;
using LowStateSubscriber = unitree::robot::go2::subscription::LowState;

class VBotInterface {
public:
    enum Mode { SIM2SIM, SIM2REAL };
    Mode mode;

    VBotInterface(Mode m) : mode(m) {}
    virtual ~VBotInterface() = default;

    virtual void init() = 0;
    virtual void send_cmd(const std::vector<MotorCmd> &cmds) = 0;
    virtual void get_state(std::vector<MotorState> &states, IMUState &imu) = 0;
    virtual void enable_motors() = 0;
    virtual void disable_motors() = 0;
    virtual bool is_timeout() = 0;
    virtual void update() = 0;
    virtual unitree::common::UnitreeJoystick* get_joystick() = 0;

    virtual std::vector<float> get_height_scan() { return std::vector<float>(HEIGHT_SCAN_SIZE, 0.0f); }
};

class Sim2SimInterface : public VBotInterface {
public:
    std::shared_ptr<LowCmdPublisher> dds_lowcmd;
    std::shared_ptr<LowStateSubscriber> dds_lowstate;

    int height_scan_sock_ = -1;
    std::vector<float> height_scan_data_;

    Sim2SimInterface() : VBotInterface(SIM2SIM) {
        dds_lowcmd = std::make_shared<LowCmdPublisher>();
        dds_lowstate = std::make_shared<LowStateSubscriber>();
        height_scan_data_.resize(HEIGHT_SCAN_SIZE, 0.0f);
    }

    void init() override {
        spdlog::info("Sim2Sim: Waiting for DDS connection...");
        dds_lowstate->wait_for_connection();
        spdlog::info("Sim2Sim: Connected.");

        height_scan_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (height_scan_sock_ >= 0) {
            int flags = fcntl(height_scan_sock_, F_GETFL, 0);
            fcntl(height_scan_sock_, F_SETFL, flags | O_NONBLOCK);

            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(HEIGHT_SCAN_PORT);
            addr.sin_addr.s_addr = INADDR_ANY;

            if (bind(height_scan_sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
                close(height_scan_sock_);
                height_scan_sock_ = -1;
                spdlog::warn("Sim2Sim: HeightScan UDP bind failed");
            } else {
                spdlog::info("Sim2Sim: HeightScan UDP receiver on port {}", HEIGHT_SCAN_PORT);
            }
        }
    }

    void send_cmd(const std::vector<MotorCmd> &cmds) override {
        dds_lowcmd->lock();
        for (int i = 0; i < (int)cmds.size() && i < (int)dds_lowcmd->msg_.motor_cmd().size(); i++) {
            auto &m = dds_lowcmd->msg_.motor_cmd()[i];
            m.q() = cmds[i].q;
            m.dq() = cmds[i].dq;
            m.kp() = cmds[i].kp;
            m.kd() = cmds[i].kd;
            m.tau() = cmds[i].tau;
        }
        dds_lowcmd->unlockAndPublish();
    }

    void get_state(std::vector<MotorState> &states, IMUState &imu) override {
        for (int i = 0; i < (int)states.size() && i < (int)dds_lowstate->msg_.motor_state().size(); i++) {
            states[i].q = dds_lowstate->msg_.motor_state()[i].q();
            states[i].dq = dds_lowstate->msg_.motor_state()[i].dq();
            states[i].tau_est = dds_lowstate->msg_.motor_state()[i].tau_est();
        }
        for (int i = 0; i < 4; i++) {
            imu.quaternion[i] = dds_lowstate->msg_.imu_state().quaternion()[i];
        }
        for (int i = 0; i < 3; i++) {
            imu.gyroscope[i] = dds_lowstate->msg_.imu_state().gyroscope()[i];
            imu.accelerometer[i] = dds_lowstate->msg_.imu_state().accelerometer()[i];
            imu.rpy[i] = dds_lowstate->msg_.imu_state().rpy()[i];
        }
    }

    bool is_timeout() override { return dds_lowstate->isTimeout(); }

    void update() override {
        dds_lowstate->update();

        if (height_scan_sock_ >= 0) {
            float buffer[HEIGHT_SCAN_SIZE];
            while (true) {
                ssize_t n = recvfrom(height_scan_sock_, buffer, sizeof(buffer), MSG_DONTWAIT, NULL, NULL);
                if (n == (ssize_t)sizeof(buffer)) {
                    memcpy(height_scan_data_.data(), buffer, sizeof(buffer));
                } else {
                    break;
                }
            }
        }
    }

    unitree::common::UnitreeJoystick* get_joystick() override {
        return &dds_lowstate->joystick;
    }

    std::vector<float> get_height_scan() override {
        return height_scan_data_;
    }

    void enable_motors() override {}
    void disable_motors() override {}
};

class Sim2RealInterface : public VBotInterface {
public:
    std::vector<MotorSerialConfig> motor_serial_configs;
    IMUConfig imu_config;
    uint8_t master_id;
    int actuator_type;
    std::vector<float> zero_offsets;

    std::vector<int> serial_fds;
    int imu_fd = -1;
    std::vector<uint8_t> imu_buffer;

    std::vector<std::unique_ptr<RobStrideMotor>> motors;
    std::vector<MotorState> motor_states;
    IMUState imu_state_cache;

    Sim2RealInterface(
        const std::vector<MotorSerialConfig> &motor_configs,
        const IMUConfig &imu_cfg,
        uint8_t mid,
        int atype,
        const std::vector<float> &offsets
    ) : VBotInterface(SIM2REAL), motor_serial_configs(motor_configs),
        imu_config(imu_cfg), master_id(mid), actuator_type(atype),
        zero_offsets(offsets) {
        int total_motors = 0;
        for (auto &cfg : motor_serial_configs) total_motors += (int)cfg.motor_ids.size();
        motor_states.resize(total_motors);
    }

    ~Sim2RealInterface() {
        for (int fd : serial_fds) {
            if (fd >= 0) close(fd);
        }
        if (imu_fd >= 0) close(imu_fd);
    }

    void init() override {
        for (auto &cfg : motor_serial_configs) {
            int fd = open_serial_port(cfg.serial_port, cfg.baudrate);
            if (fd < 0) {
                spdlog::error("Sim2Real: Failed to open motor serial port {}", cfg.serial_port);
                continue;
            }
            serial_fds.push_back(fd);
            for (auto &mid : cfg.motor_ids) {
                motors.push_back(std::make_unique<RobStrideMotor>(
                    fd, cfg.channel, master_id, mid, actuator_type));
            }
            spdlog::info("Sim2Real: {} motors on {} (ch={}, baud={})",
                         cfg.motor_ids.size(), cfg.serial_port, (int)cfg.channel, cfg.baudrate);
        }
        spdlog::info("Sim2Real: Initialized {} motors on {} serial ports",
                     motors.size(), serial_fds.size());

        imu_fd = open_serial_port(imu_config.serial_port, imu_config.baudrate);
        if (imu_fd < 0) {
            spdlog::warn("Sim2Real: Failed to open IMU serial port {}", imu_config.serial_port);
        } else {
            spdlog::info("Sim2Real: IMU serial port {} opened (baud={})",
                         imu_config.serial_port, imu_config.baudrate);
        }
    }

    void send_cmd(const std::vector<MotorCmd> &cmds) override {
        for (int i = 0; i < (int)cmds.size() && i < (int)motors.size(); i++) {
            float q_offset = (i < (int)zero_offsets.size()) ? zero_offsets[i] : 0.0f;
            motors[i]->send_motion_command(
                cmds[i].tau,
                cmds[i].q + q_offset,
                cmds[i].dq,
                cmds[i].kp,
                cmds[i].kd);
        }
    }

    void get_state(std::vector<MotorState> &states, IMUState &imu) override {
        for (int i = 0; i < (int)motors.size() && i < (int)motor_states.size(); i++) {
            float q_offset = (i < (int)zero_offsets.size()) ? zero_offsets[i] : 0.0f;
            motor_states[i].q = motors[i]->get_position() - q_offset;
            motor_states[i].dq = motors[i]->get_velocity();
            motor_states[i].tau_est = motors[i]->get_torque();
        }
        states = motor_states;
        imu = imu_state_cache;
    }

    bool is_timeout() override { return false; }

    void update() override {
        if (imu_fd >= 0) {
            read_imu();
        }
    }

    unitree::common::UnitreeJoystick* get_joystick() override { return nullptr; }

    void enable_motors() override {
        for (auto &m : motors) {
            m->set_mode(move_control_mode);
            usleep(1000);
            m->enable_motor();
            usleep(1000);
        }
        spdlog::info("Sim2Real: All motors enabled.");
    }

    void disable_motors() override {
        for (auto &m : motors) {
            m->disable_motor(0);
        }
        spdlog::info("Sim2Real: All motors disabled.");
    }

private:
    static uint16_t crc16_modbus(const uint8_t *data, size_t length) {
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

    static void euler_to_quaternion(float yaw, float pitch, float roll, float q[4]) {
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

    void read_imu() {
        uint8_t buf[256];
        int n = read(imu_fd, buf, sizeof(buf));
        if (n > 0) {
            imu_buffer.insert(imu_buffer.end(), buf, buf + n);
        }

        if (imu_buffer.size() > 512) {
            imu_buffer.erase(imu_buffer.begin(), imu_buffer.begin() + (imu_buffer.size() - 512));
        }

        while (imu_buffer.size() >= 32) {
            if (imu_buffer[0] != 0xEB || imu_buffer[1] != 0x90 ||
                imu_buffer[2] != 0xA5 || imu_buffer[3] != 0xFF) {
                imu_buffer.erase(imu_buffer.begin());
                continue;
            }

            if (imu_buffer[30] != 0x80 || imu_buffer[31] != 0x7F) {
                imu_buffer.erase(imu_buffer.begin());
                continue;
            }

            uint16_t received_crc = (uint16_t)imu_buffer[29] << 8 | (uint16_t)imu_buffer[28];
            uint16_t calculated_crc = crc16_modbus(imu_buffer.data(), 28);
            if (received_crc != calculated_crc) {
                imu_buffer.erase(imu_buffer.begin());
                continue;
            }

            float data[6];
            memcpy(data, &imu_buffer[4], 24);
            float yaw   = data[0];
            float pitch = data[1];
            float roll  = data[2];
            float wx    = data[3];
            float wy    = data[4];
            float wz    = data[5];

            euler_to_quaternion(yaw, pitch, roll, imu_state_cache.quaternion);

            imu_state_cache.gyroscope[0] = wx;
            imu_state_cache.gyroscope[1] = wy;
            imu_state_cache.gyroscope[2] = wz;

            imu_state_cache.rpy[0] = roll;
            imu_state_cache.rpy[1] = pitch;
            imu_state_cache.rpy[2] = yaw;

            imu_buffer.erase(imu_buffer.begin(), imu_buffer.begin() + 32);
        }
    }

    static speed_t get_baud_constant(int baud) {
        switch (baud) {
            case 9600: return B9600;
            case 19200: return B19200;
            case 38400: return B38400;
            case 57600: return B57600;
            case 115200: return B115200;
            case 230400: return B230400;
            case 460800: return B460800;
            case 921600: return B921600;
            case 1000000: return B1000000;
            case 2000000: return B2000000;
            case 4000000: return B4000000;
            default: return B921600;
        }
    }

    static int open_serial_port(const std::string &port, int baud) {
        int fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (fd < 0) {
            perror("open serial port");
            return -1;
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        if (tcgetattr(fd, &tty) != 0) {
            perror("tcgetattr");
            close(fd);
            return -1;
        }

        speed_t baud_const = get_baud_constant(baud);
        cfsetospeed(&tty, baud_const);
        cfsetispeed(&tty, baud_const);

        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CREAD | CLOCAL;

        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;
        tty.c_oflag &= ~ONLCR;

        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;

        if (tcsetattr(fd, TCSANOW, &tty) != 0) {
            perror("tcsetattr");
            close(fd);
            return -1;
        }

        tcflush(fd, TCIOFLUSH);

        spdlog::info("Sim2Real: Serial port {} opened (baud={})", port, baud);
        return fd;
    }
};
