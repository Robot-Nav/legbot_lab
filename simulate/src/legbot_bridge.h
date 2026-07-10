#pragma once

// MuJoCo 与宇树 DDS 之间的桥接头文件
// 订阅实机或控制算法下发的 LowCmd，将力矩/位置/速度指令写入 MuJoCo 执行器；
// 同时以 1 kHz 发布 LowState、HighState 和 WirelessController，实现软实时闭环。

#include <mujoco/mujoco.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/go2/go2.h>

#include <iostream>

#include "param.h"
#include "physics_joystick.h"

// 每个电机的传感器数量：位置、速度、力矩估计
#define MOTOR_SENSOR_NUM 3

// 桥接基类：负责传感器地址解析、场景信息打印及输入设备初始化
class VBotBridgeBase
{
public:
    VBotBridgeBase(mjModel *model, mjData *data)
    : mj_model_(model), mj_data_(data)
    {
        _check_sensor();
        if(param::config.print_scene_information == 1) {
            printSceneInformation();
        }
        if(param::config.use_joystick == 1) {
            if(param::config.joystick_type == "xbox") {
                joystick = std::make_shared<XBoxJoystick>(param::config.joystick_device, param::config.joystick_bits);
            } else if(param::config.joystick_type == "switch") {
                joystick  = std::make_shared<SwitchJoystick>(param::config.joystick_device, param::config.joystick_bits);
            } else if(param::config.joystick_type == "keyboard") {
                joystick  = std::make_shared<KeyboardJoystick>(g_sim_window);
            } else {
                std::cerr << "Unsupported joystick type: " << param::config.joystick_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    virtual void start() {}

    // 打印模型中连杆、关节、执行器和传感器的索引与名称，辅助核对传感器地址
    void printSceneInformation()
    {
        auto printObjects = [this](const char* title, int count, int type, auto getIndex) {
            std::cout << "<<------------- " << title << " ------------->> " << std::endl;
            for (int i = 0; i < count; i++) {
                const char* name = mj_id2name(mj_model_, type, i);
                if (name) {
                    std::cout << title << "_index: " << getIndex(i) << ", " << "name: " << name;
                    if (type == mjOBJ_SENSOR) {
                        std::cout << ", dim: " << mj_model_->sensor_dim[i];
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        };

        printObjects("Link", mj_model_->nbody, mjOBJ_BODY, [](int i) { return i; });
        printObjects("Joint", mj_model_->njnt, mjOBJ_JOINT, [](int i) { return i; });
        printObjects("Actuator", mj_model_->nu, mjOBJ_ACTUATOR, [](int i) { return i; });

        int sensorIndex = 0;
        printObjects("Sensor", mj_model_->nsensor, mjOBJ_SENSOR, [&](int i) {
            int currentIndex = sensorIndex;
            sensorIndex += mj_model_->sensor_dim[i];
            return currentIndex;
        });
    }

protected:
    int num_motor_ = 0;           // 执行器/电机数量
    int dim_motor_sensor_ = 0;    // 电机相关传感器总维度

    mjData *mj_data_;
    mjModel *mj_model_;

    // 各类传感器在 mj_data_->sensordata 中的起始索引，-1 表示未找到
    int imu_quat_adr_ = -1;
    int imu_gyro_adr_ = -1;
    int imu_acc_adr_ = -1;
    int frame_pos_adr_ = -1;
    int frame_vel_adr_ = -1;

    std::shared_ptr<unitree::common::UnitreeJoystick> joystick = nullptr;

    // 通过名称查找传感器地址，兼容不同 XML 的命名习惯（trunk_* / imu_* / base_* / frame_*）
    void _check_sensor()
    {
        num_motor_ = mj_model_->nu;
        dim_motor_sensor_ = MOTOR_SENSOR_NUM * num_motor_;

        int sensor_id = -1;

        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "trunk_quat");
        if (sensor_id < 0) {
            sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_quat");
        }
        if (sensor_id >= 0) {
            imu_quat_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "base_gyro");
        if (sensor_id < 0) {
            sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_gyro");
        }
        if (sensor_id >= 0) {
            imu_gyro_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "trunk_acc");
        if (sensor_id < 0) {
            sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_acc");
        }
        if (sensor_id >= 0) {
            imu_acc_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "trunk_pos");
        if (sensor_id < 0) {
            sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "frame_pos");
        }
        if (sensor_id >= 0) {
            frame_pos_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "base_linvel");
        if (sensor_id < 0) {
            sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "frame_vel");
        }
        if (sensor_id >= 0) {
            frame_vel_adr_ = mj_model_->sensor_adr[sensor_id];
        }
    }
};

// DDS 消息类型别名，对应宇树 Go2 的低层控制与状态话题
using LowCmd_t = unitree::robot::go2::subscription::LowCmd;                  // 订阅：电机指令
using LowState_t = unitree::robot::go2::publisher::LowState;                 // 发布：电机与 IMU 状态
using HighState_t = unitree::robot::go2::publisher::SportModeState;          // 发布：机体位姿速度
using WirelessController_t = unitree::robot::go2::publisher::WirelessController; // 发布：无线手柄状态

// DDS-MuJoCo 桥接实现类
class VBotBridge : public VBotBridgeBase
{
public:
    VBotBridge(mjModel *model, mjData *data) : VBotBridgeBase(model, data)
    {
        // 创建 DDS 订阅/发布对象；主题名与宇树实机保持一致
        lowcmd = std::make_shared<LowCmd_t>("rt/lowcmd");
        lowstate = std::make_unique<LowState_t>();
        lowstate->joystick = joystick;
        highstate = std::make_unique<HighState_t>();
        wireless_controller = std::make_unique<WirelessController_t>();
        wireless_controller->joystick = joystick;
    }

    // 启动 1 kHz 周期线程，与宇树实机控制频率对齐
    void start()
    {
        thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "legbot_bridge", UT_CPU_ID_NONE, 1000, [this]() { this->run(); });
    }

    // 每周期执行：更新手柄、应用 LowCmd、发布 LowState/HighState/WirelessController
    void run()
    {
        if(!mj_data_) return;
        if(lowstate->joystick) { lowstate->joystick->update(); }

        // 将 LowCmd 中的力矩、位置、速度指令通过 PD 控制器转换为执行器控制量
        {
            std::lock_guard<std::mutex> lock(lowcmd->mutex_);
            for(int i(0); i<num_motor_; i++) {
                auto & m = lowcmd->msg_.motor_cmd()[i];
                mj_data_->ctrl[i] = m.tau() +
                                    m.kp() * (m.q() - mj_data_->sensordata[i]) +
                                    m.kd() * (m.dq() - mj_data_->sensordata[i + num_motor_]);
            }
        }

        // 发布 LowState：电机位置/速度/力矩 + IMU 四元数/陀螺仪/加速度
        if(lowstate->trylock()) {
            for(int i(0); i<num_motor_; i++) {
                lowstate->msg_.motor_state()[i].q() = mj_data_->sensordata[i];
                lowstate->msg_.motor_state()[i].dq() = mj_data_->sensordata[i + num_motor_];
                lowstate->msg_.motor_state()[i].tau_est() = mj_data_->sensordata[i + 2 * num_motor_];
            }

            if(imu_quat_adr_ >= 0) {
                lowstate->msg_.imu_state().quaternion()[0] = mj_data_->sensordata[imu_quat_adr_ + 0];
                lowstate->msg_.imu_state().quaternion()[1] = mj_data_->sensordata[imu_quat_adr_ + 1];
                lowstate->msg_.imu_state().quaternion()[2] = mj_data_->sensordata[imu_quat_adr_ + 2];
                lowstate->msg_.imu_state().quaternion()[3] = mj_data_->sensordata[imu_quat_adr_ + 3];

                double w = lowstate->msg_.imu_state().quaternion()[0];
                double x = lowstate->msg_.imu_state().quaternion()[1];
                double y = lowstate->msg_.imu_state().quaternion()[2];
                double z = lowstate->msg_.imu_state().quaternion()[3];

                // 四元数转欧拉角 RPY，供上层控制使用
                lowstate->msg_.imu_state().rpy()[0] = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                lowstate->msg_.imu_state().rpy()[1] = asin(2 * (w * y - z * x));
                lowstate->msg_.imu_state().rpy()[2] = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
            }

            if(imu_gyro_adr_ >= 0) {
                lowstate->msg_.imu_state().gyroscope()[0] = mj_data_->sensordata[imu_gyro_adr_ + 0];
                lowstate->msg_.imu_state().gyroscope()[1] = mj_data_->sensordata[imu_gyro_adr_ + 1];
                lowstate->msg_.imu_state().gyroscope()[2] = mj_data_->sensordata[imu_gyro_adr_ + 2];
            }

            if(imu_acc_adr_ >= 0) {
                lowstate->msg_.imu_state().accelerometer()[0] = mj_data_->sensordata[imu_acc_adr_ + 0];
                lowstate->msg_.imu_state().accelerometer()[1] = mj_data_->sensordata[imu_acc_adr_ + 1];
                lowstate->msg_.imu_state().accelerometer()[2] = mj_data_->sensordata[imu_acc_adr_ + 2];
            }

            // tick 以 1 ms 为单位，与宇树协议保持一致
            lowstate->msg_.tick() = std::round(mj_data_->time / 1e-3);
            lowstate->unlockAndPublish();
        }

        // 发布 HighState：机体系在世界系中的位置与速度
        if(highstate->trylock()) {
            if(frame_pos_adr_ >= 0) {
                highstate->msg_.position()[0] = mj_data_->sensordata[frame_pos_adr_ + 0];
                highstate->msg_.position()[1] = mj_data_->sensordata[frame_pos_adr_ + 1];
                highstate->msg_.position()[2] = mj_data_->sensordata[frame_pos_adr_ + 2];
            }
            if(frame_vel_adr_ >= 0) {
                highstate->msg_.velocity()[0] = mj_data_->sensordata[frame_vel_adr_ + 0];
                highstate->msg_.velocity()[1] = mj_data_->sensordata[frame_vel_adr_ + 1];
                highstate->msg_.velocity()[2] = mj_data_->sensordata[frame_vel_adr_ + 2];
            }
            highstate->unlockAndPublish();
        }

        // 发布无线控制器状态，手柄更新已在 run 开头完成
        if(wireless_controller->joystick) {
            wireless_controller->unlockAndPublish();
        }
    }

    std::unique_ptr<HighState_t> highstate;
    std::unique_ptr<WirelessController_t> wireless_controller;
    std::shared_ptr<LowCmd_t> lowcmd;
    std::unique_ptr<LowState_t> lowstate;

private:
    unitree::common::RecurrentThreadPtr thread_;
};
