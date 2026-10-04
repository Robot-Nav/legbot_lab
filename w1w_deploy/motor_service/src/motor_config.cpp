//
// Created by Mu Shibo on 04/2024
//

#include <chrono>
#include <utility>

#include "motor_control/motor_config.h"

namespace {
uint64_t steady_now_ms()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace

// Motor parameter limits.
// mpointfoot 机器狗: 所有关节(ID 1=calf, 2=thigh, 3=hip)电机型号均为 RS04，
// 统一使用同一套 RS04 限位。
const MotorLimits& getMotorLimits(uint8_t motor_id) {
    static const MotorLimits rs04_limits = {
        -12.57f, 12.57f,   // P_MIN, P_MAX
        -15.0f, 15.0f,    // V_MIN, V_MAX
        0.0f, 5000.0f,    // KP_MIN, KP_MAX
        0.0f, 100.0f,     // KD_MIN, KD_MAX
        -120.0f, 120.0f   // T_MIN, T_MAX
    };

    (void)motor_id;  // 所有电机均为 RS04，使用统一限位
    return rs04_limits;
}

MotorControlSet::MotorControlSet(SendCanFrameFn send_frame,
                                 const std::vector<uint8_t>& motor_ids,
                                 FeedbackCallback feedback_callback)
    : send_frame_(std::move(send_frame)),
      feedback_callback_(std::move(feedback_callback))
{
    if (motor_ids.empty())
    {
        throw std::invalid_argument("MotorControlSet expects at least one motor id");
    }

    motor_ids_ = motor_ids;
    joints_.reserve(motor_ids.size());
    for (uint8_t id : motor_ids)
    {
        joints_.push_back(std::make_unique<RobStrite_Motor>(id, send_frame_));
    }

    joint_feedback_.pos.resize(motor_ids.size());
    joint_feedback_.vel.resize(motor_ids.size());
    joint_feedback_.tor.resize(motor_ids.size());
    joint_feedback_.temp.resize(motor_ids.size());
    joint_feedback_.error_code.resize(motor_ids.size());
    joint_feedback_.pattern.resize(motor_ids.size());
    joint_feedback_.last_rx_ms.resize(motor_ids.size(), 0);
    motor_state.resize(motor_ids.size());
    motor_cmd.resize(motor_ids.size());
}

MotorControlSet::~MotorControlSet()
{
    shutdown();
}

void MotorControlSet::shutdown()
{
    for (auto &j : joints_)
    {
        if (j)
            j->Disenable_Motor(0);
    }
}

void MotorControlSet::handleIncomingFrame(const CanFrame& frame)
{
    const uint8_t motor_id = static_cast<uint8_t>((frame.id & 0xFF00) >> 8);
    for (std::size_t i = 0; i < motor_ids_.size(); ++i)
    {
        if (motor_id == motor_ids_[i])
        {
            if (joints_[i])
            {
                joints_[i]->RobStrite_Motor_Analysis(frame.data.data(), frame.id);
                joint_feedback_.pos[i] = joints_[i]->Pos_Info.Angle;
                joint_feedback_.vel[i] = joints_[i]->Pos_Info.Speed;
                joint_feedback_.tor[i] = joints_[i]->Pos_Info.Torque;
                joint_feedback_.temp[i] = joints_[i]->Pos_Info.Temp;
                joint_feedback_.error_code[i] = joints_[i]->error_code;
                joint_feedback_.pattern[i] = joints_[i]->Pos_Info.pattern;
                joint_feedback_.last_rx_ms[i] = steady_now_ms();
            }
            break;
        }
    }

    if (feedback_callback_)
    {
        feedback_callback_(joint_feedback_);
    }
}

void MotorControlSet::forwardCommand(const CanFrame& frame) const
{
    if (send_frame_)
    {
        send_frame_(frame);
    }
}

void MotorControlSet::setFeedbackCallback(FeedbackCallback callback)
{
    feedback_callback_ = std::move(callback);
}

RobStrite_Motor& MotorControlSet::getMotor(uint8_t ID)
{
    for (std::size_t i = 0; i < motor_ids_.size(); ++i)
    {
        if (ID == motor_ids_[i])
        {
            if (!joints_[i])
                throw std::out_of_range("Motor object missing");
            return *joints_[i];
        }
    }
    throw std::out_of_range("Unknown motor id");
}
