//
// Created by Mu Shibo on 04/2024
//

#ifndef _MOTOR_CONFIG_H_
#define _MOTOR_CONFIG_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "motor_control/can_frame.h"
#include "motor_control/robstride.h"

#include <memory>

// Joint_Num is deprecated; MotorControlSet now supports a dynamic number of motors.

typedef struct 
{
	uint16_t state;
	float pos;
	float vel;
	float tor;
	float Kp;
	float Kd;
	float Tmos;
	float Tcoil;
}motor_state_t;

typedef struct 
{
	float pos_d;
	float vel_d;
	float tor_d;
	float Kp;
	float Kd;
}motor_cmd_t;

struct MotorFeedback
{
	std::vector<float> pos;
	std::vector<float> vel;
	std::vector<float> tor;
	std::vector<float> temp;
	std::vector<uint8_t> error_code;
	std::vector<int8_t> pattern;
	// 每个电机最近一次收到反馈帧(通信类型2)的时间(steady_clock 毫秒)，
	// 用于判断 CAN 通信是否在线。0 表示从未收到。
	std::vector<uint64_t> last_rx_ms;
};

class MotorControlSet
{
    public:
		using FeedbackCallback = std::function<void(const MotorFeedback&)>;

		MotorControlSet(SendCanFrameFn send_frame,
						const std::vector<uint8_t>& motor_ids,
						FeedbackCallback feedback_callback = nullptr);
		~MotorControlSet();

		void shutdown();
		void handleIncomingFrame(const CanFrame& frame);
		void forwardCommand(const CanFrame& frame) const;
		void setFeedbackCallback(FeedbackCallback callback);

	std::vector<motor_state_t> motor_state;
	std::vector<motor_cmd_t> motor_cmd;

		const MotorFeedback& getFeedback() const { return joint_feedback_; }
		RobStrite_Motor& getMotor(uint8_t ID);

	private:
	std::vector<std::unique_ptr<RobStrite_Motor>> joints_;
	std::vector<uint8_t> motor_ids_;
		SendCanFrameFn send_frame_;
		FeedbackCallback feedback_callback_{};
		MotorFeedback joint_feedback_;
};




#endif