#pragma once

// 物理仿真输入设备适配头文件
// 将 XBOX、Switch 手柄或键盘输入映射为宇树 UnitreeJoystick 统一接口，
// 供 bridge 线程发布无线控制器状态。

#include <iostream>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include "joystick/joystick.h"
#include <memory>
#include <GLFW/glfw3.h>

// MuJoCo 渲染窗口全局句柄，KeyboardJoystick 通过 glfwGetKey 读取按键
extern GLFWwindow* g_sim_window;


// XBOX 手柄适配：将 Linux js_event 按键/轴索引映射到 UnitreeJoystick
class XBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    XBoxJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);  // 有符号摇杆最大值，用于归一化
	}

    void update() override
    {
        js_->getState();
        // 功能按键映射
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]);
        X(js_->button_[2]);
        Y(js_->button_[3]);
        // 方向键为轴 6/7，负值表示上/左
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        // 扳机为轴 2/5，压下后值大于 0
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        // 摇杆轴 0/1/3/4，Y 轴取反以符合前正后负约定
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};

// 键盘适配：使用 GLFW 按键模拟手柄，便于无手柄时调试
class KeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
	KeyboardJoystick(GLFWwindow* window) : unitree::common::UnitreeJoystick(), window_(window) {}

	void update() override
	{
		if (!window_) return;

		float lx_val = 0.0f, ly_val = 0.0f, rx_val = 0.0f, ry_val = 0.0f;

		// WASD 控制左摇杆（机体前后左右平移）
		if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) ly_val += 1.0f;
		if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) ly_val -= 1.0f;
		if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) lx_val -= 1.0f;
		if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) lx_val += 1.0f;

		// 方向键控制右摇杆（机体旋转/俯仰）
		if (glfwGetKey(window_, GLFW_KEY_UP)    == GLFW_PRESS) ry_val += 1.0f;
		if (glfwGetKey(window_, GLFW_KEY_DOWN)  == GLFW_PRESS) ry_val -= 1.0f;
		if (glfwGetKey(window_, GLFW_KEY_LEFT)  == GLFW_PRESS) rx_val -= 1.0f;
		if (glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) rx_val += 1.0f;

		float lt_val = (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS) ? 1.0f : 0.0f;
		float rt_val = (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS) ? 1.0f : 0.0f;

		lx(lx_val);
		ly(ly_val);
		rx(rx_val);
		ry(ry_val);
		LT(lt_val);
		RT(rt_val);

		// 动作按键映射
		A((glfwGetKey(window_, GLFW_KEY_SPACE)       == GLFW_PRESS) ? 1 : 0);
		B((glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS) ? 1 : 0);
		X((glfwGetKey(window_, GLFW_KEY_F)           == GLFW_PRESS) ? 1 : 0);
		Y((glfwGetKey(window_, GLFW_KEY_G)           == GLFW_PRESS) ? 1 : 0);
		LB((glfwGetKey(window_, GLFW_KEY_C)          == GLFW_PRESS) ? 1 : 0);
		RB((glfwGetKey(window_, GLFW_KEY_V)          == GLFW_PRESS) ? 1 : 0);
		start((glfwGetKey(window_, GLFW_KEY_ENTER)   == GLFW_PRESS) ? 1 : 0);
		back((glfwGetKey(window_, GLFW_KEY_TAB)      == GLFW_PRESS) ? 1 : 0);
		LS((glfwGetKey(window_, GLFW_KEY_Z)          == GLFW_PRESS) ? 1 : 0);
		RS((glfwGetKey(window_, GLFW_KEY_X)          == GLFW_PRESS) ? 1 : 0);
		F1((glfwGetKey(window_, GLFW_KEY_1)          == GLFW_PRESS) ? 1 : 0);
		F2((glfwGetKey(window_, GLFW_KEY_2)          == GLFW_PRESS) ? 1 : 0);

		// 方向键同时映射到方向按键位，保持与手柄语义一致
		up((glfwGetKey(window_, GLFW_KEY_UP)         == GLFW_PRESS) ? 1 : 0);
		down((glfwGetKey(window_, GLFW_KEY_DOWN)     == GLFW_PRESS) ? 1 : 0);
		left((glfwGetKey(window_, GLFW_KEY_LEFT)     == GLFW_PRESS) ? 1 : 0);
		right((glfwGetKey(window_, GLFW_KEY_RIGHT)   == GLFW_PRESS) ? 1 : 0);
	}

private:
	GLFWwindow* window_;
};


// Switch Pro 手柄适配，轴/按键索引与 XBOX 不同
class SwitchJoystick : public unitree::common::UnitreeJoystick
{
public:
    SwitchJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);  // 有符号摇杆最大值，用于归一化
	}

    void update() override
    {
        js_->getState();
        // 功能按键映射
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]);
        X(js_->button_[3]);
        Y(js_->button_[4]);
        // 方向键为轴 6/7
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        // Switch 扳机由轴 4/5 表示
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        // 摇杆轴 0/1/2/3，Y 轴取反
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};