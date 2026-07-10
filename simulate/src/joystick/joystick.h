// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright Drew Noakes 2013-2016

// Linux 手柄事件读取头文件
// 封装 /dev/input/js* 非阻塞读取，提供按键与摇杆状态缓存。

#ifndef __JOYSTICK_H__
#define __JOYSTICK_H__

#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <map>
#include "unistd.h"


#define JS_EVENT_BUTTON 0x01 // 按键按下或释放
#define JS_EVENT_AXIS 0x02   // 摇杆或扳机轴变化
#define JS_EVENT_INIT 0x80   // 设备初始状态事件

// 单个手柄事件结构，对应 linux/joystick.h 的 js_event
class JoystickEvent
{
public:
  static const short MIN_AXES_VALUE = -32768;  // 摇杆最小值
  static const short MAX_AXES_VALUE = 32767;   // 摇杆最大值

  unsigned int time;   // 事件时间戳，单位毫秒
  short value;         // 事件数值：按键为 0/1，轴为 MIN_AXES_VALUE ~ MAX_AXES_VALUE
  unsigned char type;  // 事件类型：按键、轴或初始状态
  unsigned char number;// 按键或轴编号

  // 判断是否为按键事件
  bool isButton()
  {
    return (type & JS_EVENT_BUTTON) != 0;
  }

  // 判断是否为轴事件
  bool isAxis()
  {
    return (type & JS_EVENT_AXIS) != 0;
  }

  // 判断是否为设备连接时注入的初始状态事件
  bool isInitialState()
  {
    return (type & JS_EVENT_INIT) != 0;
  }

  // 声明友元，允许流输出访问内部字段
  friend std::ostream &operator<<(std::ostream &os, const JoystickEvent &e);
};

// 流输出函数，便于直接打印调试：cout << event << endl;
std::ostream &operator<<(std::ostream &os, const JoystickEvent &e);

// Linux 手柄设备封装类
class Joystick
{
private:
  void openPath(std::string devicePath, bool blocking = false);
  int _fd;  // 设备文件描述符

public:
  ~Joystick();

  // 默认打开第一个手柄 /dev/input/js0
  Joystick();

  // 根据编号打开 /dev/input/js{joystickNumber}
  Joystick(int joystickNumber);

  // 根据完整设备路径打开
  Joystick(std::string devicePath);

  // 禁止拷贝，避免文件描述符重复关闭
  Joystick(Joystick const &) = delete;

  // 允许移动语义
  Joystick(Joystick &&) = default;

  // 指定设备路径，并可选择阻塞模式
  Joystick(std::string devicePath, bool blocking);

  // 检测手柄是否成功打开
  bool isFound();

  // 读取一次事件并更新内部缓存，应在主循环中高频调用
  void getState();

  JoystickEvent event_;  // 最近一次事件
  int button_[20] = {0}; // 按键状态缓存，索引对应手柄按键编号
  int axis_[10] = {0};   // 轴状态缓存，索引对应手柄轴编号

  // 从内核读取单个事件，返回 true 表示读取成功
  bool sample(JoystickEvent *event);
};

#endif
