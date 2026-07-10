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

// Linux 手柄事件读取实现
// 打开 /dev/input/js* 设备，非阻塞轮询事件并维护按键/轴状态。

#include "joystick.h"

Joystick::Joystick()
{
  openPath("/dev/input/js0");
}

Joystick::Joystick(int joystickNumber)
{
  std::stringstream sstm;
  sstm << "/dev/input/js" << joystickNumber;
  openPath(sstm.str());
}

Joystick::Joystick(std::string devicePath)
{
  openPath(devicePath);
}

Joystick::Joystick(std::string devicePath, bool blocking)
{
  openPath(devicePath, blocking);
}

void Joystick::openPath(std::string devicePath, bool blocking)
{
  // 非阻塞模式避免读取时阻塞主循环；阻塞模式仅在不依赖渲染线程时使用
  _fd = open(devicePath.c_str(), blocking ? O_RDONLY : O_RDONLY | O_NONBLOCK);
}

bool Joystick::sample(JoystickEvent *event)
{
  int bytes = read(_fd, event, sizeof(*event));

  if (bytes == -1)
    return false;

  // 读取长度必须与结构体一致，否则说明事件流失步，应重新打开设备
  return bytes == sizeof(*event);
}

bool Joystick::isFound()
{
  return _fd >= 0;
}

void Joystick::getState()
{
  if (sample(&event_))
  {
    if (event_.isButton())
    {
      button_[event_.number] = event_.value;
    }
    else if (event_.isAxis())
    {
      axis_[event_.number] = event_.value;
    }
  }
}

Joystick::~Joystick()
{
  close(_fd);
}

std::ostream &operator<<(std::ostream &os, const JoystickEvent &e)
{
  os << "type=" << static_cast<int>(e.type)
     << " number=" << static_cast<int>(e.number)
     << " value=" << static_cast<int>(e.value);
  return os;
}
