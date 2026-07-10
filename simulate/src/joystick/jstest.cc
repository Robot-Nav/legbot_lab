// 手柄测试程序
// 读取 /dev/input/js0 手柄输入，按宇树遥控器键位打包为 16 位按键掩码输出。

#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <map>
#include "joystick.h"

#define GAMEPAD_TYPE 1 // 1：XBOX 手柄，0：Switch 手柄
#define MAX_AXES_VALUE 32768
#define MIN_AXES_VALUE -32768
using namespace std;

// 宇树遥控器按键位域联合体，16 位同时传输所有按键状态
typedef union
{
  struct
  {
    uint8_t R1 : 1;
    uint8_t L1 : 1;
    uint8_t start : 1;
    uint8_t select : 1;
    uint8_t R2 : 1;
    uint8_t L2 : 1;
    uint8_t F1 : 1;
    uint8_t F2 : 1;
    uint8_t A : 1;
    uint8_t B : 1;
    uint8_t X : 1;
    uint8_t Y : 1;
    uint8_t up : 1;
    uint8_t right : 1;
    uint8_t down : 1;
    uint8_t left : 1;
  } components;
  uint16_t value;
} xKeySwitchUnion;

int main(int argc, char **argv)
{
  // 打开默认手柄设备
  Joystick joystick("/dev/input/js0");

  if (!joystick.isFound())
  {
    printf("open failed.\n");
    exit(1);
  }

  xKeySwitchUnion unitree_key;

  // 手柄轴编号映射：XBOX 协议下 LX/LY/RX/RY 与扳机、方向键的索引
  map<string, int> AxisId =
      {
          {"LX", 0}, // 左摇杆 X
          {"LY", 1}, // 左摇杆 Y
          {"RX", 3}, // 右摇杆 X
          {"RY", 4}, // 右摇杆 Y
          {"LT", 2}, // 左扳机
          {"RT", 5}, // 右扳机
          {"DX", 6}, // 方向键 X
          {"DY", 7}, // 方向键 Y
      };

  // 手柄按键编号映射
  map<string, int> ButtonId =
      {
          {"X", 2},
          {"Y", 3},
          {"B", 1},
          {"A", 0},
          {"LB", 4},
          {"RB", 5},
          {"SELECT", 6},
          {"START", 7},
      };

  while (true)
  {
    // 轮询手柄事件并更新缓存
    joystick.getState();

    unitree_key.components.R1 = joystick.button_[ButtonId["RB"]];
    unitree_key.components.L1 = joystick.button_[ButtonId["LB"]];
    unitree_key.components.start = joystick.button_[ButtonId["START"]];
    unitree_key.components.select = joystick.button_[ButtonId["SELECT"]];
    unitree_key.components.R2 = (joystick.axis_[AxisId["RT"]] > 0);
    unitree_key.components.L2 = (joystick.axis_[AxisId["LT"]] > 0);
    unitree_key.components.F1 = 0;
    unitree_key.components.F2 = 0;
    unitree_key.components.A = joystick.button_[ButtonId["A"]];
    unitree_key.components.B = joystick.button_[ButtonId["B"]];
    unitree_key.components.X = joystick.button_[ButtonId["X"]];
    unitree_key.components.Y = joystick.button_[ButtonId["Y"]];
    unitree_key.components.up = (joystick.axis_[AxisId["DY"]] < 0);
    unitree_key.components.right = (joystick.axis_[AxisId["DX"]] > 0);
    unitree_key.components.down = (joystick.axis_[AxisId["DY"]] > 0);
    unitree_key.components.left = (joystick.axis_[AxisId["DX"]] < 0);

    cout << unitree_key.value << endl;

    // 限制输出频率约 100 Hz，避免终端刷屏
    usleep(10000);
  }
  return 0;
};
