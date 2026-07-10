// 文件用途：Go2 部署类型定义。将宇树 SDK 的 LowCmd/LowState 发布订阅类型重命名，便于 FSM 使用。
#pragma once

#include "unitree/dds_wrapper/robots/go2/go2.h"

using LowCmd_t = unitree::robot::go2::publisher::LowCmd;       // 底层指令发布类型
using LowState_t = unitree::robot::go2::subscription::LowState; // 底层状态订阅类型