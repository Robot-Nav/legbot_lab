#pragma once
#include <string>

namespace w1w {
namespace device_guard {

// 开发模式：设备授权检查直接通过，不验证CPU序列号
inline bool enforce(const std::string& /*component*/) {
    return true;
}

} // namespace device_guard
} // namespace w1w
