#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace serial_dds_gateway {

inline const std::unordered_map<std::string_view, uint8_t> kJointToCanId = {
    {"FR_hip_joint", 11},   {"FR_thigh_joint", 21}, {"FR_calf_joint", 31}, {"FL_hip_joint", 12},
    {"FL_thigh_joint", 22}, {"FL_calf_joint", 32},  {"RR_hip_joint", 13},  {"RR_thigh_joint", 23},
    {"RR_calf_joint", 33},  {"RL_hip_joint", 14},   {"RL_thigh_joint", 24}, {"RL_calf_joint", 34},
};

inline const std::unordered_map<uint8_t, std::string_view> kCanIdToJoint = {
    {11, "FR_hip_joint"}, {21, "FR_thigh_joint"}, {31, "FR_calf_joint"}, {12, "FL_hip_joint"},
    {22, "FL_thigh_joint"}, {32, "FL_calf_joint"}, {13, "RR_hip_joint"}, {23, "RR_thigh_joint"},
    {33, "RR_calf_joint"}, {14, "RL_hip_joint"}, {24, "RL_thigh_joint"}, {34, "RL_calf_joint"},
};

inline constexpr std::array<std::string_view, 12> kJointOrder = {
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
};

enum class MotorSerialBus { A, B };

inline constexpr std::array<uint8_t, 6> kMotorBusACanIds = {11, 21, 31, 13, 23, 33};  // FR + RR
inline constexpr std::array<uint8_t, 6> kMotorBusBCanIds = {12, 22, 32, 14, 24, 34};  // FL + RL

inline bool MotorSerialBusContains(MotorSerialBus bus, uint8_t motor_id) {
  const auto& ids = (bus == MotorSerialBus::A) ? kMotorBusACanIds : kMotorBusBCanIds;
  for (const auto id : ids) {
    if (id == motor_id) return true;
  }
  return false;
}

inline MotorSerialBus MotorSerialBusForCanId(uint8_t motor_id) {
  return MotorSerialBusContains(MotorSerialBus::A, motor_id) ? MotorSerialBus::A : MotorSerialBus::B;
}

}  // namespace serial_dds_gateway

