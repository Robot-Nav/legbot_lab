# serial_dds_gateway 构建配置：核心库 + 测试/工具可执行文件 + DDS 网关。
cmake_minimum_required(VERSION 3.12)
project(serial_dds_gateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "serial_dds_gateway: CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64)$")
  message(WARNING "未在此架构测试: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 核心静态库：协议编解码、串口帧、电机协议、IMU 帧。
add_library(serial_dds_gateway_core
  src/protocol_codec.cpp
  src/can_id_codec.cpp
  src/serial_framer.cpp
  src/lingzu_motor_protocol.cpp
  src/imu_framer.cpp
)
target_include_directories(serial_dds_gateway_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 本地编解码自测。
add_executable(one_motor_demo src/one_motor_demo.cpp)
target_link_libraries(one_motor_demo PRIVATE serial_dds_gateway_core)

# 单电机串口收发测试。
add_executable(one_motor_serial src/one_motor_serial.cpp)
target_link_libraries(one_motor_serial PRIVATE serial_dds_gateway_core)

# 双串口 12 电机测试。
add_executable(twelve_motor_serial src/twelve_motor_serial.cpp)
target_link_libraries(twelve_motor_serial PRIVATE serial_dds_gateway_core)

# 灵足串口帧回归测试。
add_executable(lingzu_frame_verify src/lingzu_frame_verify.cpp)
target_link_libraries(lingzu_frame_verify PRIVATE serial_dds_gateway_core)

# IMU 帧回归测试。
add_executable(imu_frame_verify src/imu_frame_verify.cpp)
target_link_libraries(imu_frame_verify PRIVATE serial_dds_gateway_core)

# IMU 串口实时监听。
add_executable(imu_serial_monitor src/imu_serial_monitor.cpp)
target_link_libraries(imu_serial_monitor PRIVATE serial_dds_gateway_core)

# DDS 网关可执行文件（依赖 unitree_sdk2 与 CycloneDDS 头文件/库）。
add_executable(dds_to_serial_gateway src/dds_to_serial_gateway.cpp)
target_include_directories(dds_to_serial_gateway PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  $ENV{HOME}/.local/include
)
target_link_libraries(dds_to_serial_gateway PRIVATE
  serial_dds_gateway_core
  unitree_sdk2
  ddscxx
  ddsc
  fmt
  pthread
  rt
)

