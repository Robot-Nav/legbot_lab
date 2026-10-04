cmake_minimum_required(VERSION 3.14)
project(robstride_pointfoot_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(robstride_pointfoot_bridge STATIC
        src/motor_config.cpp
        src/robstride.cpp
        src/socketcan_interface.cpp
        src/w190_canfd.cpp)

target_include_directories(robstride_pointfoot_bridge
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

add_executable(can_heartbeat
        src/can_heartbeat.cpp)

target_link_libraries(can_heartbeat
        PRIVATE
        robstride_pointfoot_bridge)

add_executable(can_service
        src/can_service.cpp)

target_link_libraries(can_service
        PRIVATE
        robstride_pointfoot_bridge
        pthread)

add_executable(can_service_wheel
        src/can_service_wheel.cpp)

target_link_libraries(can_service_wheel
        PRIVATE
        robstride_pointfoot_bridge
        pthread)

add_executable(joint_damping_test
        src/joint_damping_test.cpp)

target_link_libraries(joint_damping_test
        PRIVATE
        robstride_pointfoot_bridge
        pthread)

add_executable(single_joint_test
        src/single_joint_test.cpp)

target_link_libraries(single_joint_test
        PRIVATE
        robstride_pointfoot_bridge
        pthread)
