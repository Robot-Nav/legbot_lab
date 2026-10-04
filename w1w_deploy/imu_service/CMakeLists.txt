cmake_minimum_required(VERSION 3.14)
project(dm_imu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_library(dm_imu STATIC
  src/bsp_crc.cpp
  src/imu_driver.cpp)

target_include_directories(dm_imu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dm_imu
  PRIVATE
    Threads::Threads)

add_executable(dm_imu_cli
  src/test_imu.cpp)

target_link_libraries(dm_imu_cli
  PRIVATE
    dm_imu)

add_executable(imu_service
  src/imu_service.cpp)

target_link_libraries(imu_service
  PRIVATE
    dm_imu)
