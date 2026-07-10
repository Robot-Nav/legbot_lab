# 文件用途：Go2 控制器构建脚本。编译静态库与可执行文件，并链接 DDS、ONNX Runtime 等依赖。
cmake_minimum_required(VERSION 3.12)
project(go2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx              # CycloneDDS C++ 头文件
  /usr/local/include/iceoryx/v2.0.2      # iceoryx 头文件
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread    # DDS 相关库
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)