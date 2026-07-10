# 文件用途：Legbot 控制器构建脚本。编译静态库与可执行文件，并链接 DDS、ONNX Runtime 等依赖。
cmake_minimum_required(VERSION 3.12)
project(legbot_controller)

# 根据 x86_64 / aarch64 选择 deploy/thirdparty/ 下对应的 ONNX Runtime
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/DetectPlatform.cmake)

# 1. 查找依赖
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# 2. 头文件包含
include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx          # CycloneDDS C++ 头文件
  /usr/local/include/iceoryx/v2.0.2  # iceoryx 头文件
  $ENV{HOME}/.local/include          # unitree_sdk2 头文件（dds_wrapper）
  ${PROJECT_SOURCE_DIR}/include
  include
  ${ONNXRUNTIME_ROOT}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

# 3. 构建静态库
file(GLOB_RECURSE ADD_SRC_LIST src/*.cpp)
add_library(${PROJECT_NAME}_lib STATIC ${ADD_SRC_LIST})

# 4. 构建可执行文件
add_executable(legbot_ctrl main.cpp)

# 5. 链接：使用 target_link_libraries 严格控制顺序；被依赖库放在后面
target_link_libraries(legbot_ctrl
    ${PROJECT_NAME}_lib     # 1. 自身静态库
    unitree_sdk2            # 2. 宇树 SDK
    ddscxx                  # 3. CycloneDDS C++ 库
    ddsc                    # 4. CycloneDDS C 库
    rt
    pthread
    boost_program_options
    yaml-cpp
    fmt
    ${ONNXRUNTIME_LIB}
)
