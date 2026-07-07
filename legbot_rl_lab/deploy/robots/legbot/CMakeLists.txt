cmake_minimum_required(VERSION 3.12)
project(legbot_controller)

# x86_64 / aarch64: pick matching ONNX Runtime under deploy/thirdparty/
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/DetectPlatform.cmake)

# ==================== 1. 查找依赖 ====================
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# ==================== 2. 头文件包含 ====================
include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx          # CycloneDDS C++ headers
  /usr/local/include/iceoryx/v2.0.2  # iceoryx headers
  $ENV{HOME}/.local/include          # unitree_sdk2 headers (dds_wrapper)
  ${PROJECT_SOURCE_DIR}/include
  include
  ${ONNXRUNTIME_ROOT}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

# ==================== 3. 构建静态库 ====================
file(GLOB_RECURSE ADD_SRC_LIST src/*.cpp)
add_library(${PROJECT_NAME}_lib STATIC ${ADD_SRC_LIST})

# ==================== 4. 构建可执行文件 ====================
add_executable(legbot_ctrl main.cpp)

# ==================== 5. 链接：使用 target_link_libraries 严格控制顺序 ====================
# 注意：被依赖的目标（如 unitree_sdk2、ddscxx）必须放在后面
target_link_libraries(legbot_ctrl
    ${PROJECT_NAME}_lib    # 1. 优先链接你自己的静态库
    unitree_sdk2            # 2. 宇树 SDK 库
    ddscxx                  # 3. CycloneDDS C++ 库
    ddsc                    # 4. CycloneDDS C 库
    rt
    pthread
    boost_program_options
    yaml-cpp
    fmt
    ${ONNXRUNTIME_LIB}
)
