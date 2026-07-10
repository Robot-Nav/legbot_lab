# 检测主机 CPU 架构，并选择预编译第三方包名称。
# 供 legbot_ctrl（ONNX Runtime）使用。在目标机器（x86_64 PC 或 aarch64 香橙派）上构建。

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(LEGBOT_CPU_ARCH "aarch64")
  set(ONNXRUNTIME_DIR_NAME "onnxruntime-linux-aarch64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(LEGBOT_CPU_ARCH "x86_64")
  set(ONNXRUNTIME_DIR_NAME "onnxruntime-linux-x64-1.22.0")
else()
  message(FATAL_ERROR
    "Unsupported CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}'. "
    "Supported: x86_64, aarch64.")
endif()

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_LIST_DIR}/../thirdparty/${ONNXRUNTIME_DIR_NAME}")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(WARNING
    "ONNX Runtime not found for ${LEGBOT_CPU_ARCH} at:\n  ${ONNXRUNTIME_ROOT}\n"
    "Falling back to x64 package.")
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_LIST_DIR}/../thirdparty/onnxruntime-linux-x64-1.22.0")
endif()

file(GLOB _onnx_libs "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*" "${ONNXRUNTIME_ROOT}/lib64/libonnxruntime.so*")
if(NOT _onnx_libs)
  message(FATAL_ERROR "No libonnxruntime.so under ${ONNXRUNTIME_ROOT}/lib or lib64")
endif()
list(GET _onnx_libs 0 ONNXRUNTIME_LIB)

message(STATUS "Legbot deploy: CPU=${LEGBOT_CPU_ARCH}, ONNX Runtime=${ONNXRUNTIME_DIR_NAME}")
