# Detect host CPU and select prebuilt third-party package names.
# Used by legbot_ctrl (ONNX Runtime). Build on the target machine (x86_64 PC or aarch64 Orange Pi).

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
