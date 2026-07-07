cmake_minimum_required(VERSION 3.12)
project(serial_dds_gateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "serial_dds_gateway: CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64)$")
  message(WARNING "Untested CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_library(serial_dds_gateway_core
  src/protocol_codec.cpp
  src/can_id_codec.cpp
  src/serial_framer.cpp
  src/lingzu_motor_protocol.cpp
  src/imu_framer.cpp
)
target_include_directories(serial_dds_gateway_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(UNITREE_SDK2_ROOT "" CACHE PATH "Optional local unitree_sdk2 root")
if(NOT UNITREE_SDK2_ROOT AND EXISTS "$ENV{HOME}/legbot_mpc_ws/fatuDog_orangepi_latest/unitree_sdk2/include/unitree")
  set(UNITREE_SDK2_ROOT "$ENV{HOME}/legbot_mpc_ws/fatuDog_orangepi_latest/unitree_sdk2")
elseif(NOT UNITREE_SDK2_ROOT AND EXISTS "$ENV{HOME}/rl_lab/unitree_sdk2/include/unitree")
  set(UNITREE_SDK2_ROOT "$ENV{HOME}/rl_lab/unitree_sdk2")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(UNITREE_SDK2_ARCH "aarch64")
else()
  set(UNITREE_SDK2_ARCH "x86_64")
endif()

set(SERIAL_DDS_GATEWAY_DDS_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  $ENV{HOME}/.local/include
)
set(SERIAL_DDS_GATEWAY_DDS_LIBS
  unitree_sdk2
  ddscxx
  ddsc
  fmt
  pthread
  rt
)
set(SERIAL_DDS_GATEWAY_DDS_RPATH "")

if(UNITREE_SDK2_ROOT)
  list(APPEND SERIAL_DDS_GATEWAY_DDS_INCLUDE_DIRS
    ${UNITREE_SDK2_ROOT}/include
    ${UNITREE_SDK2_ROOT}/thirdparty/include
    ${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx
  )
  find_library(UNITREE_SDK2_LIBRARY NAMES unitree_sdk2
    HINTS ${UNITREE_SDK2_ROOT}/lib/${UNITREE_SDK2_ARCH}
    NO_DEFAULT_PATH
  )
  find_library(UNITREE_DDSCXX_LIBRARY NAMES ddscxx
    HINTS ${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH}
    NO_DEFAULT_PATH
  )
  find_library(UNITREE_DDSC_LIBRARY NAMES ddsc
    HINTS ${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH}
    NO_DEFAULT_PATH
  )
  if(UNITREE_SDK2_LIBRARY AND UNITREE_DDSCXX_LIBRARY AND UNITREE_DDSC_LIBRARY)
    set(SERIAL_DDS_GATEWAY_DDS_LIBS
      ${UNITREE_SDK2_LIBRARY}
      ${UNITREE_DDSCXX_LIBRARY}
      ${UNITREE_DDSC_LIBRARY}
      fmt
      pthread
      rt
    )
    set(SERIAL_DDS_GATEWAY_DDS_RPATH
      ${UNITREE_SDK2_ROOT}/lib/${UNITREE_SDK2_ARCH}
      ${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH}
    )
    message(STATUS "serial_dds_gateway: using local unitree_sdk2 at ${UNITREE_SDK2_ROOT} (${UNITREE_SDK2_ARCH})")
  else()
    message(WARNING "serial_dds_gateway: UNITREE_SDK2_ROOT set but libraries were not found; using linker defaults")
  endif()
endif()

function(configure_go2_dds_target target_name)
  target_include_directories(${target_name} PRIVATE ${SERIAL_DDS_GATEWAY_DDS_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE
    serial_dds_gateway_core
    ${SERIAL_DDS_GATEWAY_DDS_LIBS}
  )
  if(SERIAL_DDS_GATEWAY_DDS_RPATH)
    set_target_properties(${target_name} PROPERTIES BUILD_RPATH "${SERIAL_DDS_GATEWAY_DDS_RPATH}")
  endif()
endfunction()

add_executable(one_motor_demo src/one_motor_demo.cpp)
target_link_libraries(one_motor_demo PRIVATE serial_dds_gateway_core)

add_executable(one_motor_serial src/one_motor_serial.cpp)
target_link_libraries(one_motor_serial PRIVATE serial_dds_gateway_core)

add_executable(twelve_motor_serial src/twelve_motor_serial.cpp)
target_link_libraries(twelve_motor_serial PRIVATE serial_dds_gateway_core)

add_executable(lingzu_frame_verify src/lingzu_frame_verify.cpp)
target_link_libraries(lingzu_frame_verify PRIVATE serial_dds_gateway_core)

add_executable(imu_frame_verify src/imu_frame_verify.cpp)
target_link_libraries(imu_frame_verify PRIVATE serial_dds_gateway_core)

add_executable(imu_serial_monitor src/imu_serial_monitor.cpp)
target_link_libraries(imu_serial_monitor PRIVATE serial_dds_gateway_core)

# DDS bridge target (depends on unitree_sdk2 headers/libs)
add_executable(dds_to_serial_gateway src/dds_to_serial_gateway.cpp)
configure_go2_dds_target(dds_to_serial_gateway)

add_executable(legbot_rt_gait_pd src/legbot_rt_gait_pd.cpp)
configure_go2_dds_target(legbot_rt_gait_pd)
