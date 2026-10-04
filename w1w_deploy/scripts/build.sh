#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

command -v g++ >/dev/null || { echo "g++ is required" >&2; exit 1; }
: "${W1W_AUTHORIZED_CPU_DIGEST_HEX:?set the 64-character sealed CPU digest}"
if [[ ! ${W1W_AUTHORIZED_CPU_DIGEST_HEX} =~ ^[0-9a-f]{64}$ ]]; then
  echo "W1W_AUTHORIZED_CPU_DIGEST_HEX must be 64 lowercase hex characters" >&2
  exit 1
fi

GUARD_FLAGS=(
  "-DW1W_AUTHORIZED_CPU_DIGEST_HEX=\"${W1W_AUTHORIZED_CPU_DIGEST_HEX}\""
  -I"${ROOT_DIR}/device_guard/include"
  "${ROOT_DIR}/device_guard/src/device_guard.cpp"
)

install -d "${BUILD_DIR}/bin"
g++ -std=c++17 -O2 -DNDEBUG -pthread \
  "${GUARD_FLAGS[@]}" \
  -I"${ROOT_DIR}/motor_service/include" \
  "${ROOT_DIR}/motor_service/src/motor_config.cpp" \
  "${ROOT_DIR}/motor_service/src/robstride.cpp" \
  "${ROOT_DIR}/motor_service/src/socketcan_interface.cpp" \
  "${ROOT_DIR}/motor_service/src/w190_canfd.cpp" \
  "${ROOT_DIR}/motor_service/src/can_service_wheel.cpp" \
  -lcrypto -o "${BUILD_DIR}/bin/can_service_wheel"

g++ -std=c++17 -O2 -DNDEBUG -pthread \
  "${GUARD_FLAGS[@]}" \
  -I"${ROOT_DIR}/imu_service/include" \
  "${ROOT_DIR}/imu_service/src/bsp_crc.cpp" \
  "${ROOT_DIR}/imu_service/src/imu_driver.cpp" \
  "${ROOT_DIR}/imu_service/src/imu_service.cpp" \
  -lcrypto -o "${BUILD_DIR}/bin/imu_service"

g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wpedantic -pthread \
  "${GUARD_FLAGS[@]}" \
  -I"${ROOT_DIR}/elrs_service/include" \
  "${ROOT_DIR}/elrs_service/src/elrs_service.cpp" \
  "${ROOT_DIR}/elrs_service/src/elrs_driver.cpp" \
  -lcrypto -o "${BUILD_DIR}/bin/elrs_service"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread \
  -I"${ROOT_DIR}/elrs_service/include" \
  "${ROOT_DIR}/elrs_service/tests/test_crsf_parser.cpp" \
  "${ROOT_DIR}/elrs_service/src/elrs_driver.cpp" \
  -o "${BUILD_DIR}/bin/test_crsf_parser"
"${BUILD_DIR}/bin/test_crsf_parser"

echo "built can_service_wheel, imu_service and elrs_service"
