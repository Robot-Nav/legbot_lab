#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${ROOT_DIR}/build/sealed-build"
OUTPUT_DIR="${W1W_SEALED_OUTPUT:-${ROOT_DIR}/build/sealed-runtime}"
RUNTIME_PREFIX="${W1W_RUNTIME_PREFIX:-/opt/w1w/current}"

for command in g++ strip openssl sha256sum python3 python3-config cython3; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "sealed build requires ${command}" >&2
    exit 1
  }
done

if [[ $(uname -m) != aarch64 ]]; then
  echo "sealed build must run natively on the target ARM64 board" >&2
  exit 1
fi

CPU_SERIAL="$(awk -F: '$1 ~ /^[[:space:]]*Serial[[:space:]]*$/ {
  value=$2; gsub(/[[:space:]]/, "", value); print tolower(value); exit
}' /proc/cpuinfo)"
if [[ ! ${CPU_SERIAL} =~ ^[0-9a-f]{16}$ || ${CPU_SERIAL} == 0000000000000000 ]]; then
  echo "target Rockchip CPU serial is unavailable" >&2
  exit 1
fi

CPU_DIGEST="$(printf 'w1w-device-v1\0%s' "${CPU_SERIAL}" | sha256sum | awk '{print $1}')"
MODEL_MASTER_KEY="$(openssl rand -hex 32)"
if [[ ! ${CPU_DIGEST} =~ ^[0-9a-f]{64}$ || ! ${MODEL_MASTER_KEY} =~ ^[0-9a-f]{64}$ ]]; then
  echo "failed to generate sealed-build key material" >&2
  exit 1
fi

case "${BUILD_ROOT}" in
  "${ROOT_DIR}"/build/sealed-build) ;;
  *) echo "unsafe sealed build path: ${BUILD_ROOT}" >&2; exit 1 ;;
esac
case "${OUTPUT_DIR}" in
  "${ROOT_DIR}"/build/sealed-runtime|/tmp/w1w-sealed-runtime) ;;
  *) echo "unsafe sealed output path: ${OUTPUT_DIR}" >&2; exit 1 ;;
esac
rm -rf -- "${BUILD_ROOT}" "${OUTPUT_DIR}"
install -d -m 0755 \
  "${BUILD_ROOT}/controller" \
  "${OUTPUT_DIR}/bin" "${OUTPUT_DIR}/lib/w1w/controller" \
  "${OUTPUT_DIR}/systemd"

GUARD_SOURCE="${ROOT_DIR}/device_guard/src/device_guard.cpp"
GUARD_INCLUDE="${ROOT_DIR}/device_guard/include"
GUARD_DEFINE="-DW1W_AUTHORIZED_CPU_DIGEST_HEX=\"${CPU_DIGEST}\""
COMMON_FLAGS=(
  -std=c++17 -O3 -DNDEBUG -flto -fvisibility=hidden -pthread
  "${GUARD_DEFINE}" -I"${GUARD_INCLUDE}"
)

g++ "${COMMON_FLAGS[@]}" \
  -I"${ROOT_DIR}/motor_service/include" \
  "${GUARD_SOURCE}" \
  "${ROOT_DIR}/motor_service/src/motor_config.cpp" \
  "${ROOT_DIR}/motor_service/src/robstride.cpp" \
  "${ROOT_DIR}/motor_service/src/socketcan_interface.cpp" \
  "${ROOT_DIR}/motor_service/src/w190_canfd.cpp" \
  "${ROOT_DIR}/motor_service/src/can_service_wheel.cpp" \
  -lcrypto -o "${OUTPUT_DIR}/bin/can_service_wheel"

g++ "${COMMON_FLAGS[@]}" \
  -I"${ROOT_DIR}/imu_service/include" \
  "${GUARD_SOURCE}" \
  "${ROOT_DIR}/imu_service/src/bsp_crc.cpp" \
  "${ROOT_DIR}/imu_service/src/imu_driver.cpp" \
  "${ROOT_DIR}/imu_service/src/imu_service.cpp" \
  -lcrypto -o "${OUTPUT_DIR}/bin/imu_service"

g++ "${COMMON_FLAGS[@]}" -Wall -Wextra -Wpedantic \
  -I"${ROOT_DIR}/elrs_service/include" \
  "${GUARD_SOURCE}" \
  "${ROOT_DIR}/elrs_service/src/elrs_service.cpp" \
  "${ROOT_DIR}/elrs_service/src/elrs_driver.cpp" \
  -lcrypto -o "${OUTPUT_DIR}/bin/elrs_service"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread \
  -I"${ROOT_DIR}/elrs_service/include" \
  "${ROOT_DIR}/elrs_service/tests/test_crsf_parser.cpp" \
  "${ROOT_DIR}/elrs_service/src/elrs_driver.cpp" \
  -o "${BUILD_ROOT}/test_crsf_parser"
"${BUILD_ROOT}/test_crsf_parser"

cat > "${BUILD_ROOT}/negative_guard.cpp" <<'CPP'
#include "w1w/device_guard.h"
int main() { return w1w::device_guard::enforce("negative-license-test") ? 0 : 77; }
CPP
g++ -std=c++17 -O2 -DNDEBUG \
  '-DW1W_AUTHORIZED_CPU_DIGEST_HEX="0000000000000000000000000000000000000000000000000000000000000000"' \
  -I"${GUARD_INCLUDE}" "${GUARD_SOURCE}" "${BUILD_ROOT}/negative_guard.cpp" \
  -lcrypto -o "${BUILD_ROOT}/negative_guard"
set +e
"${BUILD_ROOT}/negative_guard" >/dev/null 2>&1
negative_status=$?
set -e
if [[ ${negative_status} -ne 77 ]]; then
  echo "negative CPU-license test failed with status ${negative_status}" >&2
  exit 1
fi

controller_sources=(config.py motor_client.py imu_client.py elrs_client.py safety.py deploy.py device_license.py)
for source in "${controller_sources[@]}"; do
  install -m 0600 "${ROOT_DIR}/controller/${source}" "${BUILD_ROOT}/controller/${source}"
done
sed -i \
  -e "s/__W1W_AUTHORIZED_CPU_DIGEST_HEX__/${CPU_DIGEST}/g" \
  -e "s/__W1W_MODEL_MASTER_KEY_HEX__/${MODEL_MASTER_KEY}/g" \
  "${BUILD_ROOT}/controller/device_license.py"
extension_suffix="$(python3-config --extension-suffix)"
read -r -a python_includes <<< "$(python3-config --includes)"
compile_extension() {
  local source_path="$1"
  local output_dir="$2"
  local module_name
  local generated_cpp
  module_name="$(basename "${source_path}" .py)"
  generated_cpp="${source_path%.py}.cpp"
  cython3 -3 --cplus -X embedsignature=False -X binding=False \
    -o "${generated_cpp}" "${source_path}"
  g++ -std=c++17 -O3 -DNDEBUG -flto -fPIC -shared -fvisibility=hidden \
    "${python_includes[@]}" "${generated_cpp}" \
    -o "${output_dir}/${module_name}${extension_suffix}"
}

for source in "${controller_sources[@]}"; do
  compile_extension \
    "${BUILD_ROOT}/controller/${source}" "${OUTPUT_DIR}/lib/w1w/controller"
done

CPU_SERIAL="${CPU_SERIAL}" MODEL_MASTER_KEY="${MODEL_MASTER_KEY}" \
python3 - "${ROOT_DIR}/controller/policy.onnx" \
  "${OUTPUT_DIR}/lib/w1w/controller/policy.onnx" <<'PY'
import hashlib
import os
import secrets
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

source = Path(sys.argv[1]).read_bytes()
serial = os.environ["CPU_SERIAL"].encode("ascii")
master_key = bytes.fromhex(os.environ["MODEL_MASTER_KEY"])
key = hashlib.sha256(b"w1w-model-key-v1\0" + master_key + b"\0" + serial).digest()
nonce = secrets.token_bytes(12)
ciphertext = AESGCM(key).encrypt(nonce, source, b"w1w-policy-v1")
Path(sys.argv[2]).write_bytes(b"W1WENC1\0" + nonce + ciphertext)
PY
read -r -a python_embed_ldflags <<< "$(python3-config --embed --ldflags)"
build_launcher() {
  local binary_name="$1"
  local module_name="$2"
  local module_subdir="$3"
  g++ "${COMMON_FLAGS[@]}" \
    "${python_includes[@]}" \
    "-DW1W_PYTHON_MODULE=\"${module_name}\"" \
    "-DW1W_MODULE_SUBDIR=\"${module_subdir}\"" \
    "${GUARD_SOURCE}" "${ROOT_DIR}/sealed_build/python_launcher.cpp" \
    -lcrypto "${python_embed_ldflags[@]}" \
    -o "${OUTPUT_DIR}/bin/${binary_name}"
}
build_launcher w1w-controller deploy controller

for template in "${ROOT_DIR}"/sealed_build/systemd/*.service; do
  sed "s|@PREFIX@|${RUNTIME_PREFIX}|g" "${template}" \
    > "${OUTPUT_DIR}/systemd/$(basename "${template}")"
done
install -m 0644 "${ROOT_DIR}/VERSION" "${OUTPUT_DIR}/VERSION"

find "${OUTPUT_DIR}" -type f \( -name '*.so' -o -path '*/bin/*' \) \
  -exec strip --strip-unneeded {} +
"${OUTPUT_DIR}/bin/w1w-controller" --license-check

if find "${OUTPUT_DIR}" -type f \( \
    -name '*.py' -o -name '*.pyc' -o -name '*.c' -o -name '*.cc' \
    -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) | grep -q .; then
  echo "sealed runtime unexpectedly contains source files" >&2
  exit 1
fi
if [[ $(head -c 7 "${OUTPUT_DIR}/lib/w1w/controller/policy.onnx") != W1WENC1 ]]; then
  echo "policy encryption check failed" >&2
  exit 1
fi

(
  cd "${OUTPUT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
chmod -R a-w "${OUTPUT_DIR}"
echo "sealed ARM64 runtime built at ${OUTPUT_DIR}"
