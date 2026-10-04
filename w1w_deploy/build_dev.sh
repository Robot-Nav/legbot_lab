#!/usr/bin/env bash
# W1W 开发模式一键编译脚本（基于官方scripts/build.sh修改）
# 用途：在开发板上编译所有C++服务，跳过设备授权检查
# 用法：./build_dev.sh
set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== W1W 开发模式编译脚本开始 ===${NC}"
echo ""

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# 检查g++
command -v g++ >/dev/null || { echo -e "${RED}错误: 未找到g++${NC}" >&2; exit 1; }
echo -e "${YELLOW}[1/5] 创建输出目录${NC}"
install -d "${BUILD_DIR}/bin"

# ========================================
# 生成开发模式的device_guard实现（跳过授权，无需OpenSSL）
# ========================================
echo -e "${YELLOW}[2/5] 生成开发模式device_guard（跳过设备授权检查）${NC}"
DEV_GUARD_DIR=$(mktemp -d)
cat > "${DEV_GUARD_DIR}/device_guard.h" << 'EOF'
#pragma once
#include <string>
namespace w1w { namespace device_guard { inline bool enforce(const std::string&) { return true; } } }
EOF

cat > "${DEV_GUARD_DIR}/device_guard.cpp" << 'EOF'
#include "device_guard.h"
// 开发模式：空实现，enforce()直接返回true
EOF

GUARD_FLAGS=(
  -I"${DEV_GUARD_DIR}"
  "${DEV_GUARD_DIR}/device_guard.cpp"
)

# ========================================
# 编译 can_service_wheel（电机服务）
# ========================================
echo -e "${YELLOW}[3/5] 编译 can_service_wheel（电机CAN服务）${NC}"
g++ -std=c++17 -O2 -DNDEBUG -pthread \
  "${GUARD_FLAGS[@]}" \
  -I"${ROOT_DIR}/motor_service/include" \
  "${ROOT_DIR}/motor_service/src/motor_config.cpp" \
  "${ROOT_DIR}/motor_service/src/robstride.cpp" \
  "${ROOT_DIR}/motor_service/src/socketcan_interface.cpp" \
  "${ROOT_DIR}/motor_service/src/w190_canfd.cpp" \
  "${ROOT_DIR}/motor_service/src/can_service_wheel.cpp" \
  -o "${BUILD_DIR}/bin/can_service_wheel"
echo "  ✅ can_service_wheel"

# ========================================
# 编译 imu_service（IMU服务）
# ========================================
echo -e "${YELLOW}[4/5] 编译 imu_service（IMU数据服务）${NC}"
g++ -std=c++17 -O2 -DNDEBUG -pthread \
  "${GUARD_FLAGS[@]}" \
  -I"${ROOT_DIR}/imu_service/include" \
  "${ROOT_DIR}/imu_service/src/bsp_crc.cpp" \
  "${ROOT_DIR}/imu_service/src/imu_driver.cpp" \
  "${ROOT_DIR}/imu_service/src/imu_service.cpp" \
  -o "${BUILD_DIR}/bin/imu_service"
echo "  ✅ imu_service"

# ========================================
# 编译 elrs_service（遥控器服务）
# ========================================
echo -e "${YELLOW}[5/5] 编译 elrs_service（ELRS遥控器CRSF服务）${NC}"
g++ -std=c++17 -O2 -DNDEBUG -pthread \
  "${GUARD_FLAGS[@]}" \
  -I"${ROOT_DIR}/elrs_service/include" \
  "${ROOT_DIR}/elrs_service/src/elrs_service.cpp" \
  "${ROOT_DIR}/elrs_service/src/elrs_driver.cpp" \
  -o "${BUILD_DIR}/bin/elrs_service"
echo "  ✅ elrs_service"

# 清理临时文件
rm -rf "${DEV_GUARD_DIR}"

# 设置权限
chmod +x "${BUILD_DIR}/bin/"*

echo ""
echo -e "${GREEN}=== 编译完成！产物列表： ===${NC}"
ls -lh "${BUILD_DIR}/bin/"
echo ""
file "${BUILD_DIR}/bin/"*
echo ""

# 验证ARM64格式
echo -e "${GREEN}=== 验证编译结果 ===${NC}"
ALL_OK=1
for bin in "${BUILD_DIR}/bin/"*; do
  if file "$bin" | grep -q "ARM aarch64.*executable"; then
    echo "  ✅ $(basename "$bin"): 有效ARM64可执行文件"
  elif file "$bin" | grep -q -E "(ELF 64-bit|executable)"; then
    echo "  ⚠️  $(basename "$bin"): 可执行文件（非ARM64？）"
  else
    echo "  ❌ $(basename "$bin"): 异常！"
    ALL_OK=0
  fi
done

if [ "$ALL_OK" -eq 1 ]; then
  echo ""
  echo -e "${GREEN}============================================${NC}"
  echo -e "${GREEN}✅ 所有服务编译成功！${NC}"
  echo -e "${GREEN}============================================${NC}"
  echo ""
  echo "⚠️  启动前确保机器人完全架空！"
  echo ""
  echo "启动命令："
  echo "  # 终端1: 电机服务"
  echo "  cd $(basename "$ROOT_DIR") && sudo ./build/bin/can_service_wheel"
  echo ""
  echo "  # 终端2: IMU服务（根据实际串口修改）"
  echo "  cd $(basename "$ROOT_DIR") && sudo ./build/bin/imu_service /dev/ttyACM0 921600"
  echo ""
  echo "  # 终端3: ELRS遥控器服务（根据实际串口修改）"
  echo "  cd $(basename "$ROOT_DIR") && sudo ./build/bin/elrs_service /dev/ttyS6 420000"
  echo ""
  echo "  # 终端4: Python控制器"
  echo "  cd $(basename "$ROOT_DIR") && source venv/bin/activate && cd controller && python deploy.py"
  echo ""
else
  echo -e "${RED}❌ 部分文件编译异常，请检查错误信息${NC}"
  exit 1
fi
