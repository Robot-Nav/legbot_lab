#!/bin/bash
# 用途：自动构建（若不存在）并启动 dds_to_serial_gateway。
# 默认使用双串口：A 口接 FR/RR 电机，B 口接 FL/RL 电机。
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"
GATEWAY_BIN="$BUILD_DIR/dds_to_serial_gateway"

echo "========== Fatu DDS Serial Gateway Launcher =========="

# 若网关二进制不存在，则自动配置并编译。
if [ ! -f "$GATEWAY_BIN" ]; then
    echo "[INFO] 构建网关..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -S "$SCRIPT_DIR" -B .
    cmake --build . -j
    cd "$SCRIPT_DIR"
fi

echo "[INFO] 启动网关..."
"$GATEWAY_BIN" --serial-port-a /dev/myttyCAN0 --serial-port-b /dev/myttyCAN1