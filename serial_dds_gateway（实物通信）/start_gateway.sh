#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"
GATEWAY_BIN="$BUILD_DIR/dds_to_serial_gateway"

echo "========== Fatu DDS Serial Gateway Launcher =========="

if [ ! -f "$GATEWAY_BIN" ]; then
    echo "[INFO] Building gateway..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -S "$SCRIPT_DIR" -B .
    cmake --build . -j
    cd "$SCRIPT_DIR"
fi

echo "[INFO] Starting gateway..."
"$GATEWAY_BIN" --serial-port-a /dev/myttyCAN0 --serial-port-b /dev/myttyCAN1