#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"

echo "========== Fatu DDS Serial Gateway Build =========="

echo "[INFO] Cleaning old build files..."
rm -rf "$BUILD_DIR"

echo "[INFO] Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "[INFO] Running CMake..."
cd "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B .

echo "[INFO] Building..."
cmake --build . -j$(nproc)

echo "[INFO] Build completed successfully!"
ls -la "$BUILD_DIR"/*.bin 2>/dev/null || true
ls -la "$BUILD_DIR"/*.exe 2>/dev/null || true
ls -la "$BUILD_DIR"/dds_to_serial_gateway* 2>/dev/null || true