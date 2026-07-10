#!/bin/bash
# 用途：一键清理并构建 serial_dds_gateway 项目。
# 流程：删除旧 build 目录 -> CMake 配置 -> 并行编译 -> 列出关键产物。
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"

echo "========== Fatu DDS Serial Gateway Build =========="

echo "[INFO] 清理旧构建目录..."
rm -rf "$BUILD_DIR"

echo "[INFO] 创建构建目录: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "[INFO] 运行 CMake 配置..."
cd "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B .

echo "[INFO] 开始编译..."
cmake --build . -j$(nproc)

echo "[INFO] 构建完成。"
ls -la "$BUILD_DIR"/*.bin 2>/dev/null || true
ls -la "$BUILD_DIR"/*.exe 2>/dev/null || true
ls -la "$BUILD_DIR"/dds_to_serial_gateway* 2>/dev/null || true