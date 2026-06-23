#!/bin/bash
# 构建并运行 sim2real 协议测试（C++ 单元测试）和硬件测试（Python）。
#
# 用法：
#   ./scripts/build_and_run_tests.sh              # 构建并运行 C++ 单元测试
#   ./scripts/build_and_run_tests.sh --cpp-only   # 仅 C++ 测试
#   ./scripts/build_and_run_tests.sh --motor      # Python 电机硬件测试
#   ./scripts/build_and_run_tests.sh --imu        # Python IMU 硬件测试

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_DIR/deploy/robots/vbot/tests"
BUILD_DIR="$TEST_DIR/build"

run_cpp_tests() {
    echo "============================================"
    echo "正在构建 C++ 协议单元测试"
    echo "============================================"

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j"$(nproc)"

    echo ""
    echo "============================================"
    echo "正在运行: test_motor_protocol"
    echo "============================================"
    ./test_motor_protocol

    echo ""
    echo "============================================"
    echo "正在运行: test_imu_protocol"
    echo "============================================"
    ./test_imu_protocol

    echo ""
    echo "所有 C++ 测试通过。"
}

run_motor_hw_test() {
    echo "正在运行电机硬件测试..."
    python3 "$SCRIPT_DIR/test_motor.py" "$@"
}

run_imu_hw_test() {
    echo "正在运行 IMU 硬件测试..."
    python3 "$SCRIPT_DIR/test_imu.py" "$@"
}

# ─── 参数解析 ──────────────────────────────────────────────────────────────

if [ $# -eq 0 ]; then
    run_cpp_tests
    exit 0
fi

case "${1:-}" in
    --cpp-only)
        run_cpp_tests
        ;;
    --motor)
        shift
        run_motor_hw_test "$@"
        ;;
    --imu)
        shift
        run_imu_hw_test "$@"
        ;;
    --all)
        run_cpp_tests
        echo ""
        echo "硬件测试需要在机器人主机上运行，此处跳过。"
        echo "请在机器人上手动执行："
        echo "  python3 scripts/test_motor.py --port /dev/ttyUSB0"
        echo "  python3 scripts/test_imu.py --port /dev/ttyUSB2"
        ;;
    *)
        echo "用法: $0 [--cpp-only|--motor|--imu|--all]"
        exit 1
        ;;
esac
