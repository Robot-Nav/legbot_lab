#!/usr/bin/env bash
set -euo pipefail

# W1W 开发测试模式 - 一键编译启动脚本
# 功能：自动编译C++服务、检查Python环境、配置CAN、启动所有服务
# 用法：sudo ./dev_start.sh [选项]
# 按 Ctrl+C 可一键停止所有服务

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${ROOT_DIR}/build/bin"
VENV_DIR="${ROOT_DIR}/venv"
LOG_DIR="/tmp/w1w"
CONFIG_PATH="${ROOT_DIR}/controller/config.yaml"

# 默认参数
IMU_DEVICE="${W1W_IMU_DEVICE:-/dev/ttyACM0}"
IMU_BAUD="${W1W_IMU_BAUD:-921600}"
ELRS_DEVICE="${W1W_ELRS_DEVICE:-/dev/ttyS6}"
ELRS_BAUD="${W1W_ELRS_BAUD:-420000}"
START_WEB=0
SKIP_BUILD=0
SOFT_GAIN=1

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${BLUE}==>${NC} $*"; }

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --imu)
            IMU_DEVICE="$2"
            shift 2
            ;;
        --imu-baud)
            IMU_BAUD="$2"
            shift 2
            ;;
        --elrs)
            ELRS_DEVICE="$2"
            shift 2
            ;;
        --elrs-baud)
            ELRS_BAUD="$2"
            shift 2
            ;;
        --web)
            START_WEB=1
            shift
            ;;
        --no-build)
            SKIP_BUILD=1
            shift
            ;;
        --hard-gain)
            SOFT_GAIN=0
            shift
            ;;
        --help|-h)
            echo "W1W 轮足机器狗 - 开发测试一键启动脚本"
            echo ""
            echo "用法: sudo $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --imu <device>      IMU串口设备 (默认: /dev/ttyACM0)"
            echo "  --imu-baud <baud>   IMU波特率 (默认: 921600)"
            echo "  --elrs <device>     ELRS串口设备 (默认: /dev/ttyS6)"
            echo "  --elrs-baud <baud>  ELRS波特率 (默认: 420000)"
            echo "  --web               启动Web监控 (端口8080)"
            echo "  --no-build          跳过编译直接启动"
            echo "  --hard-gain         使用硬增益kp=100 (默认软增益kp=65更安全)"
            echo "  --help, -h          显示帮助"
            echo ""
            echo "示例:"
            echo "  sudo $0                                    # 默认软增益，自动编译"
            echo "  sudo $0 --imu /dev/ttyUSB0 --web           # 指定IMU设备并启动web"
            echo "  sudo $0 --no-build --hard-gain             # 跳过编译使用硬增益"
            exit 0
            ;;
        *)
            error "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================================================"
echo "  W1W 轮足机器狗 - 开发测试模式一键启动"
echo "========================================================================"
echo ""

# 检查root权限
if [[ ${EUID} -ne 0 ]]; then
    error "请使用 sudo 运行此脚本（需要CAN和串口权限）"
    exit 1
fi
SUDO_USER_HOME=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)

# ========================================
# 步骤1: 编译C++服务
# ========================================
if [[ ${SKIP_BUILD} -eq 0 ]]; then
    step "编译C++服务（开发模式，跳过授权检查）..."
    bash "${ROOT_DIR}/build_dev.sh"
    info "C++服务编译完成"
    echo ""
else
    info "跳过编译步骤"
fi

# 检查二进制文件
for bin in can_service_wheel imu_service elrs_service; do
    if [[ ! -x "${BIN_DIR}/${bin}" ]]; then
        error "缺少二进制文件: ${BIN_DIR}/${bin}"
        echo "请去掉 --no-build 参数或先手动运行 ./build_dev.sh"
        exit 1
    fi
done

# ========================================
# 步骤2: 检查/创建Python虚拟环境
# ========================================
PYTHON_EXEC="${VENV_DIR}/bin/python"
if [[ ! -x ${PYTHON_EXEC} ]]; then
    step "创建Python虚拟环境并安装依赖..."
    /usr/bin/python3 -m venv "${VENV_DIR}"
    "${PYTHON_EXEC}" -m pip install --upgrade pip
    "${PYTHON_EXEC}" -m pip install numpy==1.26.4 PyYAML==6.0.2 onnxruntime==1.18.1
    info "Python虚拟环境创建完成"
else
    info "Python虚拟环境已就绪"
fi

# ========================================
# 步骤3: 配置CAN接口
# ========================================
step "配置CAN接口 (can0-can5)..."
bash "${ROOT_DIR}/scripts/setup_can.sh"
info "CAN接口配置完成"

# ========================================
# 步骤4: 准备测试配置（软增益模式）
# ========================================
TEST_CONFIG="${ROOT_DIR}/controller/config_test.yaml"
if [[ ${SOFT_GAIN} -eq 1 ]]; then
    info "使用软增益测试模式 (Kp=65, 最大速度受限，更安全)"
    cat > "${TEST_CONFIG}" << EOF
control:
  dt: 0.02

observation:
  num_actions: 16
  num_obs: 57
  history_len: 4
  wheel_sim_indices: [3, 7, 11, 15]

robot:
  joint_names:
    - FL_hip
    - FL_thigh
    - FL_calf
    - FL_wheel
    - FR_hip
    - FR_thigh
    - FR_calf
    - FR_wheel
    - RL_hip
    - RL_thigh
    - RL_calf
    - RL_wheel
    - RR_hip
    - RR_thigh
    - RR_calf
    - RR_wheel
  default_angles: [0.0, -0.68, 1.4, 0.0, 0.0, 0.68, -1.4, 0.0, 0.0, -0.68, 1.4, 0.0, 0.0, 0.68, -1.4, 0.0]
  down_angles: [0.0, -1.2, 2.2, 0.0, 0.0, 1.2, -2.2, 0.0, 0.0, -1.2, 2.2, 0.0, 0.0, 1.2, -2.2, 0.0]
  kp: [100, 100, 100, 0, 130, 130, 130, 0, 100, 100, 100, 0, 120, 120, 120, 0]
  kd: [2.2, 2.2, 2.2, 1.0, 2.2, 2.2, 2.2, 1.0, 2.2, 2.2, 2.2, 1.0, 2.2, 2.2, 2.2, 1.0]

scales:
  ang_vel: 0.25
  cmd: [2.0, 2.0, 0.25]
  dof_err: 1.0
  dof_vel: 0.05
  action: 0.25
  vel_scale: 20

max_cmd: [1.5, 0.8, 1.5]

network:
  motor:
    host: 127.0.0.1
    port: 55100
  imu:
    host: 127.0.0.1
    port: 55200
  elrs:
    host: 127.0.0.1
    port: 55201
    emergency_active_high: true

safety:
  motor_ready_timeout_s: 20.0
  motor_feedback_timeout_ms: 100.0
  controller_send_deadline_ms: 80.0
  active_acquire_timeout_ms: 50.0
  imu_timeout_ms: 100.0
  elrs_timeout_ms: 500.0
  max_temperature_c: 85.0
  rl_entry_blend_s: 0.3

model:
  path: ${ROOT_DIR}/controller/policy.onnx
EOF
    CONFIG_PATH="${TEST_CONFIG}"
else
    info "使用默认硬增益 (Kp=100)"
fi

# ========================================
# 步骤5: 启动所有后台服务
# ========================================
mkdir -p "${LOG_DIR}"

MOTOR_PID=""
IMU_PID=""
ELRS_PID=""
WEB_PID=""

cleanup() {
    echo ""
    echo "========================================================================"
    info "停止所有服务..."
    
    [[ -n ${WEB_PID} ]] && kill ${WEB_PID} 2>/dev/null || true
    [[ -n ${ELRS_PID} ]] && kill ${ELRS_PID} 2>/dev/null || true
    [[ -n ${IMU_PID} ]] && kill ${IMU_PID} 2>/dev/null || true
    [[ -n ${MOTOR_PID} ]] && kill ${MOTOR_PID} 2>/dev/null || true
    
    sleep 0.5
    
    [[ -n ${WEB_PID} ]] && kill -9 ${WEB_PID} 2>/dev/null || true
    [[ -n ${ELRS_PID} ]] && kill -9 ${ELRS_PID} 2>/dev/null || true
    [[ -n ${IMU_PID} ]] && kill -9 ${IMU_PID} 2>/dev/null || true
    [[ -n ${MOTOR_PID} ]] && kill -9 ${MOTOR_PID} 2>/dev/null || true
    
    info "所有服务已停止"
    exit 0
}
trap cleanup INT TERM

step "启动后台服务..."
echo ""

# 电机服务
"${BIN_DIR}/can_service_wheel" \
    --bind-host 127.0.0.1 \
    > "${LOG_DIR}/motor.log" 2>&1 &
MOTOR_PID=$!
echo "  ✅ 电机服务    PID: ${MOTOR_PID} → ${LOG_DIR}/motor.log"

# IMU服务
"${BIN_DIR}/imu_service" \
    --device "${IMU_DEVICE}" \
    --baud "${IMU_BAUD}" \
    --bind-host 127.0.0.1 \
    > "${LOG_DIR}/imu.log" 2>&1 &
IMU_PID=$!
echo "  ✅ IMU服务     PID: ${IMU_PID} → ${LOG_DIR}/imu.log (${IMU_DEVICE} @ ${IMU_BAUD})"

# ELRS服务
"${BIN_DIR}/elrs_service" \
    --device "${ELRS_DEVICE}" \
    --baud "${ELRS_BAUD}" \
    --stats \
    > "${LOG_DIR}/elrs.log" 2>&1 &
ELRS_PID=$!
echo "  ✅ ELRS服务    PID: ${ELRS_PID} → ${LOG_DIR}/elrs.log (${ELRS_DEVICE} @ ${ELRS_BAUD})"

# Web监控（可选）
if [[ ${START_WEB} -eq 1 ]]; then
    cd "${ROOT_DIR}"
    PYTHONPATH="${ROOT_DIR}" "${PYTHON_EXEC}" -m web.server \
        > "${LOG_DIR}/web.log" 2>&1 &
    WEB_PID=$!
    echo "  ✅ Web监控     PID: ${WEB_PID} → ${LOG_DIR}/web.log"
fi

echo ""
info "等待服务初始化 (2秒)..."
sleep 2

# 检查服务是否成功启动
all_ok=1
if ! kill -0 ${MOTOR_PID} 2>/dev/null; then
    error "电机服务启动失败!"
    echo "--- 电机服务日志 ---"
    tail -20 "${LOG_DIR}/motor.log"
    all_ok=0
fi
if ! kill -0 ${IMU_PID} 2>/dev/null; then
    error "IMU服务启动失败! 检查设备 ${IMU_DEVICE} 是否存在"
    echo "--- IMU服务日志 ---"
    tail -20 "${LOG_DIR}/imu.log"
    all_ok=0
fi
if ! kill -0 ${ELRS_PID} 2>/dev/null; then
    error "ELRS服务启动失败! 检查设备 ${ELRS_DEVICE} 是否存在"
    echo "--- ELRS服务日志 ---"
    tail -20 "${LOG_DIR}/elrs.log"
    all_ok=0
fi

if [[ ${all_ok} -eq 0 ]]; then
    error "有服务启动失败，请检查上面的日志"
    cleanup
fi

info "所有后台服务启动成功!"
echo ""

# ========================================
# 显示帮助信息
# ========================================
echo "========================================================================"
echo "  📋 服务日志查看命令 (新开终端执行):"
echo "  tail -f ${LOG_DIR}/motor.log     # 电机服务"
echo "  tail -f ${LOG_DIR}/imu.log       # IMU服务"
echo "  tail -f ${LOG_DIR}/elrs.log      # ELRS遥控器"
if [[ ${START_WEB} -eq 1 ]]; then
IP_ADDR=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -1)
echo "  tail -f ${LOG_DIR}/web.log       # Web监控"
echo ""
echo "  🌐 Web监控地址: http://${IP_ADDR}:8080/"
fi
echo ""
echo "  ⚠️  安全检查:"
echo "  - [ ] 机器人完全架空，四轮四腿都不接触地面"
echo "  - [ ] ELRS CH6 (SB开关) 在最下档（阻尼模式）"
echo "  - [ ] ELRS CH8 (SH开关) 保持在下（急停释放）"
echo "  - [ ] ELRS CH9 (SA开关) 保持在上（速度锁定）"
echo ""
echo "  按 Ctrl+C 停止所有服务"
echo "========================================================================"
echo ""

# ========================================
# 步骤6: 启动Python控制器（前台运行）
# ========================================
step "启动Python强化学习控制器..."
echo ""

cd "${ROOT_DIR}/controller"
exec "${PYTHON_EXEC}" -u deploy.py --config "${CONFIG_PATH}"
