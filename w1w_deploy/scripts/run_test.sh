#!/usr/bin/env bash
set -euo pipefail

# W1W 测试模式一键启动脚本
# 特点：
# - 使用更软的控制增益（更安全）
# - 实时打印服务日志
# - 自动检查所有依赖
# - 可选择启动Web监控
# - Ctrl+C自动清理所有进程

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/controller/config.yaml"
PYTHON_EXEC="${ROOT_DIR}/venv/bin/python"
BIN_DIR="${ROOT_DIR}/build/bin"
LOG_DIR="/tmp/w1w"
TEST_CONFIG="${ROOT_DIR}/controller/config_test.yaml"

# 默认参数
IMU_DEVICE="${W1W_IMU_DEVICE:-/dev/ttyACM0}"
IMU_BAUD="${W1W_IMU_BAUD:-921600}"
ELRS_DEVICE="${W1W_ELRS_DEVICE:-/dev/ttyS6}"
ELRS_BAUD="${W1W_ELRS_BAUD:-420000}"
START_WEB=0
SOFT_GAIN=1

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
        --hard-gain)
            SOFT_GAIN=0
            shift
            ;;
        --help)
            echo "W1W 测试模式启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --imu <device>      IMU串口设备 (默认: /dev/ttyACM0)"
            echo "  --imu-baud <baud>   IMU波特率 (默认: 921600)"
            echo "  --elrs <device>     ELRS串口设备 (默认: /dev/ttyS6)"
            echo "  --elrs-baud <baud>  ELRS波特率 (默认: 420000)"
            echo "  --web               启动Web监控 (端口8000)"
            echo "  --hard-gain         使用硬增益(默认测试用软增益)"
            echo "  --help              显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                                    # 默认软增益测试模式"
            echo "  $0 --imu /dev/ttyUSB0 --web            # IMU在ttyUSB0并启动web"
            echo "  $0 --elrs /dev/ttyS5 --hard-gain       # ELRS在ttyS5使用硬增益"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "W1W 轮足机器狗 - 测试调试模式启动"
echo "========================================================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# 检查root权限
if [[ $EUID -ne 0 ]]; then
    warn "建议使用sudo运行以获得CAN和串口权限"
    echo "继续执行，可能会有权限问题..."
    echo ""
    SUDO="sudo"
else
    SUDO=""
fi

# 检查虚拟环境
if [[ ! -x ${PYTHON_EXEC} ]]; then
    error "Python虚拟环境不存在: ${PYTHON_EXEC}"
    echo "请先运行: bash scripts/setup_policy_env.sh"
    exit 1
fi
info "Python虚拟环境检查通过"

# 检查二进制文件
for bin in can_service_wheel imu_service elrs_service; do
    if [[ ! -x "${BIN_DIR}/${bin}" ]]; then
        error "缺少二进制文件: ${BIN_DIR}/${bin}"
        echo "请先运行: ./build_dev.sh 编译"
        exit 1
    fi
done
info "C++二进制文件检查通过"

# 配置CAN
info "配置CAN接口..."
$SUDO bash "${ROOT_DIR}/scripts/setup_can.sh"
info "CAN接口配置完成"

# 如果是软增益模式，创建测试配置
if [[ ${SOFT_GAIN} -eq 1 ]]; then
    info "使用软增益测试模式 (kp=65, 更安全)"
    # 生成测试配置文件
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
  default_angles: [0.0, -0.7, 1.5, 0.0, 0.0, 0.7, -1.5, 0.0, 0.0, -0.7, 1.5, 0.0, 0.0, 0.7, -1.5, 0.0]
  down_angles: [0.0, -1.57, 2.88, 0.0, 0.0, 1.57, -2.88, 0.0, 0.0, -1.57, 2.88, 0.0, 0.0, 1.57, -2.88, 0.0]
  kp: [65, 65, 65, 0, 65, 65, 65, 0, 65, 65, 65, 0, 65, 65, 65, 0]
  kd: [2, 2, 2, 1.0, 2, 2, 2, 1.0, 2, 2, 2, 1.0, 2, 2, 2, 1.0]

scales:
  ang_vel: 0.25
  cmd: [2.0, 2.0, 0.25]
  dof_err: 1.0
  dof_vel: 0.05
  action: 0.25
  vel_scale: 20

max_cmd: [1.0, 0.5, 1.0]

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
  path: policy.onnx
EOF
    CONFIG_PATH="${TEST_CONFIG}"
else
    info "使用默认硬增益 (kp=100)"
fi

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 进程PID数组
MOTOR_PID=""
IMU_PID=""
ELRS_PID=""
WEB_PID=""

# 清理函数
cleanup() {
    echo ""
    echo "========================================================================"
    info "停止所有服务..."
    
    [[ -n ${WEB_PID} ]] && kill ${WEB_PID} 2>/dev/null || true
    [[ -n ${ELRS_PID} ]] && $SUDO kill ${ELRS_PID} 2>/dev/null || true
    [[ -n ${IMU_PID} ]] && $SUDO kill ${IMU_PID} 2>/dev/null || true
    [[ -n ${MOTOR_PID} ]] && $SUDO kill ${MOTOR_PID} 2>/dev/null || true
    
    sleep 0.5
    
    [[ -n ${WEB_PID} ]] && kill -9 ${WEB_PID} 2>/dev/null || true
    [[ -n ${ELRS_PID} ]] && $SUDO kill -9 ${ELRS_PID} 2>/dev/null || true
    [[ -n ${IMU_PID} ]] && $SUDO kill -9 ${IMU_PID} 2>/dev/null || true
    [[ -n ${MOTOR_PID} ]] && $SUDO kill -9 ${MOTOR_PID} 2>/dev/null || true
    
    info "所有服务已停止"
    exit 0
}
trap cleanup INT TERM

echo ""
info "启动电机服务..."
$SUDO "${BIN_DIR}/can_service_wheel" > "${LOG_DIR}/motor.log" 2>&1 &
MOTOR_PID=$!
echo "  PID: ${MOTOR_PID} → ${LOG_DIR}/motor.log"

info "启动IMU服务 (${IMU_DEVICE} @ ${IMU_BAUD})..."
$SUDO "${BIN_DIR}/imu_service" --device "${IMU_DEVICE}" --baud "${IMU_BAUD}" > "${LOG_DIR}/imu.log" 2>&1 &
IMU_PID=$!
echo "  PID: ${IMU_PID} → ${LOG_DIR}/imu.log"

info "启动ELRS服务 (${ELRS_DEVICE} @ ${ELRS_BAUD})..."
$SUDO "${BIN_DIR}/elrs_service" --device "${ELRS_DEVICE}" --baud "${ELRS_BAUD}" --stats > "${LOG_DIR}/elrs.log" 2>&1 &
ELRS_PID=$!
echo "  PID: ${ELRS_PID} → ${LOG_DIR}/elrs.log"

if [[ ${START_WEB} -eq 1 ]]; then
    info "启动Web监控服务..."
    cd "${ROOT_DIR}"
    PYTHONPATH="${ROOT_DIR}" "${PYTHON_EXEC}" -m web.server > "${LOG_DIR}/web.log" 2>&1 &
    WEB_PID=$!
    echo "  PID: ${WEB_PID} → ${LOG_DIR}/web.log"
fi

echo ""
info "等待服务初始化 (2秒)..."
sleep 2

# 检查服务是否成功启动
all_ok=1
if ! kill -0 ${MOTOR_PID} 2>/dev/null; then
    error "电机服务启动失败!"
    echo "日志:"
    tail -20 "${LOG_DIR}/motor.log"
    all_ok=0
fi
if ! kill -0 ${IMU_PID} 2>/dev/null; then
    error "IMU服务启动失败! 检查设备 ${IMU_DEVICE} 是否存在"
    echo "日志:"
    tail -20 "${LOG_DIR}/imu.log"
    all_ok=0
fi
if ! kill -0 ${ELRS_PID} 2>/dev/null; then
    error "ELRS服务启动失败! 检查设备 ${ELRS_DEVICE} 是否存在"
    echo "日志:"
    tail -20 "${LOG_DIR}/elrs.log"
    all_ok=0
fi

if [[ ${all_ok} -eq 0 ]]; then
    error "有服务启动失败，请检查日志"
    cleanup
fi

info "所有服务启动成功!"
echo ""
echo "========================================================================"
echo "📊 实时日志命令 (新开终端执行):"
echo "  tail -f ${LOG_DIR}/motor.log   # 电机服务日志"
echo "  tail -f ${LOG_DIR}/imu.log     # IMU服务日志"
echo "  tail -f ${LOG_DIR}/elrs.log    # ELRS服务日志"
if [[ ${START_WEB} -eq 1 ]]; then
echo "  tail -f ${LOG_DIR}/web.log     # Web服务日志"
IP_ADDR=$($SUDO ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -1)
echo ""
echo "🌐 Web监控地址: http://${IP_ADDR}:8000"
fi
echo ""
echo "⚠️  安全提示:"
echo "  - 机器人必须完全架空!"
echo "  - SB开关(CH6)拨到最上档(阻尼)启动"
echo "  - SH开关(CH8)保持在下(急停释放)"
echo "  - 按Ctrl+C停止所有服务"
echo "========================================================================"
echo ""

# 启动Python控制器
info "启动Python强化学习控制器..."
cd "${ROOT_DIR}/controller"
exec "${PYTHON_EXEC}" -u deploy.py --config "${CONFIG_PATH}"

