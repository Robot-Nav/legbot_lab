#!/usr/bin/env bash
set -euo pipefail

# W1W 一键移动到电机编码器零位
# 将选中腿的关节运动到"电机编码器原始值=0"的物理位置（MIT协议电机输出轴零位）。
# 支持选择单条腿或全部腿，其他腿保持阻尼。
# Kp=30, Kd=1.0，速度限制0.2rad/s(≈11°/s)，线性插值平滑运动。
# 到位后自动降为Kp=20/Kd=1保持，实时显示各关节角度偏差（角度制）。
# 轮毂始终阻尼不动。Ctrl+C退出。
#
# ⚠️  警告：大腿/小腿需转约160°，仅在未装腿连杆或拆解状态下使用！
#
# 用法：
#   sudo ./move_to_zero.sh                       # 交互式选择腿
#   sudo ./move_to_zero.sh --leg fl              # 只动左前腿
#   sudo ./move_to_zero.sh --leg all             # 动全部腿
#   sudo ./move_to_zero.sh --no-build            # 跳过编译
#   sudo ./move_to_zero.sh --leg fr --no-build   # 组合使用

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/build/bin"
VENV_DIR="${ROOT_DIR}/venv"
LOG_DIR="/tmp/w1w"
SKIP_BUILD=0
LEG_ARG=""
PY_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) SKIP_BUILD=1; shift ;;
        --leg)
            LEG_ARG="$2"
            if [[ ! "${LEG_ARG}" =~ ^(fl|fr|rl|rr|all)$ ]]; then
                echo "错误: --leg 参数必须是 fl/fr/rl/rr/all"
                exit 1
            fi
            PY_EXTRA_ARGS+=("--leg" "${LEG_ARG}" "--yes")
            shift 2 ;;
        -h|--help)
            echo "用法: sudo $0 [--leg fl|fr|rl|rr|all] [--no-build]"
            echo "  Kp=30, Kd=1.0，移动腿关节到电机编码器零位（MIT输出轴=0），速度11°/s"
            echo "  --leg fl/fr/rl/rr  只动指定腿，其他腿保持阻尼"
            echo "  --leg all          动全部腿（也可在交互界面选）"
            echo "  不指定 --leg 时进入交互选择界面"
            echo "  ⚠️  仅在未装腿连杆时使用！"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${BLUE}==>${NC} $*"; }

LEG_CN="${LEG_ARG:-交互选择}"
case "${LEG_ARG}" in
    fl) LEG_CN="左前腿(FL)" ;;
    fr) LEG_CN="右前腿(FR)" ;;
    rl) LEG_CN="左后腿(RL)" ;;
    rr) LEG_CN="右后腿(RR)" ;;
    all) LEG_CN="全部腿" ;;
esac

echo ""
echo "========================================================================"
echo "  W1W 轮足机器狗 - 移动到电机编码器零位"
echo "  目标腿: ${LEG_CN}"
echo "========================================================================"
echo ""

if [[ ${EUID} -ne 0 ]]; then
    error "请使用 sudo 运行（需要CAN配置权限）"
    exit 1
fi

step "停止现有控制器和服务..."
pkill -f "deploy.py" 2>/dev/null || true
pkill -f "web.server" 2>/dev/null || true
if command -v systemctl > /dev/null 2>&1; then
    for svc in w1w-motor w1w-imu w1w-elrs w1w-controller w1w-web; do
        systemctl stop "${svc}" 2>/dev/null || true
    done
fi
for bin in can_service_wheel imu_service elrs_service; do
    pkill -x "${bin}" 2>/dev/null || true
done
sleep 1
info "清理完成"

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    step "编译C++服务..."
    bash "${ROOT_DIR}/build_dev.sh"
    info "编译完成"
else
    info "跳过编译 (--no-build)"
fi

if [[ ! -x "${BIN_DIR}/can_service_wheel" ]]; then
    error "缺少 ${BIN_DIR}/can_service_wheel，请去掉 --no-build 或先编译"
    exit 1
fi

PYTHON_EXEC="${VENV_DIR}/bin/python"
if [[ ! -x ${PYTHON_EXEC} ]]; then
    step "创建Python虚拟环境..."
    /usr/bin/python3 -m venv "${VENV_DIR}"
    "${PYTHON_EXEC}" -m pip install --upgrade pip
    "${PYTHON_EXEC}" -m pip install numpy==1.26.4 PyYAML==6.0.2
    info "虚拟环境就绪"
else
    info "Python虚拟环境已就绪"
fi

step "配置CAN接口 (can0-can5)..."
bash "${ROOT_DIR}/scripts/setup_can.sh"
info "CAN配置完成"

mkdir -p "${LOG_DIR}"
MOTOR_PID=""

cleanup() {
    echo ""
    echo "========================================================================"
    info "停止电机服务..."
    [[ -n ${MOTOR_PID} ]] && kill ${MOTOR_PID} 2>/dev/null || true
    sleep 0.3
    [[ -n ${MOTOR_PID} ]] && kill -9 ${MOTOR_PID} 2>/dev/null || true
    info "电机已阻尼，可安全断电"
    exit 0
}
trap cleanup INT TERM

step "启动电机服务..."
"${BIN_DIR}/can_service_wheel" \
    --bind-host 127.0.0.1 \
    > "${LOG_DIR}/motor_move.log" 2>&1 &
MOTOR_PID=$!
echo "  PID: ${MOTOR_PID} → ${LOG_DIR}/motor_move.log"

sleep 2
if ! kill -0 ${MOTOR_PID} 2>/dev/null; then
    error "电机服务启动失败!"
    tail -20 "${LOG_DIR}/motor_move.log"
    exit 1
fi
info "电机服务启动成功"

if [[ -z "${LEG_ARG}" ]]; then
    # 交互模式，Python自己做YES确认和腿选择
    step "启动交互模式..."
    echo ""
    set +e
    cd "${ROOT_DIR}/scripts"
    "${PYTHON_EXEC}" -u move_to_zero.py "${PY_EXTRA_ARGS[@]}"
    EXIT_CODE=$?
    set -e
else
    # 指定腿模式：shell做安全确认，Python加--yes跳过确认
    echo ""
    echo "========================================================================"
    echo -e "  ${RED}${BOLD}⚠️  危险操作确认（非常重要）:${NC}"
    echo ""
    echo "  目标腿: ${BOLD}${LEG_CN}${NC}"
    echo "  此脚本将选中腿的大腿/小腿转动约160°到编码器零位！"
    echo -e "  ${RED}若腿连杆已安装，会猛烈撞击机械结构，损坏电机或连杆！${NC}"
    echo ""
    echo "  请确认："
    echo "  1. [ ] 机器人已完全悬挂/腾空，四腿四轮无任何接触"
    echo "  2. [ ] 选中腿连杆未安装（仅电机在支架上），或已完全拆解"
    echo "  3. [ ] CH8急停（SH开关）或物理急停在手边"
    echo "  4. [ ] 随时可断电或拍下急停"
    echo ""
    echo "  Kp=30, Kd=1.0，速度限制0.2rad/s(≈11°/s)，实时显示角度偏差"
    echo "  其他腿将保持阻尼不动"
    echo "========================================================================"
    echo ""

    read -r -p "确认安全，输入 YES 继续: " confirm
    if [[ "${confirm}" != "YES" ]]; then
        info "用户取消"
        cleanup
    fi

    step "移动到电机编码器零位..."
    echo ""
    set +e
    cd "${ROOT_DIR}/scripts"
    "${PYTHON_EXEC}" -u move_to_zero.py "${PY_EXTRA_ARGS[@]}"
    EXIT_CODE=$?
    set -e
fi

if [[ ${EXIT_CODE:-0} -ne 0 && ${EXIT_CODE:-0} -ne 130 ]]; then
    error "工具异常退出 (code=${EXIT_CODE:-0})"
    echo "查看日志: tail -f ${LOG_DIR}/motor_move.log"
fi

echo ""
echo "========================================================================"
echo "  电机已处于编码器零位（MIT输出轴=0）并保持位置"
echo "  按 Ctrl+C 停止服务退出"
echo "========================================================================"
echo ""

while kill -0 ${MOTOR_PID} 2>/dev/null; do
    sleep 1
done
error "电机服务意外退出"
cleanup
