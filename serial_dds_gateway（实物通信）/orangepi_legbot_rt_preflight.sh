#!/usr/bin/env bash
# 用途：实机运行前检查环境，确认无串口占用冲突或重复的低层命令发布者。
set -euo pipefail

echo "========== LEGBOT RT preflight =========="
echo "[1] 可能发布 rt/lowcmd 或占用串口的进程："
pgrep -af 'legbot_rt_gait_pd|EX34_legbot|EX35|fatu_ctrl|dds_to_serial_gateway|python3.*real_mpc|python3.*EX34' || true

echo
echo "[2] 串口占用者 (/dev/myttyCAN0 /dev/myttyCAN1 /dev/ttyUSB* /dev/ttyACM*)："
if command -v lsof >/dev/null 2>&1; then
  sudo lsof /dev/myttyCAN0 /dev/myttyCAN1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
else
  echo "lsof 未安装，使用 fuser 回退"
  sudo fuser -v /dev/myttyCAN0 /dev/myttyCAN1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
fi

echo
echo "[3] 非 dry-run 前的期望安全状态："
echo "  - 仅有一个 dds_to_serial_gateway 占用 /dev/myttyCAN0 与 /dev/myttyCAN1"
echo "  - legbot_rt_gait_pd 发布 rt/lowcmd 时，不应有 EX34/EX35/fatu_ctrl 等重复发布者运行"
echo "  - dry-run 不创建 LowCmd 发布者，因此是安全的"
echo "======================================="
