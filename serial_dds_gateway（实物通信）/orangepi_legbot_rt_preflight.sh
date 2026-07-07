#!/usr/bin/env bash
set -euo pipefail

echo "========== LEGBOT RT preflight =========="
echo "[1] Processes that may publish rt/lowcmd or own serial ports:"
pgrep -af 'legbot_rt_gait_pd|EX34_legbot|EX35|fatu_ctrl|dds_to_serial_gateway|python3.*real_mpc|python3.*EX34' || true

echo
echo "[2] Serial port owners (/dev/myttyCAN0 /dev/myttyCAN1 /dev/ttyUSB* /dev/ttyACM*):"
if command -v lsof >/dev/null 2>&1; then
  sudo lsof /dev/myttyCAN0 /dev/myttyCAN1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
else
  echo "lsof not installed; using fuser fallback"
  sudo fuser -v /dev/myttyCAN0 /dev/myttyCAN1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
fi

echo
echo "[3] Expected safe state before non-dry-run:"
echo "  - exactly one dds_to_serial_gateway should own /dev/myttyCAN0 and /dev/myttyCAN1"
echo "  - no Python EX34/EX35/fatu_ctrl publisher should be running when legbot_rt_gait_pd publishes rt/lowcmd"
echo "  - dry-run is safe because it does not create a LowCmd publisher"
echo "======================================="
