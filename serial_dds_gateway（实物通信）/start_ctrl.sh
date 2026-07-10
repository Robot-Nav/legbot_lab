#!/bin/bash
# 用途：在香橙派实机启动 fatu_ctrl 控制器（DDS 仅走本地 lo 网卡）。
# 注意：需先启动 dds_to_serial_gateway，并确保两者使用同一 DDS network。
set -e

echo "[INFO] 启动 fatu_ctrl..."
fatu_ctrl --network lo