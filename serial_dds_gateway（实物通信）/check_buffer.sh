#!/bin/bash
# 用途：检查电机与 IMU 串口设备状态、stty 配置、udev 信息及内核缓冲区上限。
set -e

echo "========== Serial Port Buffer Check =========="

# 遍历三个固定串口别名，输出设备是否存在及其关键属性。
for port in /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU; do
    if [ -e "$port" ]; then
        echo "--- $port ---"
        echo "设备信息："
        ls -la "$port"
        echo ""
        echo "stty 配置："
        stty -F "$port" -a 2>/dev/null || echo "无法读取 stty 配置"
        echo ""
        echo "UDEV 信息："
        udevadm info -a -n "$port" 2>/dev/null | grep -E "ID_VENDOR_ID|ID_MODEL_ID|DEVNAME" || echo "无法读取 udev 信息"
        echo ""
    else
        echo "--- $port ---"
        echo "设备未找到"
        echo ""
    fi
done

echo "========== Kernel Buffer Limits =========="
echo "/proc/sys/kernel/pty/max："
cat /proc/sys/kernel/pty/max 2>/dev/null || echo "N/A"
echo ""

echo "/proc/tty/ldiscs："
cat /proc/tty/ldiscs 2>/dev/null || echo "N/A"