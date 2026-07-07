#!/bin/bash
set -e

echo "========== Serial Port Buffer Check =========="

for port in /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU; do
    if [ -e "$port" ]; then
        echo "--- $port ---"
        echo "Device info:"
        ls -la "$port"
        echo ""
        echo "stty settings:"
        stty -F "$port" -a 2>/dev/null || echo "Cannot read stty settings"
        echo ""
        echo "UDEV info:"
        udevadm info -a -n "$port" 2>/dev/null | grep -E "ID_VENDOR_ID|ID_MODEL_ID|DEVNAME" || echo "Cannot read udev info"
        echo ""
    else
        echo "--- $port ---"
        echo "Device not found"
        echo ""
    fi
done

echo "========== Kernel Buffer Limits =========="
echo "/proc/sys/kernel/pty/max:"
cat /proc/sys/kernel/pty/max 2>/dev/null || echo "N/A"
echo ""

echo "/proc/tty/ldiscs:"
cat /proc/tty/ldiscs 2>/dev/null || echo "N/A"