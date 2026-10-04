#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "setup_can.sh must run as root" >&2
  exit 1
fi

for interface in can0 can1 can2 can3 can4 can5; do
  if [[ ! -e "/sys/class/net/${interface}" ]]; then
    echo "missing CAN interface: ${interface}" >&2
    exit 1
  fi
done

for interface in can0 can1 can2 can3; do
  ip link set "${interface}" down 2>/dev/null || true
  ip link set "${interface}" type can bitrate 1000000 restart-ms 100 fd off
  ip link set "${interface}" txqueuelen 100
  ip link set "${interface}" up
done

for interface in can4 can5; do
  ip link set "${interface}" down 2>/dev/null || true
  # W190 CAN-FD is stable at a 0.70 data-phase sample point (the PCAN
  # timing quantizer reports this as approximately 0.6875).
  ip link set "${interface}" type can bitrate 1000000 sample-point 0.75 sjw 1 \
    dbitrate 5000000 dsample-point 0.70 dsjw 1 restart-ms 100 fd on
  ip link set "${interface}" txqueuelen 100
  ip link set "${interface}" up
done

for interface in can0 can1 can2 can3 can4 can5; do
  ip -details -statistics link show "${interface}"
done
