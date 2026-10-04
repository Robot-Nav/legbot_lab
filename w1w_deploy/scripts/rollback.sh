#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "rollback.sh must run as root" >&2
  exit 1
fi

PREFIX="${W1W_PREFIX:-/opt/w1w}"
CURRENT="$(readlink -f "${PREFIX}/current" 2>/dev/null || true)"
PREVIOUS="$(readlink -f "${PREFIX}/previous" 2>/dev/null || true)"
if [[ -z ${PREVIOUS} || ! -d ${PREVIOUS} ]]; then
  echo "no previous W1W release is available" >&2
  exit 1
fi

systemctl stop w1w-controller.service
ln -s "${PREVIOUS}" "${PREFIX}/.rollback-current"
mv -Tf "${PREFIX}/.rollback-current" "${PREFIX}/current"
ln -sfn "${CURRENT}" "${PREFIX}/previous"
systemctl daemon-reload
echo "rolled back to ${PREVIOUS}"
echo "base services were not restarted; use w1wctl base-restart with the robot supported"

