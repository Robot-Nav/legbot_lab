#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${W1W_CONFIG:-/etc/w1w/controller.yaml}"
PYTHON_EXEC="${W1W_PYTHON:-${ROOT_DIR}/venv/bin/python}"
if [[ ! -x ${PYTHON_EXEC} ]]; then
  echo "controller Python is missing: ${PYTHON_EXEC}" >&2
  echo "run ${ROOT_DIR}/scripts/setup_policy_env.sh first" >&2
  exit 1
fi
cd "${ROOT_DIR}/controller"
exec "${PYTHON_EXEC}" -u deploy.py --config "${CONFIG_PATH}"
