#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
RUN_USER="${W1W_RUN_USER:-${SUDO_USER:-kickpi}}"

cd "${ROOT_DIR}"
if [[ ${EUID} -ne 0 ]]; then
  echo "setup_policy_env.sh must run as root" >&2
  exit 1
fi
if ! id "${RUN_USER}" >/dev/null 2>&1 || [[ ${RUN_USER} == root ]]; then
  echo "W1W_RUN_USER must name an existing non-root user" >&2
  exit 1
fi
/usr/bin/python3 -c 'import venv' >/dev/null 2>&1 || {
  echo "python3-venv is missing; install it before setting up the policy environment" >&2
  exit 1
}
/usr/bin/python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements-onnx.txt"
"${VENV_DIR}/bin/python" -m unittest discover -s "${ROOT_DIR}/tests" -v
"${VENV_DIR}/bin/python" -m compileall -q "${ROOT_DIR}/controller"
"${VENV_DIR}/bin/python" -c "
import sys
sys.path.insert(0, '${ROOT_DIR}/controller')
from config import Config
from deploy import Policy
config = Config('${ROOT_DIR}/controller/config.yaml')
policy = Policy(config)
print(f'policy warmup passed: backend={policy.backend}, input={policy.input_buffer.shape}')
"
chown -R root:root "${VENV_DIR}"
chmod -R a+rX "${VENV_DIR}"
echo "policy environment ready: ${VENV_DIR}"
