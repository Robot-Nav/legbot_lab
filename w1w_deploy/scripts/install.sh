#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "install.sh must run as root" >&2
  exit 1
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${W1W_PREFIX:-/opt/w1w}"
RUN_USER="${W1W_RUN_USER:-${SUDO_USER:-kickpi}}"
if [[ $# -ne 0 ]]; then
  echo "install.sh does not accept start options" >&2
  echo "install first, inspect the release, then run: sudo w1wctl start-base" >&2
  exit 2
fi

if ! id "${RUN_USER}" >/dev/null 2>&1 || [[ ${RUN_USER} == root ]]; then
  echo "W1W_RUN_USER must name an existing non-root user" >&2
  exit 1
fi
case "${PREFIX}" in
  /opt/*) ;;
  *) echo "W1W_PREFIX must be an absolute path below /opt" >&2; exit 1 ;;
esac

for command in ip g++ /usr/bin/python3; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "missing deployment dependency: ${command}" >&2
    exit 1
  }
done
/usr/bin/python3 -c 'import yaml' >/dev/null 2>&1 || {
  echo "missing deployment dependency: python3-yaml" >&2
  exit 1
}
"${ROOT_DIR}/scripts/build.sh"
/usr/bin/python3 -m compileall -q "${ROOT_DIR}/web" "${ROOT_DIR}/scripts"

VERSION="$(tr -d '[:space:]' < "${ROOT_DIR}/VERSION")"
RELEASE_ID="${VERSION}-$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_DIR="${PREFIX}/releases/${RELEASE_ID}"
CURRENT_LINK="${PREFIX}/current"
PREVIOUS_LINK="${PREFIX}/previous"

if [[ -e ${RELEASE_DIR} ]]; then
  echo "release already exists: ${RELEASE_DIR}" >&2
  exit 1
fi

install -d -m 0755 "${RELEASE_DIR}" "${PREFIX}/releases" /etc/w1w /run/w1w
cp -a "${ROOT_DIR}/VERSION" "${ROOT_DIR}/controller" "${ROOT_DIR}/elrs_service" \
  "${ROOT_DIR}/scripts" \
  "${ROOT_DIR}/tests" "${ROOT_DIR}/web" "${ROOT_DIR}/requirements-onnx.txt" \
  "${RELEASE_DIR}/"
install -d "${RELEASE_DIR}/bin"
install -m 0755 "${ROOT_DIR}/build/bin/can_service_wheel" "${RELEASE_DIR}/bin/"
install -m 0755 "${ROOT_DIR}/build/bin/imu_service" "${RELEASE_DIR}/bin/"
install -m 0755 "${ROOT_DIR}/build/bin/elrs_service" "${RELEASE_DIR}/bin/"
chown -R root:root "${RELEASE_DIR}"

if [[ ! -f /etc/w1w/controller.yaml ]]; then
  install -m 0644 "${ROOT_DIR}/controller/config.yaml" /etc/w1w/controller.yaml
else
  /usr/bin/python3 "${ROOT_DIR}/scripts/migrate_config.py" /etc/w1w/controller.yaml
fi
if [[ ! -f /etc/default/w1w ]]; then
  cat > /etc/default/w1w <<'EOF'
W1W_IMU_DEVICE=/dev/ttyACM0
W1W_IMU_BAUD=921600
W1W_ELRS_DEVICE=/dev/ttyS6
W1W_ELRS_BAUD=420000
W1W_ELRS_PORT=55201
W1W_ELRS_CYCLE_US=1000
W1W_ELRS_TIMEOUT_MS=100
EOF
else
  while IFS='=' read -r name value; do
    if ! grep -q "^${name}=" /etc/default/w1w; then
      printf '%s=%s\n' "${name}" "${value}" >> /etc/default/w1w
    fi
  done <<'EOF'
W1W_ELRS_DEVICE=/dev/ttyS6
W1W_ELRS_BAUD=420000
W1W_ELRS_PORT=55201
W1W_ELRS_CYCLE_US=1000
W1W_ELRS_TIMEOUT_MS=100
EOF
fi

OLD_RELEASE="$(readlink -f "${CURRENT_LINK}" 2>/dev/null || true)"
for unit in w1w-controller.service w1w-web.service w1w-elrs.service w1w-imu.service w1w-motor.service; do
  if systemctl is-active --quiet "${unit}"; then
    echo "refusing to switch releases while ${unit} is running" >&2
    echo "stop the W1W stack explicitly and rerun the installer" >&2
    exit 1
  fi
done
rm -f "${PREFIX}/.current-${RELEASE_ID}"
ln -s "${RELEASE_DIR}" "${PREFIX}/.current-${RELEASE_ID}"
mv -Tf "${PREFIX}/.current-${RELEASE_ID}" "${CURRENT_LINK}"
if [[ -n ${OLD_RELEASE} && -d ${OLD_RELEASE} && ${OLD_RELEASE} != "${RELEASE_DIR}" ]]; then
  ln -sfn "${OLD_RELEASE}" "${PREVIOUS_LINK}"
fi

for template in "${ROOT_DIR}"/systemd/*.service; do
  unit="$(basename "${template}")"
  sed -e "s|@PREFIX@|${PREFIX}|g" -e "s|@RUN_USER@|${RUN_USER}|g" \
    "${template}" > "/etc/systemd/system/${unit}"
done
ln -sfn "${CURRENT_LINK}/scripts/w1wctl" /usr/local/sbin/w1wctl

systemctl daemon-reload
systemctl disable w1w-controller.service 2>/dev/null || true

echo "installed W1W release ${RELEASE_ID} at ${RELEASE_DIR}"
echo "all services remain disabled and stopped"
echo "after inspection, run 'sudo w1wctl start-base' manually"
