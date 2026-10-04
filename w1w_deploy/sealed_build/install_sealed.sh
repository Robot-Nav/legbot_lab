#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 || $# -ne 3 ]]; then
  echo "usage: sudo install_sealed.sh SOURCE_DIR RELEASE_ID CONFIG_SOURCE" >&2
  exit 2
fi

SOURCE_DIR="$(readlink -f "$1")"
RELEASE_ID="$2"
CONFIG_SOURCE="$(readlink -f "$3")"
PREFIX=/opt/w1w
RELEASE_DIR="${PREFIX}/releases/${RELEASE_ID}"
CURRENT_LINK="${PREFIX}/current"
UNIT_DIR=/etc/systemd/system
BACKUP_DIR="/run/w1w-sealed-unit-backup-$$"
UNITS=(w1w-can.service w1w-motor.service w1w-imu.service w1w-elrs.service w1w-controller.service)
OLD_UNITS=(w1w-can.service w1w-motor.service w1w-imu.service elrs.service w1w-controller.service)

case "${SOURCE_DIR}" in
  /home/kickpi/*/build/sealed-runtime) ;;
  *) echo "unexpected sealed source path: ${SOURCE_DIR}" >&2; exit 1 ;;
esac
[[ ${RELEASE_ID} =~ ^[0-9A-Za-z._-]+$ ]] || {
  echo "invalid release id" >&2
  exit 1
}
[[ -f ${SOURCE_DIR}/SHA256SUMS && -f ${CONFIG_SOURCE} ]] || {
  echo "sealed runtime or controller configuration is missing" >&2
  exit 1
}
[[ ! -e ${RELEASE_DIR} ]] || {
  echo "release already exists: ${RELEASE_DIR}" >&2
  exit 1
}

install -d -m 0755 "${PREFIX}/releases" /etc/w1w "${BACKUP_DIR}"
cp -a "${SOURCE_DIR}" "${RELEASE_DIR}"
chown -R root:root "${RELEASE_DIR}"
chmod -R go-w "${RELEASE_DIR}"
(
  cd "${RELEASE_DIR}"
  sha256sum -c SHA256SUMS >/dev/null
)
install -m 0644 "${CONFIG_SOURCE}" /etc/w1w/controller.yaml

for unit in "${UNITS[@]}" elrs.service; do
  if [[ -f ${UNIT_DIR}/${unit} ]]; then
    cp -a "${UNIT_DIR}/${unit}" "${BACKUP_DIR}/${unit}"
  fi
done
previous_current="$(readlink "${CURRENT_LINK}" 2>/dev/null || true)"
switched=0

rollback() {
  local status=$?
  trap - ERR
  if [[ ${switched} -eq 1 ]]; then
    systemctl stop w1w-controller.service w1w-motor.service \
      w1w-imu.service w1w-elrs.service 2>/dev/null || true
    for unit in "${UNITS[@]}" elrs.service; do
      if [[ -f ${BACKUP_DIR}/${unit} ]]; then
        cp -a "${BACKUP_DIR}/${unit}" "${UNIT_DIR}/${unit}"
      else
        rm -f "${UNIT_DIR}/${unit}"
      fi
    done
    if [[ -n ${previous_current} ]]; then
      ln -sfn "${previous_current}" "${CURRENT_LINK}"
    else
      rm -f "${CURRENT_LINK}"
    fi
    systemctl daemon-reload
    systemctl enable "${OLD_UNITS[@]}" >/dev/null 2>&1 || true
    systemctl start "${OLD_UNITS[@]}" 2>/dev/null || true
  fi
  echo "sealed switch failed; old services restored" >&2
  exit "${status}"
}
trap rollback ERR

systemctl stop w1w-controller.service
sleep 0.3
systemctl stop w1w-web.service w1w-motor.service w1w-imu.service \
  w1w-elrs.service elrs.service w1w-can.service 2>/dev/null || true

for unit in "${UNITS[@]}"; do
  install -m 0644 "${RELEASE_DIR}/systemd/${unit}" "${UNIT_DIR}/${unit}"
done
ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}"
switched=1
systemctl daemon-reload
systemd-analyze verify "${UNIT_DIR}"/w1w-*.service
systemctl disable elrs.service >/dev/null 2>&1 || true
systemctl enable w1w-can.service w1w-motor.service w1w-imu.service \
  w1w-elrs.service w1w-controller.service >/dev/null
rm -f "${UNIT_DIR}/w1w-web.service"

systemctl start w1w-can.service
systemctl start w1w-motor.service w1w-imu.service w1w-elrs.service
sleep 4
systemctl is-active --quiet w1w-motor.service
systemctl is-active --quiet w1w-imu.service
systemctl is-active --quiet w1w-elrs.service
systemctl start w1w-controller.service
sleep 5
systemctl is-active --quiet w1w-controller.service

trap - ERR
rm -rf -- "${BACKUP_DIR}"
echo "sealed release active: ${RELEASE_DIR}"
