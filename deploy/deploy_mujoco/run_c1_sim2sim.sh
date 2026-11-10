#!/usr/bin/env bash
set -euo pipefail

C1_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$C1_REPO_DIR"

# Activate a MuJoCo environment first. Supply local --policy and --training-config.
exec python deploy/deploy_mujoco/deploy_c1.py \
  --terrain all --difficulty 0.5 --stair-height 0.22 --max-cmd 2.0 1.0 2.0 "$@"
