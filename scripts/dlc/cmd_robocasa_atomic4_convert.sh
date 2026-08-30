#!/usr/bin/env bash
# Convert four atomic tasks v2.1→v3 and merge (copy first; never mutate /mnt/data/junbo/data).
# Prefer running this locally: bash scripts/dlc/convert_robocasa_atomic4.sh
# Usage:
#   bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_atomic4_convert.sh

STAGE="${STAGE:-day}"
ROBOCASA_TASK="${ROBOCASA_TASK:-atomic4}"
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
ROBOCASA_REPO_ID="${ATOMIC4_REPO_ID}"
ROBOCASA_V3_DATASET="${ATOMIC4_MERGED}"

DISPLAY_NAME="lerobot-ac-atomic4-convert-$(date '+%Y%m%d-%H%M%S')"
MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-720}"

USER_COMMAND=$(cat <<EOF
bash -euo pipefail <<INNER
export DEBIAN_FRONTEND=noninteractive
cd ${CODE_DIR}
export CODE_DIR=${CODE_DIR}
export INSTALL_LIBGLX=${INSTALL_LIBGLX:-0}
source scripts/dlc/container_setup.sh
source scripts/dlc/job_runtime.sh
export ROBOCASA_TASK=atomic4
export ROBOCASA_REPO_ID=${ATOMIC4_REPO_ID}
export ROBOCASA_V3_DATASET=${ATOMIC4_MERGED}
export ATOMIC4_TASKS="${ATOMIC4_TASKS}"
export ATOMIC4_DATE=${ATOMIC4_DATE}
export ATOMIC4_RAW_ROOT=${ATOMIC4_RAW_ROOT}
export ATOMIC4_V3_ROOT=${ATOMIC4_V3_ROOT}
export ATOMIC4_MERGED=${ATOMIC4_MERGED}
export ATOMIC4_REPO_ID=${ATOMIC4_REPO_ID}
export FORCE_UV_SYNC=${FORCE_UV_SYNC:-0}
export UV_OFFLINE=${UV_OFFLINE:-0}
export SKIP_UV_SYNC=${SKIP_UV_SYNC:-1}
export LEROBOT_VENV=${LEROBOT_VENV:-}
export REQUIRE_CUDA=${REQUIRE_CUDA:-1}
lerobot_dlc_main_atomic4_convert
INNER
EOF
)
