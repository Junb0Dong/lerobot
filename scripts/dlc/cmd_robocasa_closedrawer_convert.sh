#!/usr/bin/env bash
# Convert CloseDrawer v2.1 → v3.0 on NAS (copy first; never mutate /mnt/data/junbo/data).
# Usage:
#   bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_convert.sh
# Local (no DLC):
#   bash scripts/dlc/convert_robocasa_v21_to_v30.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

DISPLAY_NAME="lerobot-ac-${ROBOCASA_TASK,,}-convert-$(date '+%Y%m%d-%H%M%S')"
MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-720}"

USER_COMMAND=$(cat <<EOF
bash -euo pipefail <<INNER
export DEBIAN_FRONTEND=noninteractive
cd ${CODE_DIR}
export CODE_DIR=${CODE_DIR}
export INSTALL_LIBGLX=${INSTALL_LIBGLX:-0}
source scripts/dlc/container_setup.sh
source scripts/dlc/job_runtime.sh
export ROBOCASA_TASK=${ROBOCASA_TASK}
export ROBOCASA_REPO_ID=${ROBOCASA_REPO_ID}
export ROBOCASA_RAW_DATASET=${ROBOCASA_RAW_DATASET}
export ROBOCASA_V3_DATASET=${ROBOCASA_V3_DATASET}
export FORCE_UV_SYNC=${FORCE_UV_SYNC:-0}
export UV_OFFLINE=${UV_OFFLINE:-0}
export SKIP_UV_SYNC=${SKIP_UV_SYNC:-1}
export LEROBOT_VENV=${LEROBOT_VENV:-}
export REQUIRE_CUDA=${REQUIRE_CUDA:-1}
lerobot_dlc_main_convert
INNER
EOF
)
