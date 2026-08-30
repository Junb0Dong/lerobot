#!/usr/bin/env bash
# Short job: create or repair NAS .venv-dlc only. No tokenizer/policy training.
# After this succeeds, training jobs should reuse the venv and skip uv/PyPI.
# Usage:
#   bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_venv.sh
# Optional CPU-only (if the quota allows GPU=0):
#   RESOURCE_GPU=0 REQUIRE_CUDA=0 bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_venv.sh

# 本作业允许 uv sync 来创建/修复 .venv-dlc。必须在 source paths.sh 之前
# 设好默认 0，否则会被训练作业的 SKIP_UV_SYNC=1 默认值盖住。
: "${SKIP_UV_SYNC:=0}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

# paths.sh 默认 REQUIRE_CUDA=1；本作业只建 venv，允许 GPU=0。
REQUIRE_CUDA="${BOOTSTRAP_REQUIRE_CUDA:-0}"
MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-120}"
DISPLAY_NAME="lerobot-ac-bootstrap-venv-$(date '+%Y%m%d-%H%M%S')"

USER_COMMAND=$(cat <<EOF
bash -euo pipefail <<INNER
export DEBIAN_FRONTEND=noninteractive
cd ${CODE_DIR}
export CODE_DIR=${CODE_DIR}
export INSTALL_LIBGLX=${INSTALL_LIBGLX:-0}
source scripts/dlc/container_setup.sh
source scripts/dlc/job_runtime.sh
export FORCE_UV_SYNC=${FORCE_UV_SYNC:-0}
export UV_OFFLINE=${UV_OFFLINE:-0}
export SKIP_UV_SYNC=${SKIP_UV_SYNC:-0}
export LEROBOT_VENV=${LEROBOT_VENV:-}
export REQUIRE_CUDA=${REQUIRE_CUDA}
lerobot_dlc_main_bootstrap
INNER
EOF
)
