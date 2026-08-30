#!/usr/bin/env bash
# One-time DLC bootstrap for the Python 3.12 + Torch 2.7 RoboCasa eval venv.

: "${SKIP_UV_SYNC:=1}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-240}"
INSTALL_LIBGLX=1
DISPLAY_NAME="lerobot-robocasa-eval-bootstrap-$(date '+%Y%m%d-%H%M%S')"

USER_COMMAND=$(cat <<EOF
bash -euo pipefail <<INNER
export DEBIAN_FRONTEND=noninteractive
cd ${CODE_DIR}
export CODE_DIR=${CODE_DIR}
export INSTALL_LIBGLX=${INSTALL_LIBGLX}
source scripts/dlc/container_setup.sh
source scripts/dlc/job_runtime.sh
export SKIP_UV_SYNC=${SKIP_UV_SYNC}
export REQUIRE_CUDA=1
lerobot_dlc_bootstrap
lerobot_dlc_ensure_uv
export PATH="${CODE_DIR}/.cache/uv-dlc/bin:\${PATH}"
export ROBOCASA_EVAL_VENV=${ROBOCASA_EVAL_VENV:-${CODE_DIR}/.venv-robocasa-dlc}
export FORCE_ROBOCASA_EVAL_BOOTSTRAP=${FORCE_ROBOCASA_EVAL_BOOTSTRAP:-0}
export RESUME_ROBOCASA_EVAL_BOOTSTRAP=${RESUME_ROBOCASA_EVAL_BOOTSTRAP:-0}
bash scripts/dlc/bootstrap_robocasa_eval.sh
INNER
EOF
)
