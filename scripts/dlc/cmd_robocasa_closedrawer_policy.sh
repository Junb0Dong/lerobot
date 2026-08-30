#!/usr/bin/env bash
# Semantic policy-only CloseDrawer job. Requires TOKENIZER_PATH.
# Usage:
#   TOKENIZER_PATH=/path/to/tokenizer bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_policy.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

if [[ ! -f "${TOKENIZER_PATH}/model.safetensors" ]]; then
  echo "[error] set TOKENIZER_PATH to a trained tokenizer directory (missing model.safetensors)" >&2
  echo "        example: TOKENIZER_PATH=/mnt/data/junbo/lerobot/outputs/dlc/<run>/tokenizer \\" >&2
  echo "                 bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_policy.sh" >&2
  return 1
fi

DISPLAY_NAME="lerobot-ac-${ROBOCASA_TASK,,}-pol-${STAGE}-$(date '+%Y%m%d-%H%M%S')"

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
export RUN_ROOT=${RUN_ROOT}
export TOKENIZER_PATH=${TOKENIZER_PATH}
export POLICY_OUTPUT_DIR=${RUN_ROOT}/policy
export POLICY_STEPS=${POLICY_STEPS}
export BATCH_SIZE=${BATCH_SIZE}
export NUM_WORKERS=${NUM_WORKERS}
export WANDB_ENABLE=${WANDB_ENABLE}
export WANDB_PROJECT=${WANDB_PROJECT}
export STAGE=${STAGE}
export FORCE_UV_SYNC=${FORCE_UV_SYNC:-0}
export UV_OFFLINE=${UV_OFFLINE:-0}
export SKIP_UV_SYNC=${SKIP_UV_SYNC:-1}
export LEROBOT_VENV=${LEROBOT_VENV:-}
export REQUIRE_CUDA=${REQUIRE_CUDA:-1}
lerobot_dlc_main_policy
INNER
EOF
)
