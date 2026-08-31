#!/usr/bin/env bash
# Submit-time path defaults. Sourced by cmd_*.sh on the submit machine.
# Override any of these before `bash scripts/dlc/submit.sh ...`.

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 提交机上 /mnt/workspace 与 /mnt/data 是同一份 NAS 的两个挂载点；`pwd` 常解析成
# /mnt/workspace。DLC 只把 NAS 挂到 /mnt/data/，容器里没有 /mnt/workspace。
if [[ "${CODE_DIR}" == /mnt/workspace/* ]]; then
  CODE_DIR="/mnt/data/${CODE_DIR#/mnt/workspace/}"
fi
# LEROBOT_VENV 同样可能从提交机 pwd 带上 /mnt/workspace。
if [[ -n "${LEROBOT_VENV:-}" && "${LEROBOT_VENV}" == /mnt/workspace/* ]]; then
  LEROBOT_VENV="/mnt/data/${LEROBOT_VENV#/mnt/workspace/}"
fi
# 容器 bootstrap（job_runtime.sh）读取这些开关。默认复用 .venv-dlc：
# import lerobot + torch 2.7.x + torch.from_numpy 成功则零网络，不装 uv / 不 sync。
# 训练作业默认 SKIP_UV_SYNC=1：探测失败即失败，GPU 上不 curl uv、不访问 PyPI。
# 建环境用 cmd_bootstrap_venv.sh（该脚本把 SKIP_UV_SYNC 默认成 0）。
# FORCE_UV_SYNC=1 只给 bootstrap 用（训练作业会忽略）。UV_OFFLINE=1 只走本地缓存。
# INSTALL_LIBGLX=1 才 apt 装 libglx0。
FORCE_UV_SYNC="${FORCE_UV_SYNC:-0}"
UV_OFFLINE="${UV_OFFLINE:-0}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
INSTALL_LIBGLX="${INSTALL_LIBGLX:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
ROBOCASA_TASK="${ROBOCASA_TASK:-CloseDrawer}"
ROBOCASA_DATE="${ROBOCASA_DATE:-20250819}"
ROBOCASA_RAW_DATASET="${ROBOCASA_RAW_DATASET:-/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic/${ROBOCASA_TASK}/${ROBOCASA_DATE}/lerobot}"
ROBOCASA_REPO_ID="${ROBOCASA_REPO_ID:-robocasa/${ROBOCASA_TASK}}"
# 转换产物写到本仓库 outputs/，绝不原地改 /mnt/data/junbo/data。
ROBOCASA_V3_DATASET="${ROBOCASA_V3_DATASET:-${CODE_DIR}/outputs/datasets/robocasa_${ROBOCASA_TASK,,}_v3}"

# Four-task robocasa atomic merge (CloseDrawer + StartCoffeeMachine +
# TurnOffMicrowave + TurnOffSinkFaucet). CloseDrawer single-task scripts
# do not use these; do not override ROBOCASA_V3_DATASET for them.
ATOMIC4_TASKS="${ATOMIC4_TASKS:-CloseDrawer StartCoffeeMachine TurnOffMicrowave TurnOffSinkFaucet}"
ATOMIC4_DATE="${ATOMIC4_DATE:-${ROBOCASA_DATE}}"
ATOMIC4_RAW_ROOT="${ATOMIC4_RAW_ROOT:-/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic}"
ATOMIC4_V3_ROOT="${ATOMIC4_V3_ROOT:-${CODE_DIR}/outputs/datasets}"
ATOMIC4_MERGED="${ATOMIC4_MERGED:-${ATOMIC4_V3_ROOT}/robocasa_atomic4_v3}"
ATOMIC4_REPO_ID="${ATOMIC4_REPO_ID:-robocasa/atomic4}"

# TOKENIZER_BATCH_SIZE is independent of policy BATCH_SIZE. Tokenizer is action-only
# and uses matched_h20-scale batches (512 on day/full). Policy stays small-batch
# because video decode is the bottleneck. STAGE=test keeps both small.
STAGE="${STAGE:-test}"
case "${STAGE}" in
  test)
    TOKENIZER_STEPS="${TOKENIZER_STEPS:-80}"
    POLICY_STEPS="${POLICY_STEPS:-40}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-4}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-2880}"
    ;;
  day)
    # Policy stays small-batch (video decode bound). Tokenizer is action-only and
    # follows actioncodec matched_h20: batch 512, 20k steps.
    TOKENIZER_STEPS="${TOKENIZER_STEPS:-20000}"
    POLICY_STEPS="${POLICY_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-512}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-2880}"
    ;;
  full)
    TOKENIZER_STEPS="${TOKENIZER_STEPS:-50000}"
    POLICY_STEPS="${POLICY_STEPS:-100000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-512}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-10080}"
    ;;
  *)
    echo "[error] STAGE must be test, day, or full, got: ${STAGE}" >&2
    return 1
    ;;
esac

ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-0.1}"
DECODER_TYPE="${DECODER_TYPE:-diffusion}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-actioncodec}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-${CODE_DIR}/outputs/dlc/${ROBOCASA_TASK,,}_${STAGE}_${RUN_STAMP}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${RUN_ROOT}/tokenizer}"
