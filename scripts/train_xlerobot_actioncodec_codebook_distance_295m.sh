#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Matched 295M CE baseline: only add normalized codebook-distance risk.
export TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/tokenizer_matched_h20_overlap_phys_token16_full20k}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/policy_codebook_distance_295m}"
export POLICY_EMBED_DIM=1024 POLICY_N_LAYERS=15 POLICY_N_HEADS=16
export POLICY_STEPS="${POLICY_STEPS:-50000}"
export CODEBOOK_DISTANCE_LOSS_WEIGHT="${CODEBOOK_DISTANCE_LOSS_WEIGHT:-0.5}"
export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"

exec bash scripts/train_xlerobot_actioncodec_oat_exact_native_nocrop_torchcodec.sh "$@"
