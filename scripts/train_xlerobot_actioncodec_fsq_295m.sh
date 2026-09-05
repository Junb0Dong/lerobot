#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/tokenizer_fsq_overlap_phys_token16_full20k}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/policy_fsq_295m}"
export POLICY_EMBED_DIM=1024 POLICY_N_LAYERS=15 POLICY_N_HEADS=16
export POLICY_STEPS="${POLICY_STEPS:-50000}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29517}"
export CODEBOOK_DISTANCE_LOSS_WEIGHT=0.0 SKIP_UV_SYNC=1
exec bash scripts/train_xlerobot_actioncodec_oat_exact_native_nocrop_torchcodec.sh \
  --policy.quantizer_type=semantic_fsq --policy.fsq_levels='[8,5,5,5]' \
  --policy.codebook_size=1000 --seed=1000 "$@"
