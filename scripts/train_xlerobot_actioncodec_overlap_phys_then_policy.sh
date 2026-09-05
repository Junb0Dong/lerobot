#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET_ROOT="${DATASET_ROOT:-$(pwd)/../data/my_dataset_merged_0902_no_head_96x128}"
TOKENIZER_OUTPUT_DIR="${TOKENIZER_OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/tokenizer_matched_h20_overlap_phys_token16_full20k}"
POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/policy_oat_exact_native_96x128_nocrop_torchcodec_overlap_phys_tokenizer}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -e "${TOKENIZER_OUTPUT_DIR}" || -L "${TOKENIZER_OUTPUT_DIR}" ]]; then
  echo "tokenizer output_dir already exists: ${TOKENIZER_OUTPUT_DIR}" >&2
  exit 1
fi

if [[ -e "${POLICY_OUTPUT_DIR}" || -L "${POLICY_OUTPUT_DIR}" ]]; then
  echo "policy output_dir already exists: ${POLICY_OUTPUT_DIR}" >&2
  exit 1
fi

echo "starting 20k tokenizer training"
DATASET_ROOT="${DATASET_ROOT}" \
OUTPUT_DIR="${TOKENIZER_OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
STEPS=20000 \
bash scripts/train_xlerobot_actioncodec_tokenizer_overlap_phys_no_head.sh

for artifact in model.safetensors model_config.json action_stats.json dataset_contract.json; do
  test -f "${TOKENIZER_OUTPUT_DIR}/${artifact}"
done

echo "starting policy training from ${TOKENIZER_OUTPUT_DIR}"
DATASET_ROOT="${DATASET_ROOT}" \
TOKENIZER_PATH="${TOKENIZER_OUTPUT_DIR}" \
OUTPUT_DIR="${POLICY_OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
bash scripts/train_xlerobot_actioncodec_oat_exact_native_nocrop_torchcodec.sh

echo "tokenizer and policy training completed"
