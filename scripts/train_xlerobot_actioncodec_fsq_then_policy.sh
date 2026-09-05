#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_ROOT="$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128"
export DATASET_ROOT="${DATASET_ROOT:-$(pwd)/../data/my_dataset_merged_0902_no_head_96x128}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TOKENIZER_OUTPUT_DIR="${TOKENIZER_OUTPUT_DIR:-${RUN_ROOT}/tokenizer_fsq_overlap_phys_token16_full20k}"
POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-${RUN_ROOT}/policy_fsq_295m}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/fsq_logs}"
PAIR_FILE="${PAIR_FILE:-${RUN_ROOT}/continuity_full20k_1024.json}"
POLICY_PAIR_FILE="${POLICY_PAIR_FILE:-${RUN_ROOT}/fsq_heldout_pairs128.json}"
for output in "${TOKENIZER_OUTPUT_DIR}" "${POLICY_OUTPUT_DIR}"; do
  if [[ -e "${output}" || -L "${output}" ]]; then
    echo "output already exists: ${output}" >&2
    exit 1
  fi
done
mkdir -p "${LOG_DIR}"
record_exit() {
  status=$?
  echo "FSQ chain exit=${status} $(date -Iseconds)"
  echo "${status}" > "${LOG_DIR}/chain.exit_code"
}
trap record_exit EXIT
echo "FSQ chain PID=$$ started $(date -Iseconds)"
OUTPUT_DIR="${TOKENIZER_OUTPUT_DIR}" STEPS=20000 \
  bash scripts/train_xlerobot_actioncodec_tokenizer_fsq.sh 2>&1 | tee "${LOG_DIR}/tokenizer20k.log"
uv run --no-sync python scripts/verify_xlerobot_actioncodec_fsq.py \
  --tokenizer-path "${TOKENIZER_OUTPUT_DIR}" 2>&1 | tee "${LOG_DIR}/tokenizer20k_verify.log"
TOKENIZER_PATH="${TOKENIZER_OUTPUT_DIR}" OUTPUT_DIR="${POLICY_OUTPUT_DIR}" POLICY_STEPS=50000 \
  bash scripts/train_xlerobot_actioncodec_fsq_295m.sh 2>&1 | tee "${LOG_DIR}/policy50k.log"
uv run --no-sync python scripts/eval_xlerobot_actioncodec_tokenizer_continuity.py \
  --tokenizer-path "${TOKENIZER_OUTPUT_DIR}" --dataset-root "${DATASET_ROOT}" \
  --pair-file "${PAIR_FILE}" --sample-count=1024 --seed=20260904 --device=cuda \
  --continuous-action-indices='[0,1,2,3,4,6,7,8,9,10]' \
  --output-json "${RUN_ROOT}/continuity_fsq20k_1024.json" 2>&1 | tee "${LOG_DIR}/tokenizer_eval.log"
# Require EMA for every requested checkpoint, keeping the comparison convention explicit.
for step in 005000 010000 015000 020000 025000 030000 035000 040000 045000 050000; do
  test -d "${POLICY_OUTPUT_DIR}/checkpoints/${step}/pretrained_model_ema"
done
uv run --no-sync python scripts/eval_xlerobot_actioncodec_policy_physical.py \
  --checkpoint-root "${POLICY_OUTPUT_DIR}" --dataset-root "${DATASET_ROOT}" \
  --pair-file "${POLICY_PAIR_FILE}" --eval-split=0.1 --seed=20260905 --device=cuda \
  --continuous-action-indices='[0,1,2,3,4,6,7,8,9,10]' \
  --output-json "${RUN_ROOT}/policy_fsq_physical_heldout128_all_ema.json" \
  2>&1 | tee "${LOG_DIR}/policy_eval.log"
for baseline in policy_oat_exact_native_96x128_nocrop_torchcodec_dp_scale_295m policy_codebook_distance_295m; do
  if [[ -d "${RUN_ROOT}/${baseline}/checkpoints/050000/pretrained_model_ema" ]]; then
    uv run --no-sync python scripts/eval_xlerobot_actioncodec_policy_physical.py \
      --checkpoint-root "${RUN_ROOT}/${baseline}" --dataset-root "${DATASET_ROOT}" \
      --steps 50000 --pair-file "${POLICY_PAIR_FILE}" --eval-split=0.1 --seed=20260905 --device=cuda \
      --continuous-action-indices='[0,1,2,3,4,6,7,8,9,10]' \
      --output-json "${RUN_ROOT}/fsq_comparison_${baseline}_heldout128_ema50k.json" \
      2>&1 | tee "${LOG_DIR}/comparison_${baseline}.log"
  else
    echo "50k EMA comparison pending: ${baseline}"
  fi
done
echo "FSQ chain completed $(date -Iseconds)"
