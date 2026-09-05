#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET_ROOT="${DATASET_ROOT:-$(pwd)/../data/my_dataset_merged_0902_no_head_96x128}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/tokenizer_matched_h20_overlap_phys_token16}"
REPO_ID="${REPO_ID:-local/my_dataset_merged_0902_no_head_96x128}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_FREQ="${LOG_FREQ:-10}"
OVERLAP_WEIGHT="${OVERLAP_WEIGHT:-0.0004}"
PHYSICAL_RECON_WEIGHT="${PHYSICAL_RECON_WEIGHT:-0.0005}"
PHYSICAL_VELOCITY_WEIGHT="${PHYSICAL_VELOCITY_WEIGHT:-0.008}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "output_dir already exists: ${OUTPUT_DIR}" >&2
  echo "choose a new OUTPUT_DIR or remove the existing directory explicitly" >&2
  exit 1
fi

echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpu=${GPU} steps=${STEPS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "overlap_weight=${OVERLAP_WEIGHT} physical_recon_weight=${PHYSICAL_RECON_WEIGHT} physical_velocity_weight=${PHYSICAL_VELOCITY_WEIGHT}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run lerobot-train-actioncodec-tokenizer \
  --repo_id="${REPO_ID}" \
  --root="${DATASET_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --action_dim=12 \
  --action_horizon=20 --latent_horizon=16 \
  --model_dim=512 --codebook_size=1024 --num_codebooks=1 \
  --encoder_cross_layers=8 --decoder_cross_layers=8 \
  --share_encoder_latent_transformer=true \
  --share_decoder_latent_transformer=true \
  --share_encoder_cross_attn=true \
  --share_decoder_cross_attn=true \
  --vq_beta=1.0 --use_vl_embedder=false \
  --decoder_type=diffusion \
  --alignment_weight=0.1 --hard_alignment_weight=0.0 \
  --overlap_consistency_weight="${OVERLAP_WEIGHT}" --overlap_shift=16 \
  --physical_recon_weight="${PHYSICAL_RECON_WEIGHT}" \
  --physical_velocity_weight="${PHYSICAL_VELOCITY_WEIGHT}" \
  --physical_unit_scale=1.0 \
  --continuous_action_indices='[0,1,2,3,4,6,7,8,9,10]' \
  --window_stride=4 \
  --learning_rate=2e-4 --adam_beta1=0.9 --adam_beta2=0.95 \
  --grad_clip=1.0 --lr_warmup_steps=1000 --lr_min_ratio=0.1 \
  --batch_size="${BATCH_SIZE}" --steps="${STEPS}" --num_workers="${NUM_WORKERS}" \
  --device=cuda --amp=true --seed=42 --log_freq="${LOG_FREQ}"
