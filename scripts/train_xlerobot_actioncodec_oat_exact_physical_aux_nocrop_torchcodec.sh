#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET_ROOT="${DATASET_ROOT:-$(pwd)/../data/my_dataset_merged_0902_no_head_96x128}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/tokenizer_matched_h20_overlap_phys_token16_full20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/policy_oat_exact_physical_aux_0904}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29526}"
BATCH_SIZE="${BATCH_SIZE:-2}"
POLICY_STEPS="${POLICY_STEPS:-50000}"
DECODED_METRICS_INTERVAL="${DECODED_METRICS_INTERVAL:-100}"

if [[ "${SKIP_UV_SYNC:-0}" != 1 ]]; then
  uv sync --locked \
    --extra training \
    --extra diffusion \
    --extra actioncodec
fi

if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "output_dir already exists: ${OUTPUT_DIR}" >&2
  echo "choose a new OUTPUT_DIR or remove the old run explicitly" >&2
  exit 1
fi

# Auxiliary losses use the complete deterministic 27-step decoder. temperature=0
# controls inference-time token generation; it is not a training-time amplitude constraint.
echo "decoded weights: action=0.5 velocity=0.1 first_target=0.5 overlap=0.2 seam=0.2"

CUDA_VISIBLE_DEVICES="${GPUS}" uv run --no-sync accelerate launch \
  --num_processes=1 --mixed_precision=bf16 --main_process_port="${MAIN_PROCESS_PORT}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/xlerobot_actioncodec \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.eval_split=0.1 --dataset.video_backend=torchcodec \
  --dataset.return_uint8=true --dataset.decode_image_size=null \
  --policy.type=actioncodec --policy.action_dim=12 --policy.num_tasks=1 \
  --policy.tokenizer_path="${TOKENIZER_PATH}" \
  --policy.vision_encoder=oat_exact_robomimic \
  --policy.image_size='[96,128]' --policy.crop_shape=null \
  --policy.decoded_action_loss_weight=0.5 \
  --policy.decoded_velocity_loss_weight=0.1 \
  --policy.decoded_first_target_loss_weight=0.5 \
  --policy.decoded_overlap_loss_weight=0.2 \
  --policy.decoded_seam_loss_weight=0.2 \
  --policy.continuous_action_indices='[0,1,2,3,4,6,7,8,9,10]' \
  --policy.physical_unit_scale=10.0 \
  --policy.token_relaxation=gumbel_st \
  --policy.token_relaxation_temperature=0.7 \
  --policy.auxiliary_batch_fraction=1.0 \
  --policy.decoded_metrics_interval="${DECODED_METRICS_INTERVAL}" \
  --policy.prefix_corruption_prob=1.0 \
  --policy.auxiliary_seed=42 \
  --policy.overlap_shift=16 \
  --policy.temperature=0 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=xlerobot_actioncodec_policy_physical_aux_0904 \
  --batch_size="${BATCH_SIZE}" --steps="${POLICY_STEPS}" --num_workers=8 \
  --prefetch_factor=4 --persistent_workers=true \
  --dataloader_multiprocessing_context=spawn \
  --log_freq=100 --eval_steps=5000 --max_eval_samples=256 \
  --save_freq=5000 --save_checkpoint=true --env_eval_freq=0 \
  --ema.enable=true --ema.power=0.75 --ema.max_decay=0.9999 \
  --accelerator.mixed_precision=bf16 \
  --tensorboard.enable=true --wandb.enable=false "$@"
