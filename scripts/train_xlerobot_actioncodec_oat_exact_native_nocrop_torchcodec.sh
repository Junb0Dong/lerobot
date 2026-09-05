#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET_ROOT="${DATASET_ROOT:-$(pwd)/../data/my_dataset_merged_0902_no_head_96x128}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/tokenizer_matched_h20}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/my_dataset_merged_0902_no_head_96x128/policy_oat_exact_native_96x128_nocrop_torchcodec_temperature0}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29505}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-300}"
POLICY_EMBED_DIM="${POLICY_EMBED_DIM:-256}"
POLICY_N_LAYERS="${POLICY_N_LAYERS:-4}"
POLICY_N_HEADS="${POLICY_N_HEADS:-4}"
POLICY_STEPS="${POLICY_STEPS:-100000}"
CODEBOOK_DISTANCE_LOSS_WEIGHT="${CODEBOOK_DISTANCE_LOSS_WEIGHT:-0.0}"

if [[ "${SKIP_UV_SYNC:-0}" != 1 ]]; then
  uv sync --locked \
    --extra training \
    --extra diffusion \
    --extra actioncodec
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output_dir already exists: ${OUTPUT_DIR}"
  echo "choose a new OUTPUT_DIR or remove the old run explicitly"
  exit 1
fi

# The dataset stores all RGB cameras at 96x128. Keeping decode_image_size=null avoids
# worker-side square resizing, and crop_shape=null disables robomimic CropRandomizer.
# temperature=0 makes action chunk generation deterministic.
LEROBOT_VIDEO_DECODER_CACHE_SIZE="${VIDEO_DECODER_CACHE_SIZE}" \
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
  --policy.embed_dim="${POLICY_EMBED_DIM}" --policy.n_layers="${POLICY_N_LAYERS}" \
  --policy.n_heads="${POLICY_N_HEADS}" \
  --policy.codebook_distance_loss_weight="${CODEBOOK_DISTANCE_LOSS_WEIGHT}" \
  --policy.temperature=0 --policy.top_k=3 \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=xlerobot_ac_policy_oat_exact_native_96x128_nocrop_torchcodec \
  --batch_size="${BATCH_SIZE}" --steps="${POLICY_STEPS}" --num_workers=8 \
  --prefetch_factor=4 --persistent_workers=true \
  --dataloader_multiprocessing_context=spawn \
  --log_freq="${LOG_FREQ:-1000}" --eval_steps="${EVAL_STEPS:-5000}" \
  --save_freq="${SAVE_FREQ:-5000}" --save_checkpoint=true \
  --env_eval_freq=0 \
  --ema.enable=true --ema.power=0.75 --ema.max_decay=0.9999 \
  --accelerator.mixed_precision=bf16 \
  --tensorboard.enable=true --wandb.enable=false "$@"
