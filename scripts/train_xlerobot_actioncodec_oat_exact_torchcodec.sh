#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET_ROOT="${DATASET_ROOT:-$(pwd)/../data/my_dataset_0901_no_head}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/outputs/xlerobot_actioncodec_0901_no_head/tokenizer_matched_h20}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/xlerobot_actioncodec_0901_no_head/policy_oat_exact_vision_torchcodec}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
BATCH_SIZE="${BATCH_SIZE:-32}"

uv sync --locked \
  --extra training \
  --extra diffusion \
  --extra actioncodec

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "output_dir already exists: ${OUTPUT_DIR}"
  echo "removing it so this run can start clean"
  rm -rf "${OUTPUT_DIR}"
fi

CUDA_VISIBLE_DEVICES="${GPUS}" uv run --no-sync accelerate launch \
  --num_processes=1 --mixed_precision=bf16 --main_process_port="${MAIN_PROCESS_PORT}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/xlerobot_actioncodec \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.eval_split=0.1 --dataset.video_backend=torchcodec \
  --dataset.decode_image_size=128 \
  --policy.type=actioncodec --policy.action_dim=12 --policy.num_tasks=1 \
  --policy.tokenizer_path="${TOKENIZER_PATH}" \
  --policy.vision_encoder=oat_exact_robomimic \
  --policy.image_size=128 --policy.crop_shape='[76,76]' \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=xlerobot_ac_policy_oat_exact_vision_torchcodec \
  --batch_size="${BATCH_SIZE}" --steps=50000 --num_workers=4 \
  --log_freq=1000 --eval_steps=5000 --save_freq=5000 --save_checkpoint=true \
  --env_eval_freq=0 \
  --ema.enable=true --ema.power=0.75 --ema.max_decay=0.9999 \
  --accelerator.mixed_precision=bf16 \
  --tensorboard.enable=true --wandb.enable=false
