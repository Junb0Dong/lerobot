#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET_ROOT="${DATASET_ROOT:-/home/ainot02/xzd/lerobot_v060/local_datasets/my_dataset_merged}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(pwd)/outputs/xlerobot_actioncodec/tokenizer_matched_h20}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/xlerobot_actioncodec/policy_oat_exact_nocrop}"
GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "output_dir already exists: ${OUTPUT_DIR}"
  echo "removing it so this run can start clean"
  rm -rf "${OUTPUT_DIR}"
fi

CUDA_VISIBLE_DEVICES="${GPUS}" uv run accelerate launch \
  --multi_gpu --num_processes=2 --mixed_precision=bf16 --main_process_port="${MAIN_PROCESS_PORT}" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/xlerobot_actioncodec \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.eval_split=0.1 --dataset.video_backend=torchcodec \
  --dataset.decode_image_size=128 \
  --policy.type=actioncodec --policy.action_dim=14 --policy.num_tasks=1 \
  --policy.tokenizer_path="${TOKENIZER_PATH}" \
  --policy.image_size=128 --policy.crop_shape=null \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=xlerobot_ac_policy_oat_exact_nocrop \
  --batch_size="${BATCH_SIZE}" --steps=50000 --num_workers=4 \
  --log_freq=1000 --eval_steps=1000 --save_freq=1000 --save_checkpoint=true \
  --env_eval_freq=0 \
  --ema.enable=true --ema.power=0.75 --ema.max_decay=0.9999 \
  --accelerator.mixed_precision=bf16 \
  --tensorboard.enable=true --wandb.enable=false
