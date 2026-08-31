# ActionCodec oat-exact policy 补齐移植（2026-08-31）

对照 `/home/ainot02/junbo/actioncodec` 的 `origin/experiment/oat-exact-policy`。
P0 模型 + P1 训练配方 + P2 指标/tests 已迁入；训练期 closed-loop rollout（P3）未做。

## 当前默认（对齐源 `semantic_actioncodec_oat_task_token` + `policy_oat_exact`）

| 项 | 值 |
| --- | --- |
| vision | `resnet_spatial`（每相机独立 ResNet + SpatialSoftmax 32 kp → 64-D） |
| crop | train/eval 都做 76×76 per-image torch RNG crop（`small_cnn` 不做） |
| RGB | encoder 内 `minus_one_to_one`；processor `VISUAL=IDENTITY` |
| STATE | processor `MIN_MAX`；ACTION 仍 `MEAN_STD`（冻结 tokenizer 合约） |
| AR | Xavier + tied head；`task_token_init_seed=42` |
| 优化器 | policy_lr=5e-5，obs_encoder_lr=1e-5，betas=(0.9, 0.95)，clip=1.0，2D decay / 1D no-decay 四组 |
| scheduler | constant warmup 100 steps |
| 采样 | `temperature=1.0`，`top_k=10` |
| 指标 | `token_ce` / `token_accuracy` / `token_top5_acc`；eval 且 `num_tasks>=2` 时 `task_token_swap_ce_gap` |

可选 `vision_encoder=oat_exact_robomimic`（需自装 robomimic）。`small_cnn` 留给旧 checkpoint 和 CI。

## 参数量（2026-08-31，xlerobot 3 相机 / 14-DoF）

在 `my_dataset_merged` 特征上实例化：3 路 RGB、`observation.state`/`action` 均为 14 维。ActionCodec 用当前 oat-exact 配方（`resnet_spatial`、`crop_shape=None`、`tokenizer_matched_h20`）。

| 模型 | 总参数 | 可训练 | 构成 |
| --- | --- | --- | --- |
| LeRobot ACT 默认（`n_decoder_layers=1`，`use_vae=True`） | 51.6M | 51.6M | 共享 ResNet-18 11.2M + encoder 17.3M + decoder 5.4M + VAE encoder 17.3M |
| ACT 论文设定（`n_decoder_layers=7`） | 83.9M | 83.9M | 文档写的 ~80M 对应这一档；LeRobot 为对齐原仓库 bug 默认只用 1 层 decoder |
| Semantic policy 可训练部分 | 13.4M | 13.4M | 每相机独立 ResNet+SpatialSoftmax 2.8M×3=8.4M + AR decoder 5.0M |
| 冻结 tokenizer（`tokenizer_matched_h20`） | 49.2M | 0 | encoder 12.6M + diffusion decoder 35.7M + VQ 0.5M |
| Semantic 全量（policy + tokenizer） | 62.6M | 13.4M | 训练时 tokenizer 冻结 |

可训练参数：semantic 是 ACT 默认的约 26%（13.4M vs 51.6M）。相机数几乎不影响 ACT（backbone 共享），但会线性增加 semantic 的 vision encoder。

## 与源 YAML 未自动套上的 trainer 开关

LeRobot 的 EMA / bf16 在 `TrainPipelineConfig`，不改全库默认。推荐：

```bash
--ema.enable=true --ema.power=0.75 --ema.max_decay=0.9999
--accelerator.mixed_precision=bf16
```

DLC `lerobot_dlc_train_policy` 已带上这两项。

本机 `--ema.enable=true` 需要 `diffusers`。**不要**单独 `uv sync --extra diffusion`（会按 extra 重建环境，卸掉 `accelerate`/`datasets`）。正确：

```bash
uv sync --locked --extra training --extra diffusion
```

失败启动若已写出 `output_dir`（哪怕只有 `tb/`），再跑同一路径会 `FileExistsError`。无 checkpoint 时可删目录重开；有 checkpoint 则换目录或 `--resume=true`。

本机启动脚本：`bash scripts/train_xlerobot_actioncodec_oat_exact.sh`（GPU 6,7，**torchcodec** + `--dataset.decode_image_size=128`，batch 16）。对照：`bash scripts/train_xlerobot_actioncodec_oat_exact_torchcodec.sh`（GPU 4,5，同样 decode_image_size=128，batch 128，输出 `policy_oat_exact_nocrop_torchcodec`，port 29503）。目录已存在会先删再训。

本机 nocrop 旧实测（pyav + 三路 AV1 640×480，无 worker resize）：稳态 `step_s≈0.36s`（`data_s≈0.18` / `updt_s≈0.21`）。应用 torchcodec cache + worker 128 后需重测 `step_s`。

## 解码加速（2026-08-31）

- `decode_video_frames` 把 `decoder_cache` 传给 torchcodec。`DatasetReader` 在 worker 内 lazy 建 LRU，容量 `max(100, n_episodes × n_rgb_videos)`（本数据 train ≈141）。
- `--dataset.decode_image_size=128`：decode 后、`image_transforms` 前，worker 内 bilinear 缩到 128 uint8。ACT 默认 `None`，仍解原生分辨率。
- 验证：`uv run pytest tests/configs/test_default.py tests/datasets/test_decode_image_size.py tests/datasets/test_video_decoder_cache.py tests/datasets/test_dataset_reader.py` 41 passed。

## 对照官方 ACT 墙钟（2026-08-31，只查）

源码公式（`docs/source/hardware_guide.mdx`）：`steps_per_epoch = ceil(frames / (num_gpus × batch_size))`。`AGENT_GUIDE.md` 表用单卡 `ceil(frames / batch_size)`，两卡时必须乘 GPU 数。ACT `n_obs_steps=1`（`configuration_act.py` 非 1 即报错）；录制命令 1 路 `front`，§5.7 推荐 2 路。ActionCodec 合约锁 `n_obs_steps=2`。

| 项 | 官方 ACT 5 epoch 锚点 | 我们 pyav | 我们 torchcodec |
| --- | --- | --- | --- |
| 数据 | ~50 ep / 45k 帧，1–2 路 640×480 | 47 train ep / 38464 帧，3 路 AV1 g=2 | 同左 |
| 全局 batch | 8（1×4090） | 16×2=32 | 128×2=256 |
| 5 epoch 的 step | **28,125**（45k/8） | **6,010**（38464/32） | **755**（ceil 38464/256=151） |
| 实际 `--steps` | 文档 5 epoch；CLI 默认 100k | **50,000 ≈ 41.6 epoch** | **50,000 ≈ 333 epoch** |
| 每 sample 解图 | 1–2 路 × 1 obs | 3 路 × 2 obs = 6 张 | 同左 |
| 每 step 解图 | 8–16 张 | 192 张 | 1,536 张 |
| 解码器 | 默认 torchcodec（有则用之） | 显式 pyav | torchcodec，batch 过大 OOM |
| 实测 | 30–60 min | ~0.36 s/step → 5 万 step ≈ 5 h + eval | ~1.19 s/step 后 OOM |

主因是 step 预算按 ACT 的 3–8 万抄了，但全局 batch 是 32/256 而非 8，等价多训了约 **8× / 66× epoch**。同配方若只跑 5 epoch（~6010 step × 0.36 s）约 **36 min**，与 ACT 锚点同量级。13M 模型不是墙钟瓶颈。`eval_steps=1000` 且无 `max_eval_samples` 时，约 50 次全量 eval（~5k 帧 × ~68 s）再加约 1 h。

## Breaking change

`outputs/xlerobot_actioncodec/policy` 等 **small_cnn + MEAN_STD state + temperature=0** checkpoint 不能直接加载新默认 encoder。要加载旧权重必须显式 `--policy.vision_encoder=small_cnn`，且旧 `config.json` 没有该字段时 `from_pretrained` 会走新默认并在 `load_state_dict` 处失败。

确定性 RoboCasa 评测可显式 `--policy.temperature=0`；新训产物默认 1.0+top_k=10，与源 formal eval 一致。旧 ckpt 仍保存 `temperature=0`。

## 验证

`uv run pytest tests/actioncodec -svv`：26 passed，1 skipped（无 robomimic）。ruff check/format 通过。

## 下一步（P3，未做）

训练循环内 closed-loop rollout、按 `mean_success_rate` 的 TopK checkpoint。仿真评测继续走现有 `lerobot-eval` / DLC RoboCasa 入口。
