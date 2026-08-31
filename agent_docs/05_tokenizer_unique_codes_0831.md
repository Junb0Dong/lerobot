# Tokenizer unique_codes 偏低：原因与 matched_h20 对齐（2026-08-31）

对照 `../actioncodec` matched_h20。独立 tokenizer 默认值已对齐该配方；CLIP 仍关闭；EMA 未加。

## 现象（旧 run）

`outputs/xlerobot_actioncodec/tokenizer`：`unique_codes` 收尾约 30，`loss_vq`/`loss_align` 很快到 0。当时配置是 batch=8、`model_dim=256`、1 轮 cross-attn、`vq_beta=0.25`。该 256-d checkpoint 与新默认不兼容，需要重训。

## 指标口径

- `unique_codes_batch`（TB 仍写一份 `unique_codes`）：当前 batch 去重数。
- `codebook_occupied_window` / `codebook_usage_window`：最近 2000 step 占用的码数 / 占比。
- `codebook_occupied_total` / `codebook_usage_total`：开训以来占用。
- `codebook_perplexity_window`：窗口直方图 perplexity。

不要把 batch unique 当成 1024 词表的全局占用。

## 当前默认（对齐 matched_h20）

- 模型：`model_dim=512`，cross_layers=8，四个 `share_*=True`，`vq_beta=1.0`，`use_vl_embedder=False`
- 优化：AdamW `(0.9, 0.95)`，`grad_clip=1.0`，warmup 1000 + cosine（`min_lr_ratio=0.1`），CUDA AMP。AMP 下 denoiser/perceiver 的 embodiment head 必须用 head 输出的 dtype 建缓冲（LayerNorm 仍是 fp32、Linear 是 fp16），与源仓库一致。
- 数据：`batch_size=512`，`steps=20000`，`window_stride=4`，只采完整 horizon 窗。tokenizer dataloader 设 `decode_videos=False`（源配方 CLIP 关闭时 `load_image=auto` 也不读图）。昨天慢 DTW 优化仍在；B=512 时旧路径会解三路 pyav，GPU-Util 掉到 0%。
- loss：soft-DTW `0.1`，`chunk_align_max_candidate_pairs=1024`，`dtw_backend=auto`
- DLC：`TOKENIZER_BATCH_SIZE` 与 policy `BATCH_SIZE` 分开；day/full tokenizer batch=512、steps=20000

未搬：EMA 码本（`action_tokenizer` 路线）。
