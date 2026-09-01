# XLeRobot ActionCodec 关键状态（2026-09-01）

本文是 XLeRobot 数据、tokenizer 和 semantic policy 的唯一状态摘要。只记录当前有效 contract、兼容性边界和下一步；历史调查与临时运行细节不再保留。

## 1. 当前数据 contract

当前目标数据集是 `/home/junbo/data/my_dataset_0901_no_head`：

- LeRobot v3.0，51 episodes、51,487 frames、30 FPS、1 task。
- `action` 和 `observation.state` 均为 12D：左臂 6D + 右臂 6D。
- 三路相机仍为 `head`、`left_wrist`、`right_wrist`；删除 head joint 不等于删除 head camera。
- 数值数据均为 finite，索引连续，action/state stats 已与 parquet 重算结果核对。
- `horizon=20`、`window_stride=4` 时有 12,650 个完整 episode-local action windows。

旧 `my_dataset_merged` 是 14D（双臂 12D + head joints 2D）。12D 与 14D 的 tokenizer、policy checkpoint 不兼容，不能混用。

## 2. Tokenizer contract

新 tokenizer 必须针对 12D 数据重训，固定使用 matched-h20 配方：

| 项 | 当前值 |
| --- | --- |
| action horizon | 20 |
| latent horizon | 16 |
| action dim | 12 |
| codebook | 单码本，vocab 1024 |
| encoder | model dim 512，8 轮 shared cross-attention |
| decoder | diffusion |
| loss | soft-DTW weight 0.1，最多 1024 candidate pairs |
| training | batch 512，20k steps，stride 4，CUDA AMP |
| input | `decode_videos=False`，tokenizer 不读取相机视频 |

旧 256D、batch 8、单轮 cross-attention、`vq_beta=0.25` 的 checkpoint 已作废。新默认 `vq_beta=1.0`，CLIP 和 EMA codebook 均未接入。

码本指标必须按以下口径解释：

- `unique_codes_batch`：当前 batch 的去重数，不代表整个 1024 词表的占用。
- `codebook_occupied_window`：最近窗口内使用过的 code 数。
- `codebook_occupied_total`：开训以来使用过的 code 数。
- 判断 collapse 时优先看 window/total occupancy 和 perplexity，不能只看单个 batch 的 unique codes。

启动脚本是 `scripts/train_xlerobot_actioncodec_tokenizer_no_head.sh`，默认输出到
`outputs/xlerobot_actioncodec_0901_no_head/tokenizer_matched_h20`。正式训练尚未启动；必须先做独立
100-step smoke，再用新的空目录跑 20k，不能覆盖旧 14D tokenizer。

## 3. Semantic policy contract

Policy 使用冻结的 12D tokenizer，当前 oat-exact 默认如下：

| 项 | 当前值 |
| --- | --- |
| observations/actions | `n_obs_steps=2`，预测 horizon 20，执行前 16 actions |
| vision | 每相机独立 `ResNet18Conv + SpatialSoftmax(32)`，输出 64D |
| crop | robomimic `CropRandomizer`，76×76 |
| normalization | RGB encoder 内 `[-1,1]`；STATE `MIN_MAX`；ACTION `MEAN_STD` |
| AR policy | 256D、4 layers、4 heads、tied token head、task embedding |
| optimizer | policy LR `5e-5`，vision LR `1e-5`，warmup 100 |
| sampling | train/formal eval 默认 `temperature=1.0`、`top_k=10`；确定性评测显式设 0 |
| trainer | 推荐 bf16，并启用 EMA |

`oat_exact_robomimic` 是新默认。`resnet_spatial` 和 `small_cnn` 只用于旧 checkpoint；不同 vision encoder 的权重形状不兼容，不做 silent fallback 或自动迁移。

Semantic policy 的 AR 主体与原 OAT 基本同构。主要差异在冻结 tokenizer：semantic tokenizer 是约 49.15M 参数的 Perceiver + learned VQ + iterative diffusion decoder；OATTok 是约 5.81M 的 register Transformer + FSQ + single-pass decoder。因此 semantic 完整模型更大、detokenize 也更慢。XLeRobot 三相机配置约 87.79M 总参数，其中约 38.64M 可训练。

## 4. 当前状态与下一步

已完成：

- 12D no-head 数据 contract 审计。
- matched-h20 tokenizer 配方和单卡启动脚本。
- oat-exact observation encoder、AR policy、processor、checkpoint 加载和指标接入。
- 旧 vision/tokenizer checkpoint 的兼容性边界已明确。

尚未完成：

1. 运行 12D tokenizer 100-step smoke，确认 shape、OOM、non-finite 和初期 codebook usage 均正常。
2. 在独立空目录完成 20k tokenizer 训练，并核对 `model.safetensors`、`model_config.json`、`action_stats.json`、`dataset_contract.json` 中的 `action_dim=12`、`horizon=20`。
3. Policy launcher 已统一指向 `my_dataset_0901_no_head`、新 tokenizer 输出目录和
   `policy.action_dim=12`；待 tokenizer 产出后再启动 policy 训练。
4. 训练期 closed-loop rollout/TopK checkpoint 尚未实现；正式仿真评测继续使用现有 `lerobot-eval` / DLC RoboCasa 路径。

## 5. 2026-09-01 提交前验证

- 数据审计：51,487 行、51 episodes、12D action/state、finite、全局 index 连续；h20/stride4
  得到 12,650 个完整窗口。
- `uv run --no-sync pytest tests/actioncodec -q`：29 passed。
- ActionCodec 修改文件通过 Ruff check/format；三个训练脚本通过 `bash -n`。
- `uv lock --check` 与 `git diff --check` 通过。
