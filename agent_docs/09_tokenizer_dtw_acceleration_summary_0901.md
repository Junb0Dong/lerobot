# Tokenizer 训练中的 DTW 加速总结（2026-09-01）

## 结论

Tokenizer 训练中的 DTW 加速主要分为两条线：

1. 加速 Soft-DTW 动态规划本身：接入 CUDA Soft-DTW，并实现按 candidate pair 维度批量向量化的 PyTorch fallback。
2. 精简 ActionTokenizer/ActionCodec 的 alignment training hot path：减少 DTW 候选数量，只计算被选中的 embedding pair，并移除不参与当前训练目标的数据和诊断计算。

两条线的目标不同：第一条降低单次 DTW 计算成本；第二条减少每个 tokenizer training step 实际需要完成的工作量。

## 1. Soft-DTW 计算本身的加速

### 原始瓶颈

Batch 内需要比较大量 action chunk pair。旧 Torch 路径会对每个 pair 单独执行 Soft-DTW，且每次动态规划还要遍历 `T × T` 网格。这会产生大量 Python 循环和细碎的 GPU kernel launch；batch 增大后，pair 数量约按 `B²` 增长，DTW 很快成为 tokenizer 的主要瓶颈。

### CUDA Soft-DTW backend

原 ActionCodec 接入了可选的 `softdtw-cuda-torch`：

- 提供 `auto / torch / cuda` backend。
- 先批量构造 `[P, T, T]` step-cost tensor，再由 CUDA Soft-DTW 处理一批 candidate pairs。
- 用 `pair_batch_size` 控制每次处理的 pair 数，默认相关配置为 `8192`，避免一次性占用过多显存。
- DTW distance 只用于 positive-pair mining 时运行在 `no_grad` 下；只有需要 soft alignment matrix 时才通过 autograd 求 cost matrix 的梯度。
- CUDA backend 缺失或输入不在 CUDA 上时显式报错，避免长训练静默退回慢实现。

原 ActionCodec 的正式 tokenizer 配方使用 `chunk_align_dtw_backend=cuda`。当前 LeRobot 尚未接入该 extension：`auto/torch` 使用下面的向量化 Torch backend，显式指定 `cuda` 会报错。

### Pair 维向量化的 PyTorch DP

LeRobot 迁移时参考 ActionTokenizer 的 DTW DP 思路，把 Torch fallback 从“逐 pair 计算”改为“在 pair 维度批量计算”：

- 先收集 `P` 个 unordered candidate pairs。
- 一次性构造 `[P, T, T]` cost tensor。
- DP 仍按依赖关系顺序遍历 `T × T` 网格，但每个 cell 都用一个向量化 Tensor op 同时更新全部 `P` 个 pair。
- 训练中不再出现外层逐 pair 的标量 Python Soft-DTW 循环。

因此算法复杂度仍是 `O(P × T²)`，但 Python 循环和 kernel launch 数从随 `P` 增长，降为主要只随 `T²` 增长，GPU 并行度显著提高。

### 实测结果

4090 微基准：

| 配置 | 旧标量实现 | Pair 维向量化 |
| --- | ---: | ---: |
| `B=8` | 约 1579 ms | 约 70 ms |
| `B=128`，8128 pairs | 未记录 | 约 70 ms |

- `B=8` 的 DTW 计算约加速 **22.5×**。
- XLeRobot 正式 tokenizer 使用快 DTW 后约为 **0.15 s/step**。
- 旧慢 run 保存在 `outputs/xlerobot_actioncodec/tokenizer_interrupted_pre_fast_dtw`；快实现对应当时的正式输出 `outputs/xlerobot_actioncodec/tokenizer`。

## 2. ActionTokenizer alignment hot-path 加速

除了优化 DTW kernel，本阶段还减少了 tokenizer 每一步不必要的计算和数据搬运。

### 稀疏 embedding pair distance

旧 `semantic_contrastive_loss` 会先构造完整的 `[B, B, D]` embedding difference tensor，再通过 positive/negative mask 选择少数 pair。

优化后先从 mask 提取 pair indices，只对被 mined 的 positive/negative pairs 做 `index_select` 和 squared distance，不再物化完整 `[B, B, D]` tensor。这降低了显存占用和无效算术量。

### 限制 DTW candidate pairs

- h20 的 Spatial、LIBERO-10、All-4 tokenizer preset 将 `chunk_align_max_candidate_pairs` 从 `2048` 降为 `1024`。
- `positive_topk`、Soft-DTW backend、gamma 和 alignment loss weight 保持不变。
- h32 historical preset 保留 `2048`，避免改变历史实验口径。

这相当于直接把 h20 每步需要精排的 unique DTW pairs 减半。

### 移除不参与当前目标的训练期工作

- tokenizer pretrain 不再计算 TCL、temporal overlap、adjacent-token overlap 和 TCL proxy。
- HDF5 loader 增加 `load_temporal_neighbors`；训练 preset 关闭后，不再读取或搬运 `prev_action/next_action`。
- 移除 trainer 中 TCL diagnostic interval 和 TCL early-stop gate 的实际使用。
- 兼容调用方所需的部分旧 output key 保留为零值。
- overlap 等分析仍可由离线 `scripts/analyze_tokenizer.py` 显式加载 temporal neighbors 后计算，不影响独立评估能力。

### 4k-step A/B 结果

Spatial h20、seed 42：

| 指标 | Baseline | Optimized |
| --- | ---: | ---: |
| wall time | 1825 s | 1745 s |
| step 3990 logged elapsed | 1797.9 s | 1713.9 s |
| median step time | 0.430 s | 0.420 s |
| best val reconstruction loss | 0.044640 | 0.040179 |
| full-val direct-decode MSE | 0.208732 | 0.193155 |
| full-val direct-decode MAE | 0.296830 | 0.283783 |
| code usage | 0.9141 | 0.9150 |
| perplexity | 649.12 | 663.68 |
| unique DTW candidate pairs | 2048 | 1024 |

结果：

- wall-clock 缩短约 **4.4%**。
- median step throughput 提升约 **2.4%**。
- 4k focused A/B 中 reconstruction、code usage 和 perplexity 未见退化。
- 该 A/B 只证明 hot-path 优化没有明显破坏短程训练质量，不能替代完整 20k training 和 downstream policy rollout。

## 3. 三个概念的关系

| 层级 | 主要改动 | 解决的问题 |
| --- | --- | --- |
| 外部 CUDA library | `softdtw-cuda-torch` backend | 用专用 CUDA 实现加速 Soft-DTW |
| Torch DTW 实现 | `[P,T,T]` cost + pair-axis vectorized DP | extension 不可用时避免逐 pair Python DP |
| Tokenizer training hot path | 候选 2048→1024、稀疏 pair distance、删除无效 diagnostics/data | 减少每个 training step 实际需要完成的工作 |

一句话总结：先把 Soft-DTW 从“每个 pair 单独跑 DP”改造成 GPU 批量计算，再通过候选限流、稀疏 embedding loss 和关闭无效的 temporal/TCL 训练期工作，进一步降低 tokenizer alignment 的总体开销。

## 4. 代码与历史记录

- LeRobot 向量化 Torch Soft-DTW：`src/lerobot/actioncodec/losses/soft_dtw.py`
- LeRobot 迁移及微基准：`agent_docs/03_actioncodec_lerobot_milestone_0830.md`
- 原 ActionCodec CUDA backend：`/home/junbo/ActionTokenizer/actioncodec/actioncodec/losses/soft_dtw.py`
- ActionTokenizer pair-axis hard-DTW 参考：`/home/junbo/ActionTokenizer/action_tokenizer/src/action_tokenizer/losses/dtw.py`
- Alignment hot-path A/B：`/home/junbo/ActionTokenizer/actioncodec/docs/agent/31_tokenizer_training_time_optimization_2026-08-05.md`
- Hot-path 优化提交：`a0d78aa perf: speed up tokenizer alignment training`
