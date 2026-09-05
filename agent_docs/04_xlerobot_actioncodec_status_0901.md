# XLeRobot ActionCodec 当前状态（更新：2026-09-05）

本文保留当前训练 contract、入口和兼容性边界。动作幅度与接缝诊断、已删除 checkpoint 的
清单和评测结果统一见 [10_xlerobot_policy_amplitude_diagnosis_0904.md](10_xlerobot_policy_amplitude_diagnosis_0904.md)；
DTW 加速结论见 [09_tokenizer_dtw_acceleration_summary_0901.md](09_tokenizer_dtw_acceleration_summary_0901.md)。
源码、当前配置及磁盘 artifact 优先于历史运行记录。

## 数据与模型 contract

- 当前 launcher 使用 `../data/my_dataset_merged_0902_no_head_96x128`，100 episodes、三路 RGB
  `head/left_wrist/right_wrist`、原生 96×128、12D action/state（双臂各 6D）。删除 head joint
  不等于删除 head camera。旧 51-episode `my_dataset_0901_no_head` 的统计不能用于当前数据。
- Tokenizer：horizon 20、latent horizon 16、model dim 512、8 轮共享 cross-attention、
  3-layer/8-head latent self-attention、diffusion decoder；训练不读取相机。
- Action 以 artifact 的 mean/std 做 MEAN_STD normalization；policy STATE 使用 MIN_MAX。
  Arm 量纲仍需以采集/部署配置确认，不能仅根据数值把 raw units 称为 degrees。
- Policy：2 帧 history，自回归 16 tokens，解码 20 actions，执行前 16 steps；tokenizer 冻结。
  OAT attention、task conditioning、KV cache、checkpoint 内 processors 保持一致。
- 视觉默认 `oat_exact_robomimic`，每相机独立 `ResNet18Conv + SpatialSoftmax(32)`，64D 特征、
  GroupNorm、RGB `[-1,1]`、ObservationEncoder ReLU。原生分辨率入口关闭 crop 和 worker resize。
  `resnet_spatial` / `small_cnn` 仍用于兼容已有配置，不做 silent fallback。
- 小 policy 为 d256/l4/h4；295M 配方为 d1024/l15/h16。VQ 295M 配方有 295,048,160
  trainable params，FSQ 配方有 294,028,256 trainable、342,655,984 total params。

## VQ、FSQ 与辅助目标

- VQ：单个 learned `1024×512` codebook、straight-through、vq_beta=1、soft-DTW alignment=.1；
  CLIP、EMA codebook 未启用。occupancy 看 window/total 与 perplexity，不用单 batch unique
  codes 代替词表占用。
- FSQ：`512 -> 4 -> [8,5,5,5] -> 512`，basis `[1,8,40,200]`、vocab 1000；无 learned codebook、
  commitment loss、dead-code refresh。DTW 沿用 mining/pooling/loss，但用量化后 4D coordinates。
  Policy 每 token 同时输出四个 scalar heads；训练取四 CE 均值，同时记录各头 accuracy、
  完整 token accuracy 和四 CE 之和 `token_nll`。默认 greedy，不直接与 VQ CE 比大小。
- FSQ 使用显式 BF16 tokenizer AMP，dtype 写入 artifact；VQ 保持 FP16 默认。
  FP16 FSQ 曾在 step0 遇到 non-finite gradient，BF16 的 100-step 验证通过。
- Tokenizer overlap/physical 配方：batch512、20k、seed42、stride4、shift16、DTW .1、
  overlap .0004、physical recon .0005、velocity .008，continuous indices
  `[0,1,2,3,4,6,7,8,9,10]`，unit_scale=1。每 batch 256 对窗口共享 timestep 和对齐 union noise。
- VQ policy 可选 codebook-distance loss：CE + .5×expected normalized squared code distance，
  codebook 冻结、FP32 geometry；它是 physical error 的 surrogate，不能保证连续性。
- VQ policy decoded auxiliary：action=.5、velocity=.1、first_target=.5、overlap=.2、seam=.2，
  unit_scale=10、Gumbel-ST temperature=.7、prefix_corruption=1、shift16。梯度经冻结 decoder
  回到 policy/obs encoder；完整 27-step decode 保留，额外诊断默认每 100 forwards，eval 每次计算。
  Diffusion sampling indices 缓存在 CPU，单 embodiment denoiser 避免逐步 CUDA scalar 同步。
- 所有辅助权重兼容默认 0；FSQ 当前拒绝非零 policy distance/decoded auxiliary losses。

## 有效入口

| 用途                                        | `scripts/` 下入口                                                                                    |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 基础 VQ tokenizer                           | `train_xlerobot_actioncodec_tokenizer_no_head.sh`                                                    |
| VQ overlap/physical tokenizer               | `train_xlerobot_actioncodec_tokenizer_overlap_phys_no_head.sh`                                       |
| 原生 96×128 VQ policy                       | `train_xlerobot_actioncodec_oat_exact_native_nocrop_torchcodec.sh`                                   |
| 历史 crop baseline                          | `train_xlerobot_actioncodec_oat_exact_torchcodec.sh`                                                 |
| VQ tokenizer → policy                       | `train_xlerobot_actioncodec_overlap_phys_then_policy.sh`                                             |
| 295M distance policy                        | `train_xlerobot_actioncodec_codebook_distance_295m.sh`                                               |
| Decoded auxiliary policy                    | `train_xlerobot_actioncodec_oat_exact_physical_aux_nocrop_torchcodec.sh`                             |
| FSQ tokenizer / policy                      | `train_xlerobot_actioncodec_tokenizer_fsq.sh` / `train_xlerobot_actioncodec_fsq_295m.sh`             |
| FSQ 完整串行链路                            | `train_xlerobot_actioncodec_fsq_then_policy.sh`                                                      |
| FSQ artifact 验收                           | `verify_xlerobot_actioncodec_fsq.py`                                                                 |
| Tokenizer continuity / policy physical eval | `eval_xlerobot_actioncodec_tokenizer_continuity.py` / `eval_xlerobot_actioncodec_policy_physical.py` |
| A/B/C/D 接缝 L2 诊断                        | `diagnose_xlerobot_chunk_boundary.py`                                                                |

启动参数及 smoke 替代命令见 [scripts/README.md](../scripts/README.md)。串行 FSQ launcher 会训练
20k tokenizer → artifact 验收 → 50k policy → 固定 pairs 离线评测；新 run 的默认 sweep 需要
全部每 5k EMA checkpoint。已有 run 的早期 checkpoint 清理后应显式指定现存 step。

## 兼容性与评测边界

- 历史 14D 数据、12D 数据不可混用；更改 crop/image geometry 必须重训视觉 policy。
  VQ artifact 缺少 quantizer_type 时按 VQ strict-load；FSQ/VQ mismatch 拒绝加载。
- Policy checkpoints 保存 tokenizer 和 processors；replay 必须使用 checkpoint 对应的
  normalization、greedy/EMA 约定和同一 episode-safe pairs。
- `dataset.eval_split` + `eval_steps` 是 offline held-out loss/metrics；`lerobot-eval` 是仿真
  benchmark；`lerobot-rollout` 是真机部署。训练期 closed-loop rollout/TopK 保存尚未实现。
- GT/tokenizer/policy 离线对照已确认主要接缝放大发生在 policy 独立窗口预测；295M distance
  50k 未改善 seam 尾部。Tokenizer 仍有连续性残差，不能将其单独视为全部误差来源。
- 最新 FSQ 20k tokenizer / 50k EMA 已完成；同口径 stride16 双臂 seam p95 为 tokenizer 6.36、
  policy 32.39，较 VQ 明显降低但仍高于 GT 2.90。完整对照见诊断文档第 17 节，真机 D 仍未测。
- FSQ、VQ 当前没有 ACT temporal ensemble，也没有 DP warm-start；chunk 内平滑不等于接缝连续。
  ACT ensemble 在 decoded 同时刻 actions 上融合，不能平均 token IDs。
- 真机 D 尚无部署日志；`my_lerobot` 是分析用 clone，不可据其 launcher 推断真实部署链路。
- 2026-09-05 已清理旧 checkpoint：保留的 policy families 是 295M CE、295M distance、FSQ。
  历史小模型及早期 checkpoint 已不可用；tokenizer、评估 JSON 和当前 policy TB 保留。

## real-robot 分支整理（2026-09-05）

- 目标远端 `https://github.com/Junb0Dong/lerobot`，从 main 建立 `real-robot`。
- 保留 FSQ、physical auxiliary、no-crop/native vision、离线诊断及正式回归测试。
- 移除独立 physical-aux smoke 脚本、已被 native 入口替代的旧方形 no-crop 脚本、空的项目
  lessons 副本；全局 corrections 统一在 `~/.codex/lessons.md`。
- 清理基础 tokenizer launcher 的 token32 实验残留，恢复 contract 要求的 token16 和与 policy
  一致的 `tokenizer_matched_h20` 输出名；crop baseline 移除自动删除既有输出的代码。
- 排除本地 `clash-for-linux-install/` 工具目录，测试缓存清理；数据、训练产物不提交。
- 验证：ActionCodec + batch preprocessing 85 passed；10 个 shell 入口通过 bash -n，8 个训练
  launcher 和文档 smoke 命令通过真实 draccus 配置解析；既有输出保护检查、Ruff、uv lock --check
  和 git diff --check 通过；待提交文件的全部适用 pre-commit hooks（含 mypy、Bandit、密钥扫描）通过。
- 本次清理 95 个生成缓存目录；环境、数据、训练产物保留在本地。
- 下一步：沿用同一 held-out physical 口径选择 FSQ 候选；如做真机验证，先确认 robot、
  ports、calibration、camera mapping、供电和急停方法，并采集 D 的实际执行日志。
