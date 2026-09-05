# XLeRobot ActionCodec policy 动作幅度诊断（2026-09-04）

## 1. 问题与结论

真机部署中，ActionCodec policy 的动作幅度明显大于 LeRobot 原生 Diffusion Policy。当前证据表明：

- tokenizer 仍有数个 raw action unit 的重建与连续性残差，但不是动作放大的主要来源；
- 主要放大发生在 semantic policy 的 autoregressive token prediction 与泛化阶段；
- 50k policy 明显过拟合，并且训练目标只有 teacher-forced token CE，没有直接约束 deterministic
  decoded action 的 physical MAE、velocity、overlap 或 rollout seam；
- 部署侧量纲、action key 顺序和控制节拍仍需在真实部署机器上核对，但它们不能解释离线 replay 中
  已经存在的大幅动作。

因此，当前问题应优先按 **policy training/checkpoint selection 问题**处理，而不是继续单独增加
tokenizer 容量或把责任归因于部署发送倍率。

## 2. 已确认的实际对象

- dataset：`/home/junbo/data/my_dataset_merged_0902_no_head_96x128`
- dataset contract：100 episodes，30 FPS，12D action/state
- tokenizer：horizon 20，latent horizon 16，single-codebook VQ 1024，diffusion decoder
- rollout：每次执行前 16 actions 后重新规划
- 实际部署 checkpoint：
  `outputs/my_dataset_merged_0902_no_head_96x128/policy_oat_exact_native_96x128_nocrop_torchcodec_overlap_phys_tokenizer/checkpoints/050000/pretrained_model_ema`
- checkpoint 配置：native `[96,128]`，`crop_shape=null`，`temperature=0`，ACTION MEAN_STD

`/home/junbo/my_lerobot` 是从另一台机器 clone 下来、仅用于静态分析的代码副本。其 launcher、相对
checkpoint path、cwd 和 import path 不能视为实际部署链路，也不能据此认定真实部署故障。

## 3. Tokenizer 独立质量

Overlap/physical tokenizer 完整 20k 训练后，在固定 4,096 个 episode-safe pairs 上相对 baseline：

- arm reconstruction MAE：`1.230 -> 0.984`，下降 20.0%；
- same-time overlap p50/p95/p99：`4.555/12.684/19.860 -> 3.184/9.120/13.638`；
- seam excess p50/p95：`3.719/11.558 -> 2.545/8.044`；
- physical velocity error p95：`0.516 -> 0.408`；
- codebook occupancy：`1024/1024`，没有明显 collapse。

这说明新 objective 有效，tokenizer 不是 codebook collapse，也不是单纯容量完全不足。不过，独立窗口
detokenize 仍会留下约 3--10 raw-unit 的 overlap/seam 尾部误差，因此 tokenizer 是误差下限的一部分。

结果文件：

- `outputs/my_dataset_merged_0902_no_head_96x128/continuity_baseline_4096.json`
- `outputs/my_dataset_merged_0902_no_head_96x128/continuity_full20k_4096.json`

## 4. Exact EMA 离线 replay

对实际部署的 50k EMA 做了只读 deterministic replay：使用
`continuity_full20k_1024.json` 的前 128 个固定、episode-safe `t/t+16` pairs，加载 checkpoint
自带的 preprocessor/postprocessor，保持 `temperature=0`。没有连接真机，也没有使用分析 clone 的
launcher。

Arm 指标均为 raw action units，除 normalized maximum 外：

| 指标                                         |   p50 |   p95 |   p99 |
| -------------------------------------------- | ----: | ----: | ----: |
| policy normalized action maximum             | 2.049 | 3.450 | 3.825 |
| policy 首个 target 相对当前 state delta      | 19.84 | 47.50 | 60.79 |
| GT rollout seam `A[15] -> B[0]`              |  0.85 |  1.97 |  4.31 |
| tokenizer-only rollout seam                  |  3.38 | 10.05 | 13.09 |
| policy rollout seam                          | 16.16 | 49.28 | 75.84 |
| tokenizer-only same-time overlap             |  3.44 | 10.93 | 13.25 |
| policy same-time overlap                     | 17.61 | 51.59 | 75.69 |
| tokenizer-only per-window reconstruction MAE |  1.06 |  2.80 |  4.32 |
| policy per-window reconstruction MAE         |  6.12 | 12.80 | 17.62 |
| tokenizer-only velocity error                | 0.078 | 0.517 | 1.014 |
| policy velocity error                        | 0.156 | 0.978 | 1.659 |

相同 GT action 经 tokenizer 独立 encode/decode 时，误差明显小于 policy 先预测 tokens 再 decode 的
误差。policy seam 的 p50/p95 约为 tokenizer-only 的 4.8/4.9 倍，说明主要误差不是 diffusion
detokenizer 单独造成，而是 policy 预测了错误或不连续的 token sequence。

## 5. Policy 训练证据

Overlap/physical tokenizer 对应的 policy 训练到 50k 时：

- train loss/token CE：约 `0.488`；
- train token accuracy：约 `0.834`；
- train top-5 accuracy：约 `0.990`；
- held-out eval loss：约 `2.7397`；
- held-out eval loss 在约 5k 后持续恶化。

训练 CE 继续下降而 held-out loss 上升，说明 50k checkpoint 明显过拟合。EMA 只能平滑同一训练轨迹
的权重，不能修复 teacher-forced token CE 与 decoded physical action 指标不一致的问题。

此外，token error 在 autoregressive generation 中会级联；某个早期 token 预测错误后，后续 token
条件也随之偏离。当前 loss 不会直接惩罚最终 action chunk 离当前 state 太远、velocity 过大或两个
重规划 chunk 在 seam 处跳变。

## 6. 部署侧判断边界

离线 replay 已使用实际 EMA 和其保存的 normalization artifacts，policy 输出在进入 robot/ZMQ 前就
已经偏大，因此统一发送倍率不是主要根因。以下部署项仍可能进一步放大风险或影响观感，但尚未在真实
部署机器上确认：

- 采集与部署是否一致使用 degrees 或 normalized `[-100,100]` position units；
- observation state、postprocessed action 和 host 写入的关节顺序是否完全一致；
- 每 16 步同步 refill 的推理延迟是否造成周期性停顿；
- 是否有 per-joint rate limit、target clamp、chunk-boundary blending 或异步预取。

这些安全与控制层检查不能代替 policy 修复；即使部署映射完全正确，当前 50k EMA 仍会输出过大的
first-target delta 和 seam。

## 7. 建议的下一步顺序

1. 用同一 128--1,024 个固定 pairs 评估 `005000` 到 `050000` 的所有 EMA checkpoints，不按最后一步
   或单独 token CE 选模型；以 decoded action MAE、first-target delta、velocity、overlap 和 seam
   综合选择候选。
2. 在完全相同的 dataset observations 上 replay 原生 Diffusion Policy，记录 normalized output、
   postprocessed output、first-target delta 和 maximum action delta，建立直接基准。
3. 若较早 ActionCodec checkpoint 明显更好，先选择较早 checkpoint；不要直接重新训练另一个 50k
   last checkpoint。
4. 下一轮 policy 训练保留 token CE，同时加入可微的 decoded physical reconstruction、velocity、
   overlap/seam 约束，并使用 episode-level held-out split 进行 checkpoint selection。
5. 真机前在实际部署机器确认量纲和 action key mapping，并保留 per-joint rate limiter、异常 target
   rejection 和急停；控制层平滑只作为安全层，不能掩盖离线 action 质量问题。

## 8. 当前限制

- 本次没有连接、移动或控制真机；
- 尚未对所有 5k 间隔 EMA checkpoint 做同规模 sweep；
- 尚未在相同 observations 上完成原生 Diffusion Policy A/B；
- 因实际部署发生在另一台机器，分析 clone 不能证明真实 runtime 的 `use_degrees`、import path、控制
  loop cadence 或发送链路配置。

Tokenizer objective、训练与完整 evaluator 的实现细节见
`agent_docs/04_xlerobot_actioncodec_status_0901.md`。

## 9. 仿真 action contract 与 EEF 表示判断（2026-09-04）

此前仿真没有出现同等程度的“动作幅度过大”，不能说明同一 policy/tokenizer 机制在真机 joint action
上也应稳定，因为两边不是同一个 action/control contract：

- 原 ActionCodec 的 LIBERO embodiment 明确定义为 7D **delta EEF**（xyz、rpy、gripper），LIBERO
  使用 `OSC_POSE`，输入先裁到 `[-1,1]`，再把单步 xyz/rpy 缩放到最多约 `0.05 m / 0.5 rad`。
- 当前 RoboCasa wrapper 虽把字段命名为 `end_effector_position/rotation`，PandaOmron 的实际
  `OSC_POSE` 配置仍是 `input_type=delta`，相同地使用 `[-1,1]` 输入和
  `0.05 m / 0.5 rad` 输出范围；LeRobot env postprocessor 还会再次 clip action。
- 当前 XLeRobot 数据则是 30 Hz 的 12D 双臂 absolute joint position target。错误 token 经
  denormalization 后可直接产生离当前反馈几十个 raw units 的目标，servo/部署链路没有仿真 OSC
  那样的 task-space 单步缩放。两者的 raw MAE 数值和危险程度不能直接比较。
- 本地 LIBERO Spatial/10 数据各有 500 episodes，而 XLeRobot 当前只有单任务 100 episodes；后者的
  50k policy 已由 held-out CE 证明过拟合。仿真 success rate 及 action clipping 也可能把错误表现为
  饱和或 rollout 失败，而不是显眼的幅度跳变。

把 action 改成 **absolute EEF** 值得做 matched A/B：task-space trajectory 通常比 joint target 更贴近
任务几何，也能在 IK 前施加 workspace、EEF step 和速度限制，因此可能降低真机上可见的 joint jump。
但它不是当前 policy 误差的自动修复：错误 token 仍可产生远处 absolute pose；IK 在奇异位形、不可达
姿态或不同解支路上还可能放大成 joint jump。XLeRobot 每臂只有 5 个非 gripper joints，full 6D pose
应采用 position-dominant、soft-orientation IK，并用当前/上一 IK 解 warm-start，同时保留 EEF 与
per-joint 双层 rate limit。

若目标是复现仿真的保守幅度，**bounded delta EEF** 比 absolute EEF 更接近原 LIBERO/RoboCasa
contract；其代价是误差会积累、长期 absolute accuracy 较差。建议在同一 100 episodes 上离线用正确
URDF/标定做 FK，构造 `absolute joint`、`absolute EEF`、`bounded delta EEF` 三个严格 matched
数据版本后，分别重训 tokenizer/policy，以 normalized action error、EEF mm/degree error、IK 后
joint delta/seam 和 rollout success 选表示。当前 action 的 degree/`[-100,100]` 量纲尚未在实际采集链
确认，量纲确认前不能直接做 FK 转换。

本节只读核对：本地 LIBERO HDF5 action 分布与 trajectory 数；原 ActionCodec embodiment 配置；当前
LeRobot LIBERO/RoboCasa wrapper；RoboSuite 固定 commit `5ce6643...` 的 PandaOmron controller 配置。
未连接真机，也未修改训练 checkpoint。

## 10. Semantic policy physical auxiliary implementation (2026-09-04)

已完成最小可用实现，默认新增 loss weights 全为 `0`，旧 policy/config 的 CE、inference API 保持兼容。开启
auxiliary 后，训练数据流为：

```text
observation -> policy logits -> ST Gumbel codebook embedding
           -> frozen full 27-step diffusion decoder -> normalized action
           -> ACTION mean/std raw units -> Huber physical losses
```

保留 teacher-forced CE 和 decoded metrics；`prefix_corruption_prob=1` 的 auxiliary 分支使用带 ST embedding
的 differentiable autoregressive generation，`0<p<1` 是 scheduled sampling，`p=0` 是 teacher-forced decoded
loss。paired overlap/seam 使用同一 episode 内 `t/t+16` 的两个 policy observation/action windows，并在
训练和 held-out eval dataloader 中共同封装；不跨 episode 或 split。physical stats 从训练时传入的
`dataset.meta.stats["action"]` 获取，evaluator 使用 checkpoint 保存的 processor stats 做 inference
unnormalization。

验证：

- `uv run --no-sync pytest tests/actioncodec -q`：49 passed；
- 修改 Python 文件的 `ruff check` 和 `ruff format --check` 已执行；`bash -n` 与 `git diff --check` 已通过；
- 100-step CUDA smoke 使用独立 `/tmp/actioncodec_policy_physical_aux_smoke_20260904_v3`，训练/eval 均完成，
  并记录 token CE、teacher-forced/free-running decoded reconstruction、velocity、overlap、seam；
- 单 checkpoint/单 pair evaluator smoke 已输出 JSON，覆盖新 smoke checkpoint 以及既有 020000 checkpoint。

尚未验证：完整 5k 间隔 EMA sweep、正式长训的稳定性/显存上限、真实部署机 action key/单位和真机安全表现。
下一步按 `CE baseline -> + decoded reconstruction -> + velocity/first-target -> + prefix corruption/free-running
-> + paired overlap/seam` 做固定 pair、held-out episode 和真机前 offline replay 的 ablation；不要在本 session
直接启动正式训练或连接真机。

正式 50k launcher 已新增：`scripts/train_xlerobot_actioncodec_oat_exact_physical_aux_nocrop_torchcodec.sh`；本次仅完成
脚本语法检查，未自动启动正式训练。

## 11. 295M capacity-matched policy replay (2026-09-05)

评估范围限制：以下前 128 pairs 覆盖 episodes 0–13，属于 policy training episodes，不是 held-out；
可用于定位 replay seam，不能作为泛化或 held-out checkpoint selection 结论。

扩容 policy 已完成 50k 训练：`embed_dim=1024`、`n_layers=15`、`n_heads=16`，使用同一 overlap/physical
tokenizer。固定 `continuity_full20k_1024.json` 前 128 个 episode-safe `t/t+16` pairs，arm indices 为
`[0,1,2,3,4,6,7,8,9,10]`，对 50k EMA 的 raw action replay 指标如下：

| 指标                               |                 GT |      tokenizer recon |       256D policy EMA |       295M policy EMA |
| ---------------------------------- | -----------------: | -------------------: | --------------------: | --------------------: |
| seam p50 / p95 / p99               | 0.85 / 1.97 / 4.31 | 3.38 / 10.05 / 13.09 | 16.16 / 49.28 / 75.84 | 17.33 / 46.10 / 71.27 |
| chunk 内最大 delta p50 / p95 / p99 | 1.44 / 3.32 / 5.00 |   1.00 / 2.20 / 2.69 |    1.13 / 2.29 / 3.28 |    1.10 / 2.12 / 2.91 |
| first-target delta p50 / p95 / p99 | 6.08 / 8.79 / 9.18 |                    — | 19.84 / 47.50 / 60.79 | 17.92 / 46.02 / 57.90 |

结论：295M 扩容没有消除动作幅度问题。chunk 内部相邻 action 跳变并不大，甚至略低于 GT；主要异常仍是
重规划时的 chunk seam，295M 相对 256D 的 p95/p99 seam 有小幅改善，但 p50 略差，不能视为解决方案。
295M 的 seam 仍约为 tokenizer recon 的 4.6 倍、GT 的 23 倍（p95）。结果文件为
`policy_dp_scale_295m_physical_arm128.json`、`policy_dp_scale_295m_physical_arm128_all_ema.json` 和
`tokenizer_full20k_continuity_128.json`；未连接真机。

## 11. Physical auxiliary 训练吞吐诊断（2026-09-05）

当前已在运行 `policy_oat_exact_physical_aux_0904`：batch 2、bf16、8 workers、
`auxiliary_batch_fraction=1`、`prefix_corruption_prob=1`，五项 physical weights 均非零。
只读 TensorBoard 检查 step 200–500：step 时间 1.315–1.503 s，update 1.303–1.485 s，
DataLoader 等待 0.0014–0.0065 s，吞吐 1.33–1.52 个主窗口/s。50k 纯训练按当前窗口约需
18–21 小时，未计 eval/save；本轮瓶颈已不是此前 AV1 数据加载。

源码确认：每步 A/B 两个窗口分别执行一次 16-token differentiable AR + 27-step decoder 并反传；
另外每个窗口各执行一次仅用于 teacher-forced metrics 的 27-step no-grad decode，因此合计
108 次 denoiser forward，54 次属于需反传的分支。A 窗口还重复 tokenize/teacher forward。
冻结 decoder 参数不会省掉 physical loss 到 policy latent 的输入梯度计算。
`sample_differentiable()` 每个 timestep 的 `step.item()`、denoiser 每步的 `if not mask.any()`
会引入 CPU/GPU 同步；实际各项占比尚未经 profiler 单独测量。

建议先将 teacher decoded metrics 改为低频计算、缓存固定 schedule 并优化单 embodiment 路径，
再测 batch 4/8 与 A/B 合批的 samples/s 和 peak memory。当前训练进程 nvidia-smi 快照约
4.6 GiB；同卡另有 physical evaluator，不能将单次 GPU utilization 当作独占利用率。
进一步可考虑 CE 全 batch、auxiliary 子集或每 K 步执行，但这些改变梯度采样/频率，须用 held-out
physical metrics 验证；batch 2 时 fraction=0.25 仍至少处理 1 个样本，不能省至四分之一。
27 改为更少 decode steps 会改变 auxiliary 对部署解码器的对齐，不作为首选。

本次仅诊断并记录，未改训练代码、配置或运行进程。下一步若实施，先做上述保持目标的优化，
再以相同 batch/数据比较稳定窗口耗时、loss/gradient 与显存；吞吐提升倍数目前未实测。

## 12. Physical auxiliary 提速实现与验证（2026-09-05）

用户已授权实施。新增 `policy.decoded_metrics_interval`（兼容默认 1），正式 physical-aux
launcher 显式设为 100，smoke launcher 设为 10，均可用 `DECODED_METRICS_INTERVAL` 覆盖。
训练每 N 次 forward 才执行额外的 teacher/free decoded diagnostics，eval 始终完整计算；
loss-bearing 分支每步仍执行，27-step diffusion、physical weights、batch 默认 2 均保持。
训练频率计数不受 eval forwards 影响；恢复进程后频率计数重新开始。应让指标频率与 log_freq
一致；teacher/free sampled metrics 只平均实际测量的 forwards，不再代表所有训练 batch。

- 固定 sampling indices 缓存在 CPU，避免每个 diffusion timestep 的 CUDA `item()`；
  single-embodiment denoiser 直接调用 head，省掉每步 `mask.any()`/boolean indexing。
  多 embodiment 保留原有 dispatch，checkpoint 参数结构不变。
- 保留 teacher forward 的 dropout 抽样，额外 free-running 诊断通过 `fork_rng` 隔离 dropout，
  因此改变诊断频率不会改变 paired loss 的 RNG、数值或梯度。
- 补强了既有 backward test：修正 decoder_type 设置位置，并直接检查 diffusion 输入 latents
  收到非零 finite 梯度，避免仅凭 CE 梯度就误认为 physical 分支畅通。

真实数据固定 episode-local pairs、完整 12D tokenizer、native 三相机、A100/bf16 短测：
每种配置 3 次 warmup + 8 次计时，包含 forward/backward、clip、AdamW step；lr=0 保持权重
固定。批次预先载入，未计 DataLoader、EMA、eval/save，不能视为正式长训的端到端速度。

| 配置                      | batch | median update s | 主窗口/s | peak allocated GiB |
| ------------------------- | ----: | --------------: | -------: | -----------------: |
| 修改前快照                |     2 |          1.4065 |    1.422 |              3.664 |
| 仅同步优化，仍逐步诊断    |     2 |          1.3629 |    1.467 |              3.664 |
| 同步优化 + 低频诊断普通步 |     2 |          1.1297 |    1.770 |              3.664 |
| 同上                      |     4 |          1.1338 |    3.528 |              3.994 |
| 同上                      |     8 |          1.1295 |    7.083 |              4.671 |

当前 p=1、batch 2 下，修改前后完整模型 loss=`9.612282752990723`，251 个梯度 tensors
逐项比较最大绝对差为 0。batch 2 普通步耗时下降约 19.7%；batch 8 增大的是每步样本数，
相同 50k steps 将看到 4 倍主窗口，不能把其约 5 倍于旧 batch-2 的吞吐误解为 50k 用时缩短 5 倍。

验证：ActionCodec suite 62 passed；新增指标频率的配置迁移/CLI 解析、p=0/0.5/1 下 paired
loss/grad/RNG 对照、eval 完整指标、CPU/CUDA schedule 与 single-head 输出/梯度对照均通过；
Ruff check/format、两个 launcher 的 `bash -n` 和 `git diff --check` 通过。测试依赖通过
`uv run --no-sync --with pytest python -m pytest tests/actioncodec -q` 临时提供，未同步训练环境。

本次未停止原进程，也未重启正式训练；检查时原训练已退出且原输出目录没有 checkpoint。
下一步用独立新 OUTPUT_DIR 启动修改后的 launcher；可先选择 BATCH_SIZE=8 验证完整稳定窗口，
按样本量和 held-out physical metrics 重新考虑 steps，避免机械沿用 50k。benchmark 位于临时
目录，仅作本次验证，正式运行继续使用仓库已有 launcher。

## 12. FSQ / geometry-aware policy 方案讨论（2026-09-05）

源码核对：当前 VQ 按 encoder latent 到 learned codebook 的平方距离分配 token；policy 基础 CE 不使用
码本距离，AR token embedding 也不是 tokenizer codebook。token ID 的数值差本身没有物理含义。
已有可选 ST decoded physical auxiliary 路径，不应重复实现。

FSQ 提供规则的低维量化网格，但 packed token + 普通 CE 仍不利用网格距离，latent 邻近也不保证
decoded action 邻近。若做 FSQ，应对归一化坐标增加距离/ordinal objective，或保留 packed 1024-way
head、用各类别对应的 FSQ 坐标计算 expected distance；例如 levels=[8,8,4,4] 保持每 latent 10 bits。
直接回归坐标可行，但需处理多模态平均和训练/推理 rounding 一致性。

最小消融建议（尚未实施/启动）：先在现有 frozen VQ 上加 CE + expected codebook-distance risk，
不改 tokenizer、AR head 或 inference，验证利用 latent geometry 是否有效；再决定是否重训 FSQ
tokenizer/policy。FSQ token 语义与旧 checkpoint 不兼容；更换 quantizer 不会消除 27-step decoder
开销。chunk seam 仍应在 decoded action 的同一物理时刻约束，不直接对齐不同窗口的 latent slots。
验收复用 episode-level held-out、free-running greedy replay 的 recon、overlap、seam p95/p99，
并保持 reductions 一致。用户本轮仅讨论可行性；未修改模型/训练代码或运行进程。

## 13. 295M codebook-distance policy 最小实现（2026-09-05）

用户随后批准基于约 300M policy 修改并试跑。已实现 `CE + weight * E_p[d(code, GT code)]`：
平方欧氏距离除以整个冻结码本的 mean pairwise squared distance（`2 * sum(var(code))`），
不做逐向量单位化、不改变原 VQ geometry；当前实际码本归一化分母约 `10.2714`。
排除 BOS，AMP 下距离/softmax 使用 FP32，梯度仅回到 policy/vision，默认 weight=0 完全保持 CE。
没有新参数或 state-dict keys，不经过 diffusion decoder，也未启用已有 decoded physical losses。
同时确保 `task_token_swap_ce_gap` 始终比较 CE，而非减去混合 total loss。

复用原 native launcher，仅增加 weight、跳过 sync 和短跑日志间隔入口；新增薄 wrapper：
`scripts/train_xlerobot_actioncodec_codebook_distance_295m.sh`。默认 `1024D/15 layers/16 heads`，
weight=0.5、batch32、50k、seed1000、90/10 episode split、同一 frozen overlap/physical tokenizer，
保持 BF16、EMA、学习率及推理不变。从头训练，**不是续训原 50k weights**，输出独立
`outputs/my_dataset_merged_0902_no_head_96x128/policy_codebook_distance_295m`。

验证：

- `uv run --no-sync python -m pytest tests/actioncodec -q`：66 passed；Ruff check/format、shell
  syntax、`git diff --check` 通过。测试覆盖近错/远错、期望距离而非平均向量抵消、AMP、归一化
  缩放/平移不变性、冻结码本、policy/vision 梯度、零权重兼容和配置保存迁移。
- 实际 295,048,160 trainable params、batch32 的 100-step CUDA/BF16/EMA smoke 完成，退出码0。
  训练末段 CE=2.52821、distance=0.55983、weighted distance=0.27992、total=2.80813；
  稳态约7 step/s，训练 logger peak allocation 7.08 GiB。
- smoke eval 仅取32个 held-out samples，step100 CE=1.50145、distance=0.59078、total=1.79683；
  仅验证链路，不能据此证明泛化或 seam 改善。
- 普通/EMA checkpoint 保存成功；EMA 重载，合成 observation 下 BF16 action `[1,20,12]` finite；
  同一权重将 loss weight 从0.5切为0，预测逐元素一致。
- 当前环境缺少 pytest/ruff，按 uv.lock 版本补装测试工具；未同步/裁剪训练依赖。

Smoke 路径：`/tmp/actioncodec_codebook_distance_295m_smoke_20260905`。复现命令：

```bash
OUTPUT_DIR=/tmp/actioncodec_codebook_distance_295m_smoke_20260905 \
POLICY_STEPS=100 LOG_FREQ=10 EVAL_STEPS=50 SAVE_FREQ=100 \
bash scripts/train_xlerobot_actioncodec_codebook_distance_295m.sh --max_eval_samples=32
```

已存在目录不可重复使用；复跑需另设 OUTPUT_DIR。正式50k尚未启动，旧 checkpoint 未覆盖，未连接真机。
正式入口：`bash scripts/train_xlerobot_actioncodec_codebook_distance_295m.sh`。
下一步正式训练后，用相同 held-out episode pairs 比较旧295M CE与新loss的 free-running
reconstruction、overlap、seam p95/p99；不要把混合 eval_loss 与旧纯CE直接比较。

## 14. 295M distance-loss 完整训练后 held-out replay（2026-09-05）

正式 `policy_codebook_distance_295m` 已完成50k（约13:33），保存普通/EMA checkpoint；本次用户要求
检查动作跳变。复用 `scripts/eval_xlerobot_actioncodec_policy_physical.py`，对新模型及原295M纯CE的
50k EMA 进行 CUDA FP32、greedy temperature=0 离线自由生成，使用各 checkpoint 自带 processors。
同一 tokenizer、架构、seed1000、90/10 train/eval split；新模型 distance weight=0.5。

本次 seed=20260905、eval_split=0.1、stride4、shift16、batch8，固定512个 episode-safe pairs；
覆盖全部 held-out episodes90–99（各44–71 pairs），不是先前 training episodes0–13的128-pair样本。
arm indices `[0,1,2,3,4,6,7,8,9,10]`，不含 grippers；单位 raw action units，未确认是 degrees。

| 指标（p50 / p95 / p99）          |                    GT |                原295M CE |         295M CE+distance |
| -------------------------------- | --------------------: | -----------------------: | -----------------------: |
| seam，最大关节 `abs(B[0]-A[15])` | 0.806 / 2.159 / 2.803 | 17.571 / 49.920 / 67.592 | 16.875 / 50.695 / 73.519 |
| 每20步chunk内最大相邻action差    | 1.361 / 2.858 / 4.662 |    1.153 / 2.200 / 2.631 |    1.147 / 2.170 / 2.604 |
| 同时刻overlap，4步×10关节mean    |             0 / 0 / 0 |  6.795 / 15.507 / 20.661 |  6.807 / 16.403 / 22.085 |

结论：chunk间跳变仍严重。新模型seam p95/p99约为GT的23.5/26.2倍；相对旧CE，seam p50下降
4.0%，p95上升1.6%，p99上升8.8%，没有显示出接缝尾部改善。chunk内依然比GT平滑，因此主要问题
仍是不同重规划窗口的轨迹不一致。新模型first-target/state delta p50/p95/p99=22.74/52.79/80.04。
latent-distance surrogate本轮未解决physical seam；不能因此断言所有权重、早期checkpoint或FSQ都无效。

结果：outputs同一数据集父目录下 `policy_codebook_distance_295m_heldout512_050000.json` 和
`policy_ce_295m_heldout512_050000.json`。新文件的pair_starts传给旧模型作为pair-file，已断言512个
pairs及全部GT quantiles一致。未修改训练/模型代码、未连接真机、未中止同卡正在运行的FSQ训练。
下一步优先验收显式decoded reconstruction/paired overlap/seam约束；若选择更早checkpoint，仍用
这批held-out pairs及同口径指标，不拿不同样本上的旧结果直接比较。

## 15. Outputs checkpoint 清理（2026-09-05 16:42）

按用户要求，仅保留三个 policy 的最新完整 checkpoint（普通+EMA+training_state+last）及对应 TB：

- `policy_oat_exact_native_96x128_nocrop_torchcodec_dp_scale_295m`：050000。
- `policy_codebook_distance_295m`：050000。
- 正在训练的 `policy_fsq_295m`：清理时最新完整035000，TB已记录37000，训练进程未中止。

永久删除上述三个 run 的24个旧step目录，以及其余7个旧/smoke policy 的 checkpoints和TB：
旧 `xlerobot_actioncodec_no_head` vision policy；`xlerobot_actioncodec_0901_no_head` vision/nocrop
两组；当前merged数据集的native、temperature0、overlap_phys_tokenizer三组及FSQ smoke100。
未留恢复副本；**此前提及的这些旧权重及早期checkpoint现已不可用**，相关评估JSON仍保留。
独立 tokenizer artifacts/TB、评估JSON和其他非checkpoint文件未改动，尤其保留当前训练依赖的FSQ
tokenizer及原VQ tokenizer。

`du -sk outputs`：293178256 -> 22087696 KiB，释放约258.5 GiB；清理后outputs约21.1 GiB。
验证38个删除目标均不存在；仅剩上述3个policy checkpoint根目录及policy TB；6份最新普通/EMA
safetensors headers可读、bundled tokenizer和training_state存在、last链接有效，TB事件可读。
本次是一次性清理，没有改变保存策略；运行中的FSQ后续仍会每5k生成新checkpoint。

## 16. A/B/C/D chunk boundary L2 诊断（2026-09-05）

按用户给定 L2 定义重新评估原295M CE与CE+distance的50k EMA，不能沿用旧表的最大单关节差。
新增 `scripts/diagnose_xlerobot_chunk_boundary.py`：从policy held-out episodes90–99起点连续切chunk，
H20有473 chunks/463 boundaries，执行stride16有587 chunks/577 boundaries；无跨episode或尾部padding。
B使用同一checkpoint内frozen tokenizer，C使用真实GT observations自由greedy AR预测；CUDA FP32。
normalization与checkpoint processors、predict_action_chunk路径、deterministic decode均有数值断言。
两个policy的A/B、GT tokens及索引逐元素一致。该held-out划分针对policy，不是tokenizer独立测试集。

| raw L2 p50 / p95 / p99 |                A GT |          B tokenizer |            C CE policy |          C CE+distance |
| ---------------------- | ------------------: | -------------------: | ---------------------: | ---------------------: |
| H20，完整12D           | 1.19 / 3.08 / 45.01 | 4.78 / 14.99 / 45.68 | 35.43 / 75.65 / 104.35 | 38.04 / 76.94 / 108.72 |
| stride16，完整12D      |  1.17 / 3.02 / 5.19 | 4.28 / 13.01 / 43.85 |  31.98 / 69.70 / 94.96 |  30.53 / 69.88 / 95.16 |
| stride16，双臂10D      |  1.16 / 2.90 / 3.79 | 4.25 / 12.14 / 16.75 |  26.79 / 65.80 / 94.30 |  26.35 / 67.92 / 95.16 |

GT grippers只取0/45，H20全12D的GT p99来自开合阶跃，不能误判为双臂不连续。
stride16双臂chunk内部单步p95为A/B/C=2.84/2.76/2.37，而边界p95为2.90/12.14/65.80：
B已出现连续性残差，主要放大发生在C的独立窗口预测；distance-loss未解决。不能单凭此实验继续
分解感知、token classification、AR累积或泛化误差。原始GT也有少数内部较大步进，不代表全部数据完美。

D **未测**：当前没有真实部署机的chunk/实际下发action日志，未连接真机。需提供checkpoint身份、
episode/时间、预测chunk、实际执行长度及postprocess/限幅后commands；脚本有固定stride raw chunk
NPZ入口。变长执行/异步控制必须按真实最后执行action对齐，不能将离线C冒充D。

结果均在同一dataset outputs父目录：`boundary_jump_295m_ce_20260905/`和
`boundary_jump_295m_distance_20260905/`各保存summary.json、chunks.npz；
`boundary_jump_295m_report_20260905/`保存report.md、boundaries.csv及CDF PNG/PDF。
验证：5个L2/时间对齐/episode reset/normalization/nonfinite测试通过；两个完整CUDA评估退出0；
Ruff与git diff --check通过。下一步补D日志，结合预测接缝与执行command接缝判断闭环额外贡献。

## 17. 完训FSQ同口径boundary L2（2026-09-05）

FSQ tokenizer20k、policy50k EMA已完成。复用第16节脚本及全部held-out连续窗口，CUDA FP32/greedy，
H20为463 boundaries，stride16为577 boundaries。FSQ bundled tokenizer四个artifact的SHA256与
独立tokenizer完全一致，FSQ/VQ action stats相同；GT actions、episode/frame及全部A指标逐元素一致。
实际FSQ trainable params=294,028,256，沿用295M run名；levels=[8,5,5,5]、1000码、四scalar heads。

| FSQ raw L2 p50 / p95 / p99 |                A GT |     B FSQ tokenizer |          C FSQ policy |
| -------------------------- | ------------------: | ------------------: | --------------------: |
| H20完整12D                 | 1.19 / 3.08 / 45.01 | 2.73 / 7.32 / 45.02 | 14.04 / 48.28 / 57.34 |
| stride16完整12D            |  1.17 / 3.02 / 5.19 | 2.51 / 6.57 / 10.09 | 11.88 / 47.35 / 53.23 |
| stride16双臂10D            |  1.16 / 2.90 / 3.79 |  2.47 / 6.36 / 8.76 | 10.40 / 32.39 / 46.10 |

stride16双臂p95相对VQ：B下降47.7%，C下降50.8%，全部10个episode的B/C p95均下降。
raw arm reconstruction MAE：B 0.936->0.487（-48.0%），C 8.173->3.216（-60.7%）。
FSQ显著降低本批次的数值误差，但C seam p95仍为GT的11.2倍、B的5.1倍；C内部单步p95仅2.51，
尚未消除独立窗口接缝。GT gripper 0/45阶跃仍单独解释。D缺真实chunk/commands日志，仍未测。
这是quantizer、alignment表示、policy输出头及tokenizer训练dtype同时变化的方案对照，不能把
改善单独归因FSQ量化本身，也不能比较FSQ scalar CE与VQ token CE或据此证明真机success提升。

输出：同一dataset outputs下`boundary_jump_295m_fsq_20260905/`包含summary.json、chunks.npz、
boundaries.csv、report.md、fsq_vs_vq_cdf.png/pdf。完整GPU评估退出0，normalization、公开预测路径
和deterministic decoder断言通过；沿用已有指标测试，本次未改模型或诊断脚本。下一步补D，或用
同一held-out physical口径选FSQ候选checkpoint；不要把tokenizer训练样本称为独立held-out测试集。
