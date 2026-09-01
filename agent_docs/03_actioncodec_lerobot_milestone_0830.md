# ActionCodec 接入 LeRobot：训练里程碑与部署缺口（2026-08-30）

把 `/mnt/data/junbo/actioncodec-oat-exact-policy` 的语义 tokenizer + semantic policy 迁进本仓库。当前里程碑：**训练链路和 RoboCasa rollout 接口已打通**（数据 → tokenizer → 冻结 tokenizer 训 policy → checkpoint 重载 → 原生 `lerobot_eval`）。正式 simulator 结果仍以 DLC smoke/rollout 产物为准。

细节（loss 修复、DLC 踩坑、吞吐数字）见 `agent_docs/02_policy_dataset_dataloader_migration_0828.md` 第 9 节。本文只记结论和接下来要做的事。

## 1. 冻结的 contract

| 项 | 值 |
| --- | --- |
| 数据 | LeRobotDataset **v3**（原始 robocasa 是 v2.1，必须先复制再转，禁止改 `/mnt/data/junbo/data`） |
| 动作窗口 | horizon=20，fps=20（1 秒） |
| 语义 token | latent_horizon=16，单码本 vocab=1024 |
| Policy | `n_obs_steps=2`，`n_action_steps=16`，task-token 条件；当前默认与 XLeRobot 状态见 `04_xlerobot_actioncodec_status_0901.md` |
| 动作维 | 跟数据走。RoboCasa PandaOmron 是 **12**，不是配置默认的 7 |
| 任务数 | `num_tasks >= 1`。单任务数据集把 task embedding 做成 1 类，task scalar 用 `max(1, num_tasks-1)` 避免除零 |

只保证源仓库的模型结构、forward/推理和 **loss 语义**；checkpoint 格式按 LeRobot 重做（`model.safetensors` + 分文件配置）。

## 2. 已经做完的事

### 2.1 代码接入

- **Tokenizer 核心**：`src/lerobot/actioncodec/`（Perceiver、RVQ、diffusion decoder、semantic/soft-DTW、独立 trainer）。
- **独立训练入口**：`lerobot-train-actioncodec-tokenizer`。默认 **diffusion decoder** + **soft-DTW**（`alignment_weight` → `weight_chunk_align`，默认 0.1，与源仓库 matched 配方一致）。hard-DTW 走独立的 `hard_alignment_weight`（默认 0）。artifact：`model.safetensors`、`model_config.json`、`action_stats.json`、`dataset_contract.json`。
- **Semantic policy**：`src/lerobot/policies/actioncodec/`，注册名 `actioncodec`，走原生 `lerobot-train`。processor 复用默认归一化（STATE MIN_MAX、ACTION MEAN_STD），并把 `task_index` 映射成 `task_uid`。当前 XLeRobot contract 见 `04_xlerobot_actioncodec_status_0901.md`。
- **Checkpoint**：保存时把冻结 tokenizer 打进 `pretrained_model/tokenizer`，配置写成相对路径；`from_pretrained` 能解析回来。
- **测试**：`tests/actioncodec`（26 passed / 1 skipped，无 robomimic）。新增文件按仓库 ruff 0.14.1 过了 check/format。

源实现是判定基准。移植时修过几处会改语义或直接训崩的偏差（稠密对比损失对角线 NaN、丢了 `_chunk_cost`、soft-DTW 标准化口径、hard-DTW 相似度缺因子、量化器熵归一化）。与源一致的设计（soft-DTW 用 `.mean(-1)`、扩散从零采样、只走第一个码本等）不要再当 bug 改。

### 2.2 数据与训练验证

- **本机 CPU**：CloseDrawer v2.1→v3，tokenizer 120 步 + policy 60 步，重载后 `predict_action_chunk` 为 `(B, 20, 12)` 且有限。语义 DTW 路径梯度有限。
- **DLC 单任务 smoke（4090）**：
  - tokenizer 80 步成功（`dlc1kwqe4qhqshsn`）。
  - policy 40 步成功（`dlc1a3enm2kewpxk`），加载上一份 tokenizer。
- **四任务合训**：CloseDrawer / StartCoffeeMachine / TurnOffMicrowave / TurnOffSinkFaucet。`action_dim=12`，三路 256 相机一致。官方 `aggregate_datasets` 合成 `outputs/datasets/robocasa_atomic4_v3`（432 ep / 56934 frames / **num_tasks=9**）。
- 旧配方（Perceiver + hard-DTW）：`dlc1611reyv25bro`，tokenizer/policy 各 10000 步。
- 新默认（**diffusion + soft-DTW 0.1**）：`dlcwe3hu26clf4dx`（`lerobot-ac-atomic4-pipe-day-20260830-162108`），同样 10000+10000、batch 8。产出 `outputs/dlc/atomic4_day_20260830_162108/`。[控制台](https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlcwe3hu26clf4dx&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942)

`MultiLeRobotDataset` 在 `make_dataset` 里仍是 `NotImplementedError`，所以合集走 `aggregate_datasets`，按任务字符串重编号，避免四个数据集各自从 0 撞号。

### 2.3 DLC 怎么跑

入口：`scripts/dlc/`。风格对齐 oat-exact（`aliyun pai-dlc create-job`）。

- 镜像：PAI `pytorch:2.7.0-gpu-py312-cu128-...`。仓库 `uv.lock` 钉的是 Linux **torch 2.11.0+cu128**，**不要改 lock 去迁就镜像**。
- 训练作业默认 **禁止 `uv sync`**：复用 NAS 上的 `.venv-dlc`（`--system-site-packages`，torch/numpy 用镜像）。提交机 `.venv`（torch 2.11+cu130）不能当 DLC 环境。
- 不要预建空的 `policy/` 输出目录，否则 `lerobot-train` 在 `resume=False` 时会 `FileExistsError`。
- 路径必须是 `/mnt/data/junbo/lerobot`（DLC 不挂 `/mnt/workspace`）。
- 单任务：`STAGE=test bash scripts/dlc/submit.sh`
- 四任务：`STAGE=day bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_atomic4_pipeline.sh`

稳态约 3–4 step/s（4090、batch=4、3 路 256、pyav）。瓶颈是 CPU 解视频，不是 GPU。

## 3. RoboCasa simulator 部署（2026-08-30 接入）

仓库已有通用入口，**没有** ActionCodec 专用部署脚本：

| 目标 | CLI | 现状 |
| --- | --- | --- |
| 仿真评测 | `lerobot-eval --env.type=robocasa` | 已接 task UID、两帧在线 history、action clip/统计、固定协议和 DLC 入口 |
| 真机 | `lerobot-rollout` | 给 SO-101 / Koch 等，不是 PandaOmron 厨房仿真 |
| 离线推理 | `predict_action_chunk` / `select_action` | 已用 atomic4 真实 checkpoint 重载并完成一次 CPU `select_action` |

### 3.1 已冻结的 rollout contract

1. `configs/robocasa/atomic4_actioncodec_task_indices.json` 保存 dataset 的实际语言到全局 `task_index` 映射：0/1/3/5/7；2/4/6/8 仍是训练 metadata 中的占位，policy `num_tasks=9` 不变。未知语言立即失败。
2. `ActionCodecPolicy.select_action` 在线维护 `[-1,0]` 两帧 history；首帧复制，执行缓存动作期间仍逐步更新，第 17 次调用以第 15/16 帧重新规划。`reset()` 同时清空 action/history。
3. checkpoint 的 unnormalizer 后、RoboCasa env 前 clamp 12D action 到 `[-1,1]`，把 clip element count/fraction 写入 `eval_info.json` 和 summary。
4. wrapper 不再在 terminal step 内部抢先 reset；Gymnasium vector env 管 autoreset。显式 evaluator seed 原样传给 worker，避免把 worker index 重复加到 42/43/... 上。
5. 固定任务为 CloseDrawer / StartCoffeeMachine / TurnOffMicrowave / TurnOffSinkFaucet；报告将 StartCoffeeMachine 标注为 CoffeePressButton。固定 target、seed 42、Lightwheel、EGL、async 4 env、官方 horizon 450/300/300/300、`h20/n16/nobs2/naction16`。新训产物默认 `temperature=1.0` + `top_k=10`（对齐 oat-exact formal eval）；旧 CNN checkpoint 仍是 `temperature=0` argmax。确定性评测可显式 `--policy.temperature=0`。

### 3.2 DLC 入口和产物

```bash
bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_robocasa_eval.sh
bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_atomic4_eval.sh
```

bootstrap 从已验证的 `.venv-dlc` 复制出独立 `.venv-robocasa-dlc`，固定 RoboCasa
`a07e365c958c4216cd6bbd5f30b47f09a65c6f00`、RoboSuite
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`、MuJoCo 3.3.1，并下载 Lightwheel 资产。
正式 eval 脚本严格串行执行 pretrain 2/task、target 2/task、target 20/task；前一段失败则不进入下一段。
每段保存 protocol、checkpoint 前后 hash、官方 eval_info、每任务视频和 summary。checkpoint 默认是
`outputs/dlc/atomic4_day_20260829_171919/policy/checkpoints/last/pretrained_model`。

本地验收：16 个 focused tests 通过；ruff、bash syntax、DLC dry-run 通过；真实 checkpoint 在 CPU
严格重载，三路 256 image + 16D state + task 7 得到有限 `(1,12)` action。首次 simulator bootstrap
作业为 `dlcsi0r2ifp5x1qf`，最终状态和后续 eval JobId 在本节完成后补记。

### 3.3 真机部署（更后）

当前数据是 RoboCasa 仿真，和真机不是同一套本体/相机/动作空间。真机要另做：标定与相机映射、确认 12 维动作含义、控制频率与 horizon=20 对齐、急停。在仿真 eval 稳定之前不要上真机。

### 3.4 训练侧未关账（不挡部署原型）

- 四任务 10k+10k 是否收敛、码本是否塌缩（短 smoke 里 `unique_codes` 会掉到个位数）。
- `STAGE=full`（50k/100k）未跑。
- 源仓库 soft-DTW 可选 `softdtw-cuda-torch` 加速（`dtw_backend=cuda`）。本仓库已把 **Torch 回退改成 pair 维向量化**（`action_tokenizer` 的 DP 思路）：`chunk_soft_dtw_targets` 对 `[P, T, T]` 做一次格子循环，不再对每对跑标量 Python DP。`auto`/`torch` 走这条路径；`cuda` 仍保留接口但未接 extension，会显式报错。GPU 7 微基准（4090）：B=8 标量 1579ms → 向量化 70ms；B=128（8128 对）仍约 70ms。XLeRobot tokenizer 已于 2026-08-30 14:30 用该实现在 GPU 6 重启（旧 run 留在 `tokenizer_interrupted_pre_fast_dtw`）；约 0.15s/step。
- tokenizer 训练下一步瓶颈曾是三路视频解码。2026-08-31：提交机 `.venv` 已从 Anaconda 3.12.7 换成 uv-managed CPython 3.12.13（`.python-version` 指向 managed 解释器，已 gitignore）。`import torchcodec` 成功，默认 backend 为 `torchcodec`。单路 3 帧 decode ~54ms，`dataset[0]` 三路各 1 帧 ~16ms。DLC `.venv-dlc` 仍走镜像 Python，不受此次提交机 venv 重建影响。
- 独立 tokenizer 默认已对齐 `../actioncodec` matched_h20（`model_dim=512`、8 轮 shared cross-attn、`vq_beta=1.0`、batch 512、stride 4、cosine warmup、CUDA AMP）。tokenizer 读数据时 `decode_videos=False`。旧 256-d XLeRobot tokenizer checkpoint 作废；当前 12D no-head 训练状态见 `agent_docs/04_xlerobot_actioncodec_status_0901.md`。

## 4. 建议顺序

1. 等 `dlcsi0r2ifp5x1qf` 完成固定 simulator 环境和 EGL smoke。
2. 提交三阶段 eval；只在 pretrain/target 2-task smoke 都通过后接受 target 20/task。
3. 根据每任务成功率、视频与 action clipping fraction 决定是否加长训练。

**一句话：** 模型、loss、两段训练与原生 `lerobot_eval` adapter 已进入 LeRobot；diffusion + soft-DTW（`alignment_weight=0.1`）DLC 作业已能正常训练，本批代码准备合入。当前只差 DLC simulator smoke 和 target 20/task 的实测结果，不再缺 task UID 或在线 history 接口。

## 5. XLeRobot 单任务（2026-08-30）

- tokenizer 默认 soft-DTW 比较 **归一化 action chunk**，不再构造 `delta_state`；独立 trainer 把 `loss/recon/vq/align/unique_codes_batch/codebook_occupied_*` 写到 `{output_dir}/tb`。`alignment_loss_config` 带上 `chunk_align_dtw_backend=auto`、`chunk_align_pair_batch_size=8192`、`chunk_align_max_candidate_pairs=1024`。架构默认对齐 matched_h20。
- policy 允许 `num_tasks=1`。accelerate 多进程下 `FileExistsError` 只由 rank 0 检查，避免 TensorBoard `mkdir` 和 peer `validate()` 抢目录。
- 数据用原始 `my_dataset_merged`（14D、30 FPS、1 task），不裁剪。
- 正式 tokenizer（2026-08-30 14:30–14:53，GPU 6，快 DTW ~0.15s/step）：`outputs/xlerobot_actioncodec/tokenizer`。step 500 `unique_codes=56`，收尾约 18–32（不是长期为 1）。旧慢 run 在 `tokenizer_interrupted_pre_fast_dtw`。
- policy（GPU 6+7，`eval_split=0.1`，pyav）写入 `outputs/xlerobot_actioncodec/policy`。该 run 是 **small_cnn + MEAN_STD state + temperature=0**，与当前 oat-exact 默认不兼容；兼容性边界见 `agent_docs/04_xlerobot_actioncodec_status_0901.md`。
