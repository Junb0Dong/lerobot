# Policy 迁移、LeRobotDataset 与 DataLoader 指南

> 复核日期：2026-08-28。本文以当前仓库源码（LeRobotDataset v3.0、项目版本 0.6.2）为准。

## 1. 先区分三种评估

- **离线 held-out loss**：训练数据按 task 做 episode 级切分，使用 `--dataset.eval_split` 和 `--eval_steps`。这只衡量数据分布上的损失，不运行环境。
- **仿真 benchmark**：`lerobot-eval` 创建注册的 Gymnasium 环境，调用 policy 的 `select_action`，统计 reward/success。policy 的 observation/action contract 必须和环境一致。
- **真实机器人**：使用 `lerobot-rollout`（或 `lerobot-record --policy.path`）连接硬件，经过机器人和 policy processor 后执行动作。需要单独确认相机、关节顺序、标定和安全措施。

## 2. LeRobotDataset v3.0 的目录和必需元数据

典型数据集目录如下：

```text
dataset-root/
├── data/chunk-000/file-000.parquet       # 每帧的数值、索引、任务等
├── meta/
│   ├── info.json                          # schema、fps、路径模板、计数
│   ├── stats.json                         # 归一化统计量
│   ├── tasks.parquet                      # task_index -> 文本任务
│   └── episodes/chunk-000/file-000.parquet# 每 episode 的边界和文件位置
└── videos/<feature-key>/chunk-000/file-000.mp4 # 可选，视频特征
```

`meta/info.json` 至少要能描述：

- `codebase_version`（当前为 `v3.0`）、正数 `fps`；
- `features`：每个特征的 `dtype`、`shape`、`names`，视觉特征还应有 `info`（深度图使用 `info.is_depth_map=true`，并可声明 `depth_unit`）；
- `data_path` / `video_path` 模板，以及 `total_episodes`、`total_frames`、`total_tasks`；
- 可选 `robot_type`、`splits`、语言工具定义。

数据帧由 Parquet 中的 schema 决定，不能只放一个自定义 pickle。文件中必须包含框架自动字段：
`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`（分别为 float32/int64 标量特征）。读取时 `LeRobotDataset.__getitem__` 会再添加 `task` 字符串。

推荐的业务特征命名：

| 用途 | key 示例 | dtype / shape |
| --- | --- | --- |
| 关节状态 | `observation.state` | `float32`, `(D,)`，`names` 为固定关节顺序 |
| RGB 相机 | `observation.images.front` | `image` 或 `video`, `(H,W,3)`，`names=[height,width,channels]` |
| 深度相机 | `observation.images.depth` | `image` 或 `video`, `(H,W,1)`，`info.is_depth_map=true` |
| 动作标签 | `action` | `float32`, `(A,)`，`names` 为动作维度顺序 |
| 环境状态（仿真） | `observation.environment_state` | 数值向量，供环境 policy 使用 |

图像在 feature metadata 中以 `(H,W,C)` 描述，但转换为 policy feature 后会变成 channel-first `(C,H,W)`。特征名不能包含 `/`；同一数据集所有 episode 的 schema、FPS、相机 key 和动作维度必须一致。

`stats.json` 应为每个需要归一化的 feature 提供 `mean`、`std`（以及 policy/normalizer 所需的 `min`、`max` 或 `q01`、`q99` 等）。录制或转换后优先使用仓库的统计量计算工具，不要手写一套与 checkpoint 不一致的归一化。数据集 v2.1 需要先用 `convert_dataset_v21_to_v30.py` 转换。

## 3. 一个样本如何进入 policy

训练入口 `lerobot-train` 的实际链路是：

```text
DatasetConfig
  -> LeRobotDatasetMetadata
  -> 根据 policy config 的 delta indices 生成 delta_timestamps
  -> LeRobotDataset.__getitem__
       (Parquet 数值 + episode 边界 + 视频按 timestamp 解码 + image transform)
  -> PyTorch DataLoader
  -> uint8 图像转 float32 / 255、rename_map
  -> policy preprocessor（device、normalizer、相对动作等）
  -> policy(batch) -> (loss, logging_dict)
```

非 streaming 数据使用 `EpisodeAwareSampler`，按 episode 打乱，避免时间窗口跨 episode；`delta_timestamps` 超出 episode 边界时会复制边界帧，并附加 `<key>_is_pad`。视频解码发生在 DataLoader worker 内，多相机并行解码。默认 DataLoader 参数来自 `TrainPipelineConfig`：`batch_size=8`、`num_workers=4`、`prefetch_factor=4`、`persistent_workers=true`、multiprocessing context=`spawn`；GPU 时启用 `pin_memory`。可用 `--dataset.streaming=true` 改为流式数据集，但 streaming 不支持 `eval_split`。

批次通常是一个字典，key 与 dataset features 相同；数值字段形状为 `[B,D]`，单帧图像为 `[B,C,H,W]`，带时间窗口时为 `[B,T,...]`。语言字段存在时使用语言专用 collate，否则使用 PyTorch 默认 collate。

## 4. 把已有 policy 迁移进来

### 4.1 先对齐数据 contract

1. 用 `LeRobotDataset(repo_id, root=...)` 读取数据，检查 `dataset.meta.features`、`dataset.meta.fps`、`dataset.meta.camera_keys`、`dataset.meta.stats`。
2. 确认 policy 输入只引用 dataset 中存在的 key。已有 checkpoint 的 key 不同可以使用 `--rename_map='{"observation.images.front":"observation.images.camera1"}'`；rename 会同时作用于 features、batch 和 stats。
3. 确认动作维度和 `action.names` 顺序。动作的单位、范围（位置/速度/增量）必须和训练数据及部署机器人完全相同。
4. 根据 policy 所需历史帧/动作 chunk 实现 config 的 `observation_delta_indices`、`action_delta_indices`；factory 会把 index 除以 dataset FPS 生成时间戳。

### 4.2 实现 policy config

新 policy 至少需要一个继承 `PreTrainedConfig` 的 dataclass，并注册：

```python
@PreTrainedConfig.register_subclass("my_policy")
class MyPolicyConfig(PreTrainedConfig):
    @property
    def observation_delta_indices(self): return [0]

    @property
    def action_delta_indices(self): return list(range(16))

    @property
    def reward_delta_indices(self): return None

    def get_optimizer_preset(self): ...
    def get_scheduler_preset(self): ...
    def validate_features(self): ...
```

`input_features` / `output_features` 可留空，让 `make_policy` 从 dataset metadata 自动推导；需要固定 schema 时再显式填写。`validate_features()` 应拒绝缺失相机、错误 state/action shape，而不是等到 forward 才出现难懂的 shape error。

### 4.3 实现 policy model

模型继承 `PreTrainedPolicy`，并定义 `config_class`、`name`。训练和推理的最低接口是：

- `get_optim_params()`：返回 policy-specific optimizer 参数；
- `forward(batch) -> (loss, logging_dict)`：训练 loss 必须是可反传 Tensor，日志值应为 Python 标量；若支持 sample weighting，再接受 `reduction="none"`；
- `reset()`：环境 reset 时清理 observation/action cache；
- `predict_action_chunk(batch)`：需要 action chunk 时输出形如 `[B, chunk, action_dim]`；
- `select_action(batch)`：推理时返回当前一步动作，负责历史缓存和 chunk 队列。

保存/加载使用 `save_pretrained` / `from_pretrained`，配置和权重放在 checkpoint 的 `pretrained_model/`。不要只发布裸权重：训练生成的 `policy_preprocessor.json`、`policy_postprocessor.json` 及其 stats 也是推理 contract 的一部分。

### 4.4 实现 processor

为新 policy 提供 `make_<policy>_pre_post_processors`（或复用通用 factory），至少覆盖：

- dataset batch -> policy 输入的 key 映射、device、图像布局和 dtype；
- state/action 的 normalization/unnormalization；
- 如使用相对动作，配置 `RelativeActionsProcessorStep` 与对应的 absolute action postprocessor；
- policy 输出 -> 机器人/环境动作的 key、shape、范围。

训练时 processor 使用 dataset stats；从 checkpoint 加载时优先使用 checkpoint 中保存的 processor 和 stats。重新计算 stats 或手工改 normalizer 会造成训练/部署不一致。

### 4.5 注册和最小验证

把 config/model 放在 `src/lerobot/policies/my_policy/`，在 factory 的 lazy import 约定下提供 `configuration_my_policy.py` 和 `modeling_my_policy.py`。也可以做成第三方插件包，包名使用 `lerobot_policy_` 前缀并通过 registry 注册。

先做一个 batch 级 smoke test：

```python
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_policy

ds = LeRobotDataset("user/my_dataset", root="/path/to/dataset")
cfg = MyPolicyConfig(device="cpu")
policy = make_policy(cfg=cfg, ds_meta=ds.meta)
batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in ds[0].items()}
loss, logs = policy(batch)
assert loss.ndim == 0 and loss.isfinite()
```

随后运行 `uv run pytest tests/policies tests/datasets -svv`，再用一个很小的 `--steps=2 --num_workers=0 --batch_size=1` 训练检查反向传播、保存和 reload。确认无误后再提高 worker 数和 batch size。

## 5. 训练、离线评估和部署命令模板

从头训练（feature 从数据集推导）：

```bash
uv run lerobot-train \
  --dataset.repo_id=<user>/<dataset> \
  --policy.type=my_policy \
  --policy.device=cuda \
  --batch_size=8 --num_workers=4 \
  --output_dir=outputs/train/my_policy \
  --job_name=my_policy
```

带 held-out loss：

```bash
uv run lerobot-train \
  --dataset.repo_id=<user>/<dataset> \
  --dataset.eval_split=0.1 --eval_steps=1000 \
  --policy.type=my_policy
```

从已有 checkpoint 微调时用 `--policy.path=<checkpoint-or-hub-repo>`；恢复中断训练则用 checkpoint 的 `train_config.json` 配合 `--resume=true`。迁移旧 checkpoint 且 key 名不同，可以额外传 `--rename_map`，但它要求已有 pretrained policy。

仿真评估：

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/my_policy/checkpoints/last/pretrained_model \
  --env.type=<matching_env> \
  --eval.n_episodes=50 --policy.device=cuda
```

真实机器人部署（先短时、低速、空载并准备急停）：

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/my_policy/checkpoints/last/pretrained_model \
  --robot.type=<robot_type> --robot.port=<port> \
  --task="<training-compatible task>" --duration=30
```

## 6. 迁移前检查清单

- [ ] 所有 episode 的 FPS、Parquet schema、相机 key、state/action names 一致。
- [ ] `meta/info.json`、`stats.json`、`tasks.parquet`、`meta/episodes/` 和对应 data/video 文件齐全。
- [ ] `dataset[0]` 能读出 tensor；视频能在 DataLoader worker 中解码；没有跨 episode 的时间窗口。
- [ ] policy config 的 delta indices 与数据 FPS 对齐，输入/输出 feature shape 通过 `validate_features()`。
- [ ] 训练和部署使用同一套 pre/postprocessor、normalization stats、相机分辨率和动作单位。
- [ ] 已分别验证：两步训练、checkpoint reload、仿真 `lerobot-eval`（若有环境）、真机短 rollout。

## 7. 当前决策和后续动作

- 本文只记录当前仓库已实现的 v3.0 map-style/streaming dataloader contract，不把旧版 v2 格式当作原生输入。
- 下一步若要真正接入某个 policy，需要拿到该 policy 的输入输出 schema、历史帧/动作窗口、动作单位和目标环境/机器人，再按第 4 节实现 config、model、processor 与 smoke tests。

## 8. 熟悉项目与迁移的推荐路线

不要从硬件或复杂 VLA policy 开始。先以 ACT 为参考实现，按以下顺序阅读和实验：

1. 先看 `scripts/lerobot_train.py`，定位 dataset、policy、processor、optimizer 和训练循环的创建位置。
2. 再看 `policies/factory.py`、`policies/pretrained.py`，理解 feature 推导、动态注册、checkpoint 加载和 policy 最低接口。
3. 接着看 `datasets/dataset_metadata.py`、`datasets/lerobot_dataset.py`、`datasets/sampler.py`，确认一个样本的 key、shape、时间窗口和 episode 边界。
4. 最后看 `processor/pipeline.py`、`processor/normalize_processor.py` 和 `policies/act/`，分别理解通用数据变换、归一化以及一个完整 policy 的 config/model/processor 配套关系。

迁移前应先写清自己的 policy contract：输入 key 和 shape、图像布局与 dtype、历史帧需求、动作 chunk/horizon、动作顺序和单位、归一化方式、训练 loss、推理时是否有缓存。实现顺序建议为 `configuration_<name>.py` → `modeling_<name>.py` → `processor_<name>.py` → batch smoke test → 两步训练和 checkpoint reload；确认离线链路后再做仿真或真机 rollout。

## 9. ActionCodec 语义 tokenizer/policy（2026-08-28）

训练里程碑与部署缺口的总览见 `03_actioncodec_lerobot_milestone_0830.md`。本节留 loss 修复、DLC 踩坑和吞吐细节。

- 已冻结 contract：LeRobotDataset v3、action horizon=20、latent horizon=16、单码本 vocab=1024；policy 使用 n_obs_steps=2、n_action_steps=16 和 task token。
- 新增独立 `lerobot-train-actioncodec-tokenizer` 入口，tokenizer artifact 使用 `model.safetensors`、配置、action stats 和 dataset contract 分文件保存。
- semantic policy 通过 `lerobot-train` 的原生 policy factory 接入；processor 复用默认 normalize/unnormalize pipeline，并将 `task_index` 映射为 `task_uid`。
- 2026-08-28 复核：已补齐源实现对应的 Perceiver cross/self-attention 配置、x0 diffusion decoder/采样、RVQ dead-code refresh 生命周期、完整 artifact 参数及 VL/InfoNCE 可选辅助损失。

### 9.1 与源实现的 loss parity 修复（2026-08-28）

逐函数比对 `/mnt/data/junbo/actioncodec-oat-exact-policy` 后，修复了四处移植引入的偏差。源实现是判定基准，凡与源一致的写法一律保留。

- `semantic_contrastive_loss` 曾用稠密距离矩阵 `(emb[:,None]-emb[None]).pow(2).sum(-1).sqrt()`，对角线 `sqrt(0)` 在反传时产生 `0/0`，使整个 `embeddings.grad` 变 NaN；只要 `weight_align` 或 `weight_chunk_align` 大于 0，tokenizer 训练即不可用。已改回源实现的稀疏 `_masked_pair_squared_distances`，前向值逐值不变，仅修复梯度。
- `soft_dtw.py` 丢失了源实现的 `_chunk_cost`（`[M, T, D]`，在时间和特征两轴取均值），导致 `trajectory_soft_dtw_alignments` 把 3 维张量传给只收 2 维的 `_step_cost` 而必抛 `ValueError`。已补回并让两个公共函数按输入秩分派。
- `chunk_soft_dtw_targets` 的标准化误用全 batch 统计量，源实现的 `_normalize_delta_state` 只用运动中的 chunk；静止样本会压低 std 并改变正样本选取。已修回。
- `chunk_hard_dtw_targets` 的相似度缺因子：矩阵处漏 `temperature`、行内负样本挖掘处漏 `scale`。源实现两处均为 `exp(-d/scale/temperature)` 且挖掘复用矩阵。已统一。
- `ResidualVectorQuantizer` 在 forward 内重复内联 `max(math.log(codebook_size), 1e-6)`，绕过了 `__init__` 里带 `max(codebook_size, 2)` 保护的 `_entropy_norm`。已改为使用 `_entropy_norm`。

与源实现一致、**不应**改动的已知限制（复核时勿再"修复"）：soft-DTW 步代价用 `.mean(-1)` 而非 `.sum(-1)`、alignment 归一化到和为 1、扩散采样从零张量而非高斯噪声起步、`tokenize`/`detokenize` 只走第一个码本、以及默认 embodiment config 写死 `freq=20/duration=1` 使 perceiver 路径固定 20 步（由 `ActionCodecTokenizerConfig.validate()` 强制 `horizon=20` 兜住）。

### 9.2 代码质量与验证状态（2026-08-28）

- `src/lerobot/actioncodec/**` 是新建顶层包，不在 `pyproject.toml` 的 `D` 豁免列表内，已按仓库"终点 100% 覆盖"的方向补齐全部 Google 风格 docstring，而非新增豁免。
- `perceiver.py` 中被文件末尾别名遮蔽的两个简化版 `ActionPerceiverEncoder/Decoder` 及其专用 `_Block`/`_embedding` 已整体删除，`Exact*` 类直接占用公开名字。
- 验证（ruff 0.14.1，与 `.pre-commit-config.yaml` 同版本）：新增文件 `ruff check` 与 `ruff format --check` 全绿；`pytest tests/actioncodec` 8 passed；`lerobot-train` 两步训练 + checkpoint 保存 + 重载 + `predict_action_chunk`/`select_action` 全部通过，重载时 tokenizer 相对路径能正确解析；`lerobot-train-actioncodec-tokenizer --alignment_weight=1.0` 三步训练产出的 128 个权重张量全部有限。
- 仍未做：soft-DTW CUDA backend parity、与源实现的完整训练日志字段对齐、仿真与真机 rollout。

### 9.4 DLC 提交（2026-08-28）

在真实 robocasa CloseDrawer 上测试当前 tokenizer + semantic policy 的 PAI DLC 脚本，风格对齐 oat-exact（`aliyun pai-dlc create-job`、`scripts/dlc/{env,cmd,submit}.sh`）。

**镜像 vs 仓库依赖（不要擅自降 lock）：**

- 仓库：`requires-python >= 3.12`，`pyproject.toml` 写 `torch>=2.7,<2.12.0`。
- `uv.lock` 的 Linux 实际钉死 **torch 2.11.0+cu128**（以及 `nvidia-nvshmem-cu12` 等 CUDA wheel）。
- DLC 镜像（用户指定 PAI 官方 URI）：`pytorch:2.7.0-gpu-py312-cu128-ubuntu24.04-ngc25.03-deep-ep-4d7302ff-1770029761`，自带 Python 3.12 + torch 2.7.0 + cu128。
- 不以 lock 覆盖镜像 torch。默认复用 NAS 上已有的 `.venv-dlc`（`import lerobot, torch`、torch 2.7.x、且 `torch.from_numpy` 可用则跳过 `uv sync`）。没有可用环境时才创建 `.venv-dlc --system-site-packages`，`uv sync --frozen --inexact` 时 `--no-install-package torch`（及 torchvision / torchcodec / triton / numpy / nvidia-* / cuda-*）。训练前断言 `torch.__version__` 以 `2.7` 开头且 `from_numpy` 可用。未改 `uv.lock` / `pyproject.toml`。`FORCE_UV_SYNC=1` 强制重装；不要 uv sync 提交机 `.venv`。

- 入口：`scripts/dlc/`。默认 `STAGE=test` 两段式：`bash scripts/dlc/submit.sh` 或 `scripts/dlc/cmd_robocasa_closedrawer_pipeline.sh`。
- 数据：原始路径 `/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot` 是 v2.1。官方转换脚本只支持原地，DLC/本地脚本会先 `cp -a` 到 `<repo>/outputs/datasets/robocasa_closedrawer_v3` 再转。不要改 `/mnt/data/junbo/data`，也不要抢 `/tmp/ac_data`。
- CloseDrawer 与默认 contract 的差异：`action_dim=12`（配置默认 7）。脚本从 `meta/info.json` 读取，tokenizer 传 `--action_dim=12`，policy 传 `--policy.action_dim=12 --policy.push_to_hub=false`。`num_tasks` 至少为 2。
- 本地转换：`bash scripts/dlc/convert_robocasa_v21_to_v30.sh`（已把 CloseDrawer 拷到 `outputs/datasets/robocasa_closedrawer_v3`，原始 v2.1 未改）。正式训练把 `STAGE=full`。wandb 默认关闭。
- 本机 `127.0.0.1:7890` 代理未开，需 `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy` 后再 `aliyun pai-dlc create-job`。`submit.sh` 已会自动 unset。

失败作业 `dlcl7kyyyjuv9e03`（`lerobot-ac-closedrawer-pipe-test-20260828-223140`）：Status=Failed，ExitCode=1。训练 CLI / `action_dim` / 转换都还没跑到。卡在容器 `uv sync --frozen --extra dataset --extra training`：`uv.lock` 的 Linux torch 是 `2.11.0+cu128`，要从 PyPI 拉 `nvidia-nvshmem-cu12==3.4.5`（约 139MB）等 CUDA wheel。默认 `UV_HTTP_TIMEOUT=30s`，日志：

```text
× Failed to download `nvidia-nvshmem-cu12==3.4.5`
  ├─▶ Failed to extract archive:
  │   nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
  ├─▶ I/O operation failed during extraction
  ╰─▶ Failed to download distribution due to network timeout. Try increasing
      UV_HTTP_TIMEOUT (current value: 30s).
hint: `nvidia-nvshmem-cu12` (v3.4.5) was included because `lerobot` (v0.6.2) depends on `torch` (v2.11.0+cu128)
```

路径修复（不改 loss）：`scripts/dlc/paths.sh`：提交机 `/mnt/workspace` 与 `/mnt/data` 是同一份 NAS；DLC 只挂 `/mnt/data/`。`pwd` 若解析成 `/mnt/workspace/...` 则改写成 `/mnt/data/...`。误提的 `dlcg86brextmhn5f` 因 `cd /mnt/workspace/junbo/lerobot` 在容器里不存在，25 秒即失败。

作业 `dlc111bvgqhkrmic`（`lerobot-ac-closedrawer-pipe-test-20260828-231517`）已用 PAI 2.7.0 镜像，但 bootstrap 仍按 lock 拉 torch 2.11（日志：`Downloading torch (782.3MiB)` / `nvidia-cudnn-cu12 (627.4MiB)`）。尚未进入训练，已 stop。改成跳过镜像 torch 后再提 `STAGE=test`：JobId `dlc1myilu2h2pxn3`（`lerobot-ac-closedrawer-pipe-test-20260828-232507`）。UserCommand 为 `cd /mnt/data/junbo/lerobot`。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1myilu2h2pxn3&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942

**`dlc1myilu2h2pxn3` 失败原因（2026-08-29）：** Status=Failed，ExitCode=1，跑了约 3 小时。`Prepared 73 packages in 175m 04s`（`UV_LINK_MODE=copy` 往 NAS 拷包），装入 `numpy==2.2.6`。镜像 torch `2.7.0a0+7c8ec84dab.nv25.03` 按 NumPy 1.x 编译。日志：

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
...
Failed to initialize NumPy
...
  File ".../video_utils.py", line 176, in decode_video_frames_pyav
    loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
RuntimeError: Numpy is not available
```

tokenizer 在读第一批视频时崩掉。不是 loss 问题。未改 `uv.lock` / `pyproject.toml` / `/mnt/data/junbo/data`。

**环境优化（2026-08-29）：** 探测改为 `import lerobot, torch` + torch 2.7.x + `torch.from_numpy`。venv 里的 numpy 2.x 会删掉，改用镜像 1.x。可复用则零网络（不装 uv、不 sync、不 apt libglx0）。sync 跳过 numpy；`UV_LINK_MODE` 默认 hardlink（copy 在 NAS 上 175 分钟）；优先 `--offline` 走 `.cache/uv-dlc`；uv 二进制可放 `.cache/uv-dlc/bin/uv`。新增 `scripts/dlc/cmd_bootstrap_venv.sh`。开关：`FORCE_UV_SYNC` / `SKIP_UV_SYNC` / `UV_OFFLINE` / `LEROBOT_VENV`。

重提 `STAGE=test`：JobId `dlc1kwqe4qhqshsn`（`lerobot-ac-closedrawer-pipe-test-20260829-103453`）。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1kwqe4qhqshsn&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942 。启动后约 15 秒已 reuse：删掉 venv numpy 2.2.6，改用镜像 numpy 1.26.4，日志 `reusing /mnt/data/junbo/lerobot/.venv-dlc (no uv install, no uv sync, no PyPI)`，跳过 apt libglx0，已进入 tokenizer（`action_dim=12 steps=80`）。没有 `Downloading` / `Prepared N packages`。

**tokenizer「阶段的问题」定性（2026-08-29）：** 不要把环境和训练公式混在一起。

- `dlc1myilu2h2pxn3` 挂在读视频：`RuntimeError: Numpy is not available`（venv numpy 2.2.6 vs 镜像 torch 的 NumPy 1.x ABI）。这是环境，不是 tokenizer 模型。已用镜像 numpy 1.26.4 绕过；不要为了 numpy>=2 去改 `uv.lock`。
- `dlc1kwqe4qhqshsn` tokenizer **跑完了**：step 0–79 loss 有限（例如 step 0 `loss: 2.0372 recon: 0.6753`，step 79 `loss: 2.9692 recon: 0.5550`），日志 `TOKENIZER_OK .../tokenizer`，产物 `model.safetensors` / `model_config.json` / `action_stats.json` / `dataset_contract.json`（`action_dim=12`）。无 NaN、shape、CLI、OOM 异常。`unique_codes` 在 80 步 smoke 里落到 1–7 是短跑码本利用率低，不是实现 bug；未改 loss 数学。
- 该作业失败在 policy：`lerobot_dlc_train_policy` 预建了空的 `policy/`，`lerobot-train` 在 `resume=False` 时抛 `FileExistsError: Output directory .../policy already exists`。tokenizer CLI 没有这项检查（保存时 `mkdir(..., exist_ok=True)`），预建 tokenizer 目录无同类风险。

**DLC 脚本修复（只改 `scripts/dlc/`，未改 `uv.lock` / pyproject torch / `train.py` 防覆盖逻辑）：**

- policy 只 `mkdir -p "$(dirname output_dir)"` 和 `RUN_ROOT/logs`，不再预建空的 `policy/`。
- 训练作业默认 `SKIP_UV_SYNC=1`：探测 `.venv-dlc` 能 `import lerobot, torch` 且 torch 2.7 则直接训；失败就失败，GPU 上不装包、不 curl uv、不访问 PyPI。`FORCE_UV_SYNC=1` 只给 `cmd_bootstrap_venv.sh`（该脚本把 SKIP 默认成 0）。
- 已删上次空的 `policy/`；tokenizer 产物保留。v3 数据和 tokenizer 都在，因此只重提 policy（`STAGE=test` + `TOKENIZER_PATH`），不重跑 tokenizer。
- 新作业 JobId `dlc1a3enm2kewpxk`（`lerobot-ac-closedrawer-pol-test-20260829-105950`）**Succeeded**。`SKIP_UV_SYNC=1`，reuse `.venv-dlc`（镜像 torch 2.7 / numpy 1.26.4），无 `installing uv` / `Downloading` / `Prepared N packages`。policy 40 步跑通，`POLICY_OK`，无 FileExistsError。`TOKENIZER_PATH=.../closedrawer_test_20260829_103453/tokenizer`，`POLICY_OUTPUT_DIR=.../closedrawer_test_20260829_105950/policy`。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1a3enm2kewpxk&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942

**DLC STAGE=test 吞吐（2026-08-29，作业日志原文，未猜）：** 「3–4 steps/s」对应 **policy** tqdm 的 `step/s`（`dlc1a3enm2kewpxk`），不是 `smp/s`。GPU `NVIDIA GeForce RTX 4090`，`ResourceConfig` CPU 16 / 64Gi / GPU 1（`GPUType` 空）。test **没有缩小模型**：tokenizer `model_dim=256` layers=3；policy `embed_dim=256` `n_layers=4` `image_size=128`（数据仍是 3 路 256 视频 × `n_obs_steps=2`），可学习 5.36M / 总计 14.4M（冻结 tokenizer）。batch=4、workers=2、`video_backend=pyav`、`device=cuda`、wandb off。

- policy 40 步：Start `11:01:23` → step 40 `11:01:43`（20s，含首步 8.99s warmup）。稳态 `step_s` 0.279/0.296/0.289，`data_s` 0.265–0.282，`updt_s` 0.014，`smp/s` 14，tqdm 3.04–3.85 `step/s`。瓶颈是 pyav CPU 解码，不是 GPU。
- tokenizer 80 步（`dlc1kwqe4qhqshsn`）：step 0 `10:36:29` → step 79 `10:36:51`（22s，约 3.6 step/s）。tokenizer CLI 不打印 `step_s`。
- 本地 CPU（`/tmp/pol_closedrawer.log`）：稳态约 1.5–1.7 `step/s`，`updt_s` 0.59–0.71（算力瓶颈）。oat-exact 同配额无对照吞吐日志。
- `STAGE=full` 只改步数/batch/workers（50k/100k、batch 8、workers 4），**不改模型宽度**。无 full 实测 `step_s`。

### 9.3 真实 robocasa CloseDrawer 验证（2026-08-28）

CPU only、2 核。`torchcodec` 缺 `libavutil` 回退 `pyav`，属环境问题。原始数据目录未改。

- **数据**：`/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot`，v2.1，110 ep / 15670 frames，`action_dim=12`，`state_dim=16`，fps=20，3 路 256 视频。
- **转换**：脚本原地写盘，必须先 `cp -r` 再跑 `convert_dataset_v21_to_v30.py --push-to-hub=false`。副本 `/tmp/ac_data/CloseDrawer`，约 13 秒，63M→39M，`codebase_version=v3.0`。
- **tokenizer**：`--action_dim=12 --alignment_weight=0.5 --steps=120 --batch_size=8 --device=cpu`。recon 0.72→0.42，align 项非零，128 个权重全部有限。step 100 recon 冲到 3.72 是 `dead_code_threshold=100` 的死码刷新，随后恢复。语义 DTW 路径 109 个参数梯度全部有限（9.1 的 NaN 修复在真实数据上得到验证）。产物 `outputs/ac_tok_closedrawer/`。
- **policy**：`--policy.type=actioncodec --policy.push_to_hub=false --policy.action_dim=12 --policy.tokenizer_path=... --steps=60`。loss 39.18→6.04。checkpoint 把 tokenizer 打成相对路径 `tokenizer`；重载后 `predict_action_chunk` 形状 `(3, 20, 12)` 且有限。
- **TurnOffMicrowave**：tokenizer 40 步跑通，recon 1.58→0.67。该任务 `total_tasks=1`，而 `ActionCodecConfig.validate_features()` 要求 `num_tasks >= 2`，单独训 policy 会被契约挡住——契约设计，不是 bug。
- **代码**：仅 `src/lerobot/actioncodec/trainer.py` 增加 `log_freq` 与 step 日志（loss/recon/vq/align/unique_codes），未改 loss 数学。`pytest tests/actioncodec` 仍 8 passed。
- **未覆盖**：diffusion decoder、CLIP 辅助损失、长时间收敛、仿真/真机。

### 9.5 四任务 robocasa atomic 合训（2026-08-29）

四个 v2.1 任务都是 PandaOmron、`action_dim=12`、`state_dim=16`、fps=20、同一组三路 256 相机。相机和动作维度无不一致，没有丢任务。

| 任务 | episodes | frames | info.total_tasks | 实际语言指令 |
| --- | ---: | ---: | ---: | --- |
| CloseDrawer | 110 | 15670 | 2 | Close the right/left drawer |
| StartCoffeeMachine | 108 | 13722 | 1 | Press the button on the coffee machine... |
| TurnOffMicrowave | 108 | 15233 | 1 | Press the stop button on the microwave |
| TurnOffSinkFaucet | 106 | 12309 | 1 | Turn off the sink faucet |

`make_dataset` 的 `MultiLeRobotDataset` 当前是 `NotImplementedError`，`DatasetConfig.repo_id` 只收 str。合集用官方 `aggregate_datasets`（`lerobot-edit-dataset --operation.type merge` 的底层）。按 task **字符串** 重编号，避免各自从 0 编号撞号。

合并产物 `outputs/datasets/robocasa_atomic4_v3`：432 ep / 56934 frames / `num_tasks=9`。帧上实际出现的 `task_index` 是 0/1/3/5/7；2/4/6/8 是 `tasks.jsonl` 里未使用的任务名占位。policy 必须用 9 而不是 5，否则 max id=7 会越界。原始 `/mnt/data/junbo/data` 仍是 v2.1。

DLC 入口（不改 CloseDrawer 单任务脚本）：`scripts/dlc/cmd_robocasa_atomic4_{convert,tokenizer,policy,pipeline}.sh`。默认 `STAGE=day`：tokenizer **10000**、policy **10000**、batch 8、workers 4、`alignment_weight=1.0`。4090+pyav 按 CloseDrawer test 的 ~3 step/s（batch=4）外推，batch=8 约 2 step/s，10k+10k 约 2–4 小时，不必降到 3k。

本地已转换+merge。GPU 作业 `SKIP_UV_SYNC=1`，reuse `.venv-dlc`，不预建空 `policy/`。JobId `dlc1611reyv25bro`（`lerobot-ac-atomic4-pipe-day-20260829-171919`）：tokenizer 10000 + policy 10000，batch 8，workers 4，`alignment_weight=1.0`。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1611reyv25bro&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942
