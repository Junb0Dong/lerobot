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
