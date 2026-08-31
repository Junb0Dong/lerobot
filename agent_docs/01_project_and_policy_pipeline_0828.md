# LeRobot 项目与 Policy 流水线记忆

> 复核时间：2026-08-30。
>
> 仓库：`Junb0Dong/lerobot`，分支 `main`，基准 commit `bf31dd794ffb4f87380aba3912f64421e8352d3c`，项目版本 `0.6.2`。

## 1. 项目定位

LeRobot 是完整的 PyTorch 机器人学习框架，不只是硬件驱动。它提供统一的 LeRobotDataset、policy、processor、仿真环境、真实机器人抽象、训练/checkpoint、Hub 发布、仿真评估和真机部署。

项目要求 Python 3.12+，依赖和开发命令以 `pyproject.toml`、`uv.lock` 为准。源码位于 `src/lerobot/`，CLI 由 `pyproject.toml [project.scripts]` 注册。开发环境优先使用：

```bash
uv sync --locked
uv sync --locked --extra test --extra dev
uv run pytest tests -svv --maxfail=10
pre-commit run --all-files
```

## 2. 核心目录

```text
src/lerobot/
  scripts/          train/eval/rollout/record/replay/teleoperate 等 CLI 入口
  configs/          draccus dataclass 配置和 recipe
  datasets/         LeRobotDataset、metadata、streaming、工具和 dataset factory
  policies/         ACT、Diffusion、SmolVLA、Pi0/Pi0.5、X-VLA 等
  processor/        observation/action 的可组合预处理与后处理流水线
  robots/           真实机器人实现
  teleoperators/    leader arm、键盘、手柄、手机等遥操作设备
  cameras/ motors/  相机和电机抽象
  envs/             仿真 benchmark 环境和 EnvConfig
  rollout/          真机部署 context、strategy 和 inference engine
  async_inference/  独立 GPU policy server + robot client
  common/           checkpoint、训练和控制公共逻辑

tests/              按模块组织的 pytest；包含训练、policy、rollout、async 等测试
docs/source/        当前用户文档（MDX）
examples/           数据集、训练、硬件和 RTC 示例
docker/             用户/CI 和各仿真 benchmark 镜像
```

## 3. CLI 与职责边界

- `lerobot-record`：遥操作并录制 LeRobotDataset；也能以 policy 产生动作。
- `lerobot-train`：从本地或 Hub 数据集训练/微调 policy 或 reward model。
- `lerobot-eval`：在 `EnvConfig` 对应的仿真 benchmark 中批量 rollout，计算 reward/success。
- `lerobot-rollout`：在真实机器人上部署 policy；是当前源码和 `docs/source/inference.mdx` 推荐的统一入口。
- `lerobot-replay`：把指定数据集 episode 的动作重放到机器人，用于验证数据和硬件一致性。
- `lerobot-teleoperate`、`lerobot-calibrate`、`lerobot-setup-motors`：真机准备与遥操作。

不要混淆三类“评估”：

1. **离线 held-out loss**：训练时设置 `--dataset.eval_split=<比例> --eval_steps=<频率>`；按 task 留出最后一部分 episode。
2. **仿真成功率**：使用 `lerobot-eval --env.type=...`；要求 policy 与对应 benchmark 的 observation/action contract 匹配。
3. **真机成功率**：使用 `lerobot-rollout` 的 `base` 快速观察，或 `episodic` 录制多次评估 episode，再人工/工具统计成功率。

`AGENT_GUIDE.md` 的部分真机评估示例仍使用 `lerobot-record --policy.path`，该方式仍受 record 支持；新部署功能和策略以 `docs/source/inference.mdx` 与 `src/lerobot/rollout/` 为准。

## 4. 数据集链路

`LeRobotDataset` 将 episode-aware 的状态、动作、任务和同步图像/视频统一起来，可从 Hub 下载或从 `DatasetConfig.root` 读取本地数据。训练入口通过 `make_train_eval_datasets()` 完成：

```text
DatasetConfig
  → LeRobotDatasetMetadata
  → 根据 policy 配置解析 delta timestamps
  → LeRobotDataset / StreamingLeRobotDataset
  → 可选按 task 做 episode 级 train/eval split
  → DataLoader
```

关键配置包括 `repo_id`、`root`、`revision`、`episodes`、`exclude_episodes`、`streaming`、`video_backend` 和 `eval_split`。相机 key、状态/action feature 的顺序、FPS、dataset revision 和统计量会直接影响训练及部署兼容性。

数据训练前至少应：可视化 episode、检查丢帧/模糊/错误动作、核对 feature shape，并在真机上用 `lerobot-replay` 验证动作语义。

## 5. Policy 训练调用链

入口是 `src/lerobot/scripts/lerobot_train.py`：

```text
CLI / YAML recipe
  → draccus 解析 TrainPipelineConfig
  → 创建 Accelerate runtime 与并行拓扑
  → 创建 train/eval dataset
  → make_policy(policy config + dataset metadata/stats)
  → make_pre_post_processors()
  → DataLoader batch → preprocessor → policy.forward/loss
  → optimizer/scheduler/EMA 更新
  → 日志、离线 loss、仿真 eval、checkpoint、Hub
```

基础训练示例：

```bash
uv run lerobot-train \
  --dataset.repo_id=<user>/<dataset> \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_task \
  --job_name=act_task \
  --policy.repo_id=<user>/act_task
```

`make_policy()` 根据 dataset metadata 自动建立 input/output features；processor 负责图像、语言、device、归一化、相对/绝对动作等转换。部署时必须连同 checkpoint 中保存的 processor 使用，不能只拿裸模型权重另写归一化逻辑。

训练支持 Accelerate、多 GPU、PEFT、EMA、TensorBoard、W&B、streaming dataset、HF Jobs 和多种 checkpoint 格式。需要额外依赖的 policy/env/robot 使用 `pyproject.toml` 中对应 extra，不应无条件导入可选依赖。

训练指标默认写入 `{output_dir}/tb` 的 TensorBoard 事件文件（`--tensorboard.enable=false` 可关）。独立 tokenizer 入口同样写 `{output_dir}/tb`。查看：

```bash
tensorboard --logdir outputs/<job>/tb --port 6006
```

W&B 仍默认关闭，只有显式 `--wandb.enable=true` 才会连接；两个 backend 可同时开。

## 6. Checkpoint 与恢复

默认 checkpoint 位于 `outputs/train/<run>/checkpoints/<step>/`，主要结构是：

```text
<step>/
  pretrained_model/
    model.safetensors（或 PEFT/DCP 对应模型文件）
    config.json
    train_config.json
    policy_preprocessor*.json/safetensors
    policy_postprocessor*.json/safetensors
  training_state/
    optimizer、scheduler、step、RNG 等恢复状态
```

推理路径应指向 `pretrained_model/` 或 Hub model repo。恢复训练使用：

```bash
uv run lerobot-train \
  --config_path=outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

恢复时 checkpoint 内的 config、processor stats 和 training state 是权威来源；不要用当前数据集统计量随意覆盖已保存的 processor 状态。

## 7. 仿真评估

`lerobot-eval` 加载 `--policy.path`，由 env factory 创建向量化 Gymnasium 环境，并执行批量 rollout。示例：

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model \
  --env.type=<matching_env> \
  --eval.n_episodes=50 \
  --policy.device=cuda
```

benchmark 往往有重型原生依赖，优先使用 `docker/Dockerfile.benchmark.*`。不能用任意真实机器人数据训练出的 policy 直接跑 LIBERO/MetaWorld 等环境，除非 observation/action schema 和任务定义已做适配。

## 8. 真机部署

`lerobot-rollout` 的构建顺序是 policy → processor → hardware，policy 加载失败时不会先连接机器人。随后读取 observation，经 robot/政策 preprocessor 转换，执行 policy inference，再经 postprocessor/robot action processor 发给硬件。

主要 strategy：

- `base`：纯自主执行，不录制；适合首次低风险验证。
- `episodic`：按 episode 评估并录制，可在 episode 间 reset。
- `sentry`：长时间连续运行、分段录制并上传。
- `highlight`：环形缓冲，按键保存重要片段。
- `dagger`：policy 与人工纠正交替，收集 intervention 数据。

主要 inference backend：

- `sync`：默认同步推理，兼容所有 policy。
- `rtc`：后台生成并拼接 action chunk，适合 Pi0/Pi0.5/SmolVLA 等慢模型；policy 必须实现 RTC contract。

首次部署示例：

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=<checkpoint-or-hub-repo> \
  --robot.type=<registered_robot> \
  --robot.port=<port> \
  --task="<training-compatible task>" \
  --duration=30
```

真机前必须确认 policy feature、相机 key/顺序/分辨率/FPS、action key 顺序、标定 ID、动作范围和急停方式。先短时、低速、空载测试，再执行完整任务。

## 9. 远程异步推理

`src/lerobot/async_inference/` 提供 gRPC policy server 和 robot client。安装 `async` extra 后，GPU 机器运行 server，机器人控制机运行 client；policy 类型、路径、device 和 chunk 参数由 client 握手传给 server。

```bash
# GPU server
uv run python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

# Robot computer（其余 robot/policy 参数按实际配置补充）
uv run python -m lerobot.async_inference.robot_client \
  --server_address=<gpu-host>:8080 \
  --robot.type=<registered_robot> \
  --policy_type=<policy> \
  --pretrained_name_or_path=<checkpoint-or-hub-repo>
```

`actions_per_chunk`、`chunk_size_threshold` 和 aggregation 方法决定平滑性与响应速度。跨机器时 server 绑定 `0.0.0.0`，client 使用 GPU 主机的实际局域网地址，并检查防火墙和延迟。

## 10. XLeRobot 集成状态

当前 LeRobot fork 内没有原生 `xlerobot` robot type。要把相邻的 XLeRobot 项目部署到本框架，推荐新增独立可编辑安装的 `lerobot_robot_xlerobot`（以及需要时的 `lerobot_teleoperator_xlerobot`）插件：

1. 将 XLeRobot 的电机总线、相机、observation/action feature 和安全限制移植到当前 `Robot` API；
2. 用当前 processor/rollout contract 对齐 feature 顺序和动作语义；
3. 通过 `@RobotConfig.register_subclass("xlerobot")` 注册，并让包名满足自动发现前缀；
4. 增加 mock/unit tests 后，再做校准、teleoperate、record、replay、短时 rollout 的分阶段真机验证。

第三方插件由 `register_third_party_plugins()` 自动导入，支持前缀 `lerobot_robot_`、`lerobot_camera_`、`lerobot_teleoperator_`、`lerobot_policy_` 和 `lerobot_env_`。不要直接把旧 XLeRobot 文件复制进当前 `src/lerobot` 后假设 API 兼容。

当前只完成仓库克隆与架构梳理，尚未开始 XLeRobot 插件移植、环境安装或任何真机操作。
