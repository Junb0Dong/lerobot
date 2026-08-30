# PAI DLC：ActionCodec tokenizer + semantic policy

提交方式与 oat-exact / actioncodec-robocasa 相同：`env.sh` 管配额镜像 NAS VPC，`submit.sh` 调 `aliyun pai-dlc create-job`。W&B key 只放未跟踪的 `scripts/dlc/.env`。

本仓库 `requires-python >= 3.12`，镜像用 PAI 官方 **pytorch 2.7.0 + py312 + cu128**，不用 oat-exact 的 py310 镜像。`uv.lock` 的 Linux torch 是 **2.11.0+cu128**；DLC 以镜像为准，`uv sync` 跳过 torch/CUDA 整栈，禁止覆盖成 2.11。资源规格仍是 canonical 的 CPU 16 / Memory 64Gi / GPU 1。提交机上 `/mnt/workspace` 与 `/mnt/data` 是同一份 NAS；DLC 只挂 `/mnt/data/`，`paths.sh` 会把仓库路径改写成 `/mnt/data/...`。

## 提交

本机若 `http_proxy=http://127.0.0.1:7890` 且该端口没开，`submit.sh` 会自动取消代理后再调 `aliyun`。也可手动：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/dlc/submit.sh
```

```bash
# 一次性：aliyun configure --region cn-wulanchabu
# 可选：cp scripts/dlc/.env.example scripts/dlc/.env 并填写 WANDB_API_KEY

# 只打印请求
DRY_RUN=1 bash scripts/dlc/submit.sh

# CloseDrawer 小规模两段式测试（默认）：v2.1→v3 转换 + tokenizer + policy
bash scripts/dlc/submit.sh
# 等价于
STAGE=test bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_pipeline.sh

# 只转换
bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_convert.sh

# 只建 / 修复 NAS 上的 .venv-dlc（之后训练作业应直接 reuse，不再拉包）
bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_venv.sh

# 只训 tokenizer
bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_tokenizer.sh

# 只训 policy（必须指向已有 tokenizer 目录）
TOKENIZER_PATH=/mnt/data/junbo/lerobot/outputs/dlc/<run>/tokenizer \
  bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_policy.sh

# 把步数改成正式训练量
STAGE=full bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_pipeline.sh

# 四任务 atomic（CloseDrawer + StartCoffeeMachine + TurnOffMicrowave + TurnOffSinkFaucet）
# 先本地转换+merge，再交 GPU：一个 tokenizer + 一个 policy
# STAGE=day：tokenizer 10000 / policy 10000 / batch 8 / workers 4
# tokenizer 默认 decoder_type=diffusion，alignment_weight=0.1（soft-DTW）
bash scripts/dlc/convert_robocasa_atomic4.sh
STAGE=day bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_atomic4_pipeline.sh

# 一次性建立固定 RoboCasa / RoboSuite simulator 环境并做 EGL smoke。
# 该作业创建独立 .venv-robocasa-dlc，不修改训练用 .venv-dlc。
bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_robocasa_eval.sh

# 评估已训练的 atomic4 ActionCodec checkpoint：
# pretrain 2/task -> target 2/task -> target 20/task，任一阶段失败即停止。
bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_atomic4_eval.sh
```

## RoboCasa rollout 评估

`cmd_robocasa_atomic4_eval.sh` 固定使用训练产物
`outputs/dlc/atomic4_day_20260829_171919/policy/checkpoints/last/pretrained_model`，也可在提交时用
`CHECKPOINT=/mnt/data/.../pretrained_model` 显式覆盖。评估不会续训或改写 checkpoint；每个阶段前后都会
核验模型、processor 与 tokenizer 文件的 SHA-256。

评估协议为 seed 42、EGL、Lightwheel objects、异步 vector env、确定性 argmax（checkpoint
`temperature=0`）、两帧 observation history、每 16 步重规划 20-step chunk。任务固定为
`CloseDrawer`、`StartCoffeeMachine`、`TurnOffMicrowave`、`TurnOffSinkFaucet`，horizon 从固定的
RoboCasa registry 读取（450/300/300/300）。报告中 `StartCoffeeMachine` 同时标注历史别名
`CoffeePressButton`。

输出位于 `outputs/dlc/robocasa_atomic4_eval_<stamp>/`。每个阶段均包含 `protocol.json`、
`checkpoint_hashes_before.json`、官方 `eval_info.json`、rollout 视频和经严格计数校验的
`summary.json`；summary 还记录 simulator/package 版本和 action clipping 比例。正式结果只接受
`target_20/summary.json`。如果 `.venv-robocasa-dlc` 缺失或 commit/version probe 失败，评估作业会
立即退出，需先运行 bootstrap 作业，不能在正式 rollout 中临时安装依赖。

本地转换（不提交 DLC，也不改原始数据）：

```bash
bash scripts/dlc/convert_robocasa_v21_to_v30.sh
# 四个 atomic 任务各自转 v3，再用官方 aggregate_datasets 合并（全局 task_index）
bash scripts/dlc/convert_robocasa_atomic4.sh
```

## 数据

原始 CloseDrawer 是 **LeRobot v2.1**：

```text
/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot
```

`convert_dataset_v21_to_v30.py` 只支持原地转换。脚本会先 `cp -a` 到：

```text
<repo>/outputs/datasets/robocasa_closedrawer_v3
```

再转换这份拷贝。`/mnt/data/junbo/data` 不会被改写。不要复用 `/tmp/ac_data`（那是其它验证作业的路径）。

CloseDrawer 实测：`action_dim=12`、`state_dim=16`、`fps=20`、3 路 256×256、110 episodes、`total_tasks=2`。tokenizer / policy 都从 `meta/info.json` 读 `action_dim`，不会用配置默认值 7。policy 的 `--policy.num_tasks` 至少为 2。

四任务 atomic（`STAGE=day`）把同一目录下 CloseDrawer / StartCoffeeMachine / TurnOffMicrowave / TurnOffSinkFaucet 各自 `cp -a` 到 `outputs/datasets/robocasa_<task>_v3` 再转 v3.0，然后用官方 `aggregate_datasets` 合成：

```text
<repo>/outputs/datasets/robocasa_atomic4_v3
```

`MultiLeRobotDataset` 在 `make_dataset` 里是 `NotImplementedError`，`DatasetConfig.repo_id` 也只收单个字符串，所以不用列表 concat。官方 merge 按 **task 字符串** 重编号 `task_index`，避免四个数据集各自从 0 编号撞号。合并后 `action_dim=12`、`state_dim=16`、fps=20、同一组三路相机、432 ep / 56934 frames、`num_tasks=9`（5 条实际语言指令 + 4 条未使用的任务名占位）。`TurnOffMicrowave` 单独 `total_tasks=1`，合在一起后 policy 才能过 `num_tasks >= 2`。

## 训练入口

- tokenizer：`.venv/bin/python -m lerobot.scripts.lerobot_train_actioncodec_tokenizer`
- policy：`.venv/bin/python -m lerobot.scripts.lerobot_train --policy.type=actioncodec --policy.push_to_hub=false --policy.tokenizer_path=...`

tokenizer 默认 `decoder_type=diffusion`、`alignment_weight=0.1`（soft-DTW / `weight_chunk_align`）。hard-DTW 需显式设 `hard_alignment_weight`。`WANDB_ENABLE=0`。打开 W&B：`WANDB_ENABLE=1` 并在 `scripts/dlc/.env` 写 key。

容器默认**复用** NAS 上的 `<repo>/.venv-dlc`：`python -c "import lerobot, torch, numpy; torch.from_numpy(...)"` 成功且 torch 以 `2.7` 开头则**零网络、不装 uv、不 `uv sync`**。

**训练作业（pipeline / tokenizer / policy，以及 convert）默认 `SKIP_UV_SYNC=1`：** 探测失败就失败，GPU 上不 curl uv、不访问 PyPI、不 `uv sync`。不要用提交机 `.venv`。建环境请走 `cmd_bootstrap_venv.sh`（该脚本把 `SKIP_UV_SYNC` 默认成 0）。`FORCE_UV_SYNC=1` 在训练作业上会被忽略。

`lerobot-train` 在 `resume=False` 时若 `output_dir` 已存在会 `FileExistsError`。`lerobot_dlc_train_policy` **只建** `$(dirname output_dir)` 和 `RUN_ROOT/logs`，**不**预建空的 `policy/`（`dlc1kwqe4qhqshsn` 的根因）。tokenizer CLI 没有这项检查，保存时 `mkdir(parents=True, exist_ok=True)`，脚本仍可 `mkdir -p` tokenizer 输出目录。

NGC 镜像的 torch 按 **NumPy 1.x** 编译。`uv.lock` 的 numpy 2.2.6 一旦装进 venv，`torch.from_numpy` 会报 `RuntimeError: Numpy is not available`。已有 venv 里的 numpy 2.x 会被删掉，改用镜像自带的 1.x。不要为了 numpy>=2 去改 `uv.lock`。

`uv` 二进制缓存在 `<repo>/.cache/uv-dlc/bin/uv`，不要每次 `curl https://astral.sh/uv/install.sh`。训练用 `opencv-python-headless`，默认不 `apt-get install libglx0`（那会连带 mesa/libllvm）。

不要默认用提交机 `.venv`。venv 的 `bin/python` symlink 和 `pyvenv.cfg` 的 `home=` 绑死创建时的解释器绝对路径；提交机 uv 的 CPython（例如 `.../uv-python/cpython-3.12...`）在 DLC 官方镜像里通常不存在，`source .venv/bin/activate` 会失败。本仓库提交机 `.venv` 即便解释器在 NAS 上、容器里能启动，也是 **torch 2.11.0+cu130** 且 `include-system-site-packages = false`，和镜像 2.7.0+cu128 不合，探测会拒绝。对照：actioncodec-robocasa 的 lerobot v0.6.1 作业直接跑 NAS 预装环境、不 pip/uv；oat-exact 仍每次 `uv sync --frozen`。

```bash
# 默认：有可用 .venv-dlc 就直接训（日志应出现 SKIP_UV_SYNC=1 和 reusing ... 且没有 Downloading）
bash scripts/dlc/submit.sh

# 指定已有环境（仍要过 import lerobot + torch 2.7 + from_numpy 探测；不会 uv sync 进这个目录）
LEROBOT_VENV=/mnt/data/junbo/lerobot/.venv-dlc bash scripts/dlc/submit.sh

# 只训 policy（数据 v3 和 tokenizer 已在时最省 GPU）
TOKENIZER_PATH=/mnt/data/junbo/lerobot/outputs/dlc/<run>/tokenizer \
  bash scripts/dlc/submit.sh scripts/dlc/cmd_robocasa_closedrawer_policy.sh

# 只建 / 重建 .venv-dlc（不会动提交机 .venv；这是唯一默认允许 uv sync 的作业）
bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_venv.sh
FORCE_UV_SYNC=1 bash scripts/dlc/submit.sh scripts/dlc/cmd_bootstrap_venv.sh

# 训练作业上显式允许 sync（不推荐）：SKIP_UV_SYNC=0 bash scripts/dlc/submit.sh
```

怎么确认没在拉包：作业日志里应有 `SKIP_UV_SYNC=1: no curl uv, no uv sync, no PyPI`、`reusing /mnt/data/junbo/lerobot/.venv-dlc` 和 `reusing ... (no uv install, no uv sync, no PyPI)`，且**没有** `installing uv`、`uv sync`、`Downloading `、`Prepared N packages`、`syncing because`。怎么确认 policy 不再撞 FileExistsError：日志里 policy 启动后应出现训练 step，而不是 `Output directory .../policy already exists`。

镜像已带 torch 2.7.0+cu128；`uv.lock` 钉的是 2.11.0+cu128。首次（或 FORCE）sync 用 `--frozen --inexact --no-install-package torch`（以及 torchvision / torchcodec / triton / numpy / nvidia-* / cuda-*）跳过 lock 的 CUDA 整栈和 numpy 2.x，训练前断言 `torch.__version__` 仍以 `2.7` 开头且 `torch.from_numpy` 可用。不要从 `download.pytorch.org` 再拉 wheel。torchcodec 缺 `libavutil.so.56` 会回退 pyav，这是预期现象。

产出：

```text
outputs/dlc/closedrawer_<stage>_<stamp>/
  tokenizer/{model.safetensors,model_config.json,action_stats.json,dataset_contract.json}
  policy/checkpoints/last/pretrained_model/

outputs/dlc/atomic4_<stage>_<stamp>/   # 四任务同一套 tokenizer + policy
  tokenizer/...
  policy/...
```

## 最近作业

- 失败 `dlc1myilu2h2pxn3`（`lerobot-ac-closedrawer-pipe-test-20260828-232507`）：uv sync 在 NAS `copy` 上准备 73 个包花了 175 分钟，装入 numpy 2.2.6，镜像 torch（NumPy 1.x ABI）在 `torch.from_numpy` 时报 `RuntimeError: Numpy is not available`。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1myilu2h2pxn3&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942
- 失败 `dlc1kwqe4qhqshsn`（`lerobot-ac-closedrawer-pipe-test-20260829-103453`）：tokenizer 已 `TOKENIZER_OK`，policy 因预建空 `policy/` 撞 `FileExistsError`。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1kwqe4qhqshsn&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942
- 重提 policy-only `STAGE=test`：`dlc1a3enm2kewpxk`（`lerobot-ac-closedrawer-pol-test-20260829-105950`）。**Succeeded**（约 2 分钟）。`SKIP_UV_SYNC=1`，reuse `.venv-dlc`（torch 2.7 / numpy 1.26.4），无 uv/PyPI，无 FileExistsError，`POLICY_OK`，loss 5.468→约 1.5。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlc1a3enm2kewpxk&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942
- 四任务 atomic `STAGE=day`（diffusion + soft-DTW）：`dlcwe3hu26clf4dx`（`lerobot-ac-atomic4-pipe-day-20260830-162108`）。tokenizer 10000 + policy 10000，`DECODER_TYPE=diffusion`，`ALIGNMENT_WEIGHT=0.1`，batch 8，workers 4，`SKIP_UV_SYNC=1`。数据 `outputs/datasets/robocasa_atomic4_v3`。产出 `outputs/dlc/atomic4_day_20260830_162108/`。控制台：https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=dlcwe3hu26clf4dx&region=cn-wulanchabu&regionId=cn-wulanchabu&workspaceId=241942
