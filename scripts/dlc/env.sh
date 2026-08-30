#!/usr/bin/env bash
# ============================================
# PAI DLC 固定环境配置
# 填一次即可。日常提交改 cmd_*.sh，不要改这个文件。
#
# 配额 / NAS / VPC 与 oat-exact 相同（scripts/dlc/env.sh）。
# 镜像不用 oat-exact 的 py310：本仓库 requires-python >= 3.12。
# 使用 PAI 官方 pytorch 2.7.0 + py312 + cu128。uv.lock 的 Linux torch
# 是 2.11.0+cu128；DLC 复用镜像自带的 2.7.0，禁止 uv sync 覆盖。
# 容器默认复用 NAS 上的 <repo>/.venv-dlc。训练作业默认 SKIP_UV_SYNC=1：
# 探测通过则直接训；失败就失败，GPU 上不装包、不 curl uv、不访问 PyPI。
# NGC torch 需要 NumPy 1.x，venv 里的 numpy 2.x 会删掉。不要把提交机 .venv
# 当默认环境。开关：FORCE_UV_SYNC / LEROBOT_VENV / UV_OFFLINE / SKIP_UV_SYNC。
# 首次建环境：scripts/dlc/cmd_bootstrap_venv.sh。
# ============================================

# --- 地域 & 工作空间 ---
export REGION="cn-wulanchabu"
export WORKSPACE_ID="241942"

# --- 计算资源 ---
# 本工作空间禁止公共资源组（不传 resource-id 会报 PayAsYouGoJob is forbidden）。
# quota-4090 / quota8zqfbd61176 是配额资源，JobSpecs 必须用 ResourceConfig，不能用 EcsSpec。
export RESOURCE_ID="quota8zqfbd61176"
export RESOURCE_CPU="16"
export RESOURCE_MEMORY="64Gi"
export RESOURCE_GPU="1"
export POD_COUNT=1
export JOB_TYPE="PyTorchJob"
export WORKER_TYPE="Worker"
# 可选。oat-exact 后期部分任务用过 32Gi；默认不设，保持与 canonical env.sh 一致。
# export RESOURCE_SHARED_MEMORY="32Gi"

# --- 镜像：必须使用 ListImages 返回的原样 URI ---
# PAI 官方 pytorch 2.7.0-gpu-py312-cu128。不要换成 oat-exact py310，也不要
# 让容器 uv sync 把 torch 升级成 lock 里的 2.11.0+cu128。
export IMAGE="dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/pytorch:2.7.0-gpu-py312-cu128-ubuntu24.04-ngc25.03-deep-ep-4d7302ff-1770029761"

# --- 数据/代码挂载（NAS 或 CPFS）---
export DATA_URI="nas://29377a494cb-afl20.cn-wulanchabu.nas.aliyuncs.com/"
export MOUNT_PATH="/mnt/data/"

# --- 运行参数 ---
export MAX_RUNNING_MINUTES=2880
export PRIORITY=9
export ACCESSIBILITY="PRIVATE"

# --- DLC 在启动命令前 pip install 的包。本项目用 uv，默认留空 ---
export THIRD_PARTY_LIBS=""

# --- Settings ---
export ENABLE_SANITY_CHECK=false
export ENABLE_RDMA=false

# MuJoCo EGL 需要 NVIDIA 图形库在容器启动时挂载。只在 UserCommand 里 export 来不及。
# 不要继承提交机上的 NVIDIA_DRIVER_CAPABILITIES（那是当前机器的能力，常缺 graphics）。
export DLC_NVIDIA_DRIVER_CAPABILITIES="${DLC_NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"

# --- VPC：NAS/CPFS 挂载需要，与文件系统同一 VPC ---
export VPC_ID="vpc-0jl0qszzyyp06tgve81xu"
export VSWITCH_ID="vsw-0jl20rpd66vzf0ou1p3fg"
export SECURITY_GROUP_ID="sg-0jlhz9qmc343k5d62xss"
export EXTENDED_CIDRS="10.0.2.0/24,10.0.1.0/24"

# --- 可选：已有数据集 / 代码源 ID，没有就留空 ---
export DATASET_ID=""
export CODE_SOURCE_ID=""
