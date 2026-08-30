#!/usr/bin/env bash
# ============================================
# PAI DLC 任务提交（aliyun CLI）
#
# 用法:
#   bash submit.sh
#       加载 env.sh + cmd.sh 后提交
#   bash submit.sh cmd.sh
#       加载 env.sh + 指定命令文件后提交
#   bash submit.sh "任务名" "训练命令"
#       加载 env.sh，命令行覆盖任务名和启动命令
#
# 只打印请求不提交:
#   DRY_RUN=1 bash submit.sh
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi

usage() {
  cat <<'EOF'
用法:
  bash submit.sh
  bash submit.sh <cmd-file.sh>
  bash submit.sh <任务名> "<训练命令>"

先编辑 env.sh 填 NAS URI，再编辑 cmd.sh 填启动命令。
EOF
}

load_cmd_file() {
  local cmd_file="$1"
  if [[ ! -f "$cmd_file" ]]; then
    echo "[error] 找不到命令文件: $cmd_file" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$cmd_file"
}

case "$#" in
  0)
    load_cmd_file "${SCRIPT_DIR}/cmd.sh"
    ;;
  1)
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
      usage
      exit 0
    fi
    load_cmd_file "$1"
    ;;
  2)
    DISPLAY_NAME="$1"
    USER_COMMAND="$2"
    ;;
  *)
    usage
    exit 1
    ;;
esac

DISPLAY_NAME="${DISPLAY_NAME:?请设置 DISPLAY_NAME}"
USER_COMMAND="${USER_COMMAND:?请设置 USER_COMMAND}"

if [[ -z "${DATA_URI:-}" || "${DATA_URI}" == *"<your-fsid>"* ]]; then
  echo "[error] 请先在 env.sh 里填写真实的 DATA_URI" >&2
  exit 1
fi
if [[ -z "${RESOURCE_ID:-}" ]]; then
  echo "[error] 请先在 env.sh 里填写 RESOURCE_ID。不传会走公共资源组，报 PayAsYouGoJob is forbidden" >&2
  exit 1
fi
if [[ -z "${RESOURCE_CPU:-}" || -z "${RESOURCE_MEMORY:-}" || -z "${RESOURCE_GPU:-}" ]]; then
  echo "[error] 配额任务必须设置 RESOURCE_CPU / RESOURCE_MEMORY / RESOURCE_GPU（ResourceConfig）" >&2
  exit 1
fi
if [[ -z "${VPC_ID:-}" || -z "${VSWITCH_ID:-}" || -z "${SECURITY_GROUP_ID:-}" ]]; then
  echo "[error] 请先在 env.sh 里填写 VPC_ID / VSWITCH_ID / SECURITY_GROUP_ID" >&2
  exit 1
fi

RESOURCE_SHARED_MEMORY_JSON=""
if [[ -n "${RESOURCE_SHARED_MEMORY:-}" ]]; then
  RESOURCE_SHARED_MEMORY_JSON=",\"SharedMemory\":\"${RESOURCE_SHARED_MEMORY}\""
fi
JOB_SPECS=$(cat <<EOF
[{"Type":"${WORKER_TYPE}","Image":"${IMAGE}","PodCount":${POD_COUNT},"ResourceConfig":{"CPU":"${RESOURCE_CPU}","Memory":"${RESOURCE_MEMORY}","GPU":"${RESOURCE_GPU}"${RESOURCE_SHARED_MEMORY_JSON}}}]
EOF
)

DATA_SOURCES="["
sep=""
if [[ -n "${DATASET_ID:-}" ]]; then
  DATA_SOURCES="${DATA_SOURCES}${sep}{\"DataSourceId\":\"${DATASET_ID}\",\"MountPath\":\"${MOUNT_PATH}\"}"
  sep=","
fi
DATA_SOURCES="${DATA_SOURCES}${sep}{\"Uri\":\"${DATA_URI}\",\"MountPath\":\"${MOUNT_PATH}\"}]"

EXTENDED_CIDRS_JSON="["
cidr_sep=""
IFS=',' read -r -a _cidrs <<< "${EXTENDED_CIDRS:-}"
for cidr in "${_cidrs[@]}"; do
  cidr="${cidr// /}"
  [[ -z "${cidr}" ]] && continue
  EXTENDED_CIDRS_JSON="${EXTENDED_CIDRS_JSON}${cidr_sep}\"${cidr}\""
  cidr_sep=","
done
EXTENDED_CIDRS_JSON="${EXTENDED_CIDRS_JSON}]"

USER_VPC="{\"VpcId\":\"${VPC_ID}\",\"SwitchId\":\"${VSWITCH_ID}\",\"SecurityGroupId\":\"${SECURITY_GROUP_ID}\""
if [[ "${EXTENDED_CIDRS_JSON}" != "[]" ]]; then
  USER_VPC="${USER_VPC},\"ExtendedCIDRs\":${EXTENDED_CIDRS_JSON}"
fi
USER_VPC="${USER_VPC}}"

SETTINGS="{\"EnableSanityCheck\":${ENABLE_SANITY_CHECK},\"EnableRDMA\":${ENABLE_RDMA}}"

echo "=========================================="
echo "  PAI DLC 任务提交"
echo "=========================================="
echo "  任务名:   ${DISPLAY_NAME}"
echo "  地域:     ${REGION}"
echo "  工作空间: ${WORKSPACE_ID}"
echo "  配额:     ${RESOURCE_ID}"
echo "  资源:     CPU=${RESOURCE_CPU} Memory=${RESOURCE_MEMORY} GPU=${RESOURCE_GPU}"
if [[ -n "${RESOURCE_SHARED_MEMORY:-}" ]]; then
  echo "  共享内存: ${RESOURCE_SHARED_MEMORY}"
fi
echo "  镜像:     ${IMAGE}"
echo "  挂载:     ${DATA_URI} -> ${MOUNT_PATH}"
echo "  VPC:      ${VPC_ID}"
DLC_GRAPHICS_CAPS="${DLC_NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
echo "  EGL:      NVIDIA_DRIVER_CAPABILITIES=${DLC_GRAPHICS_CAPS}"
echo "  命令:"
echo "${USER_COMMAND}" | sed 's/^/    /'
echo "=========================================="

ALIYUN_ARGS=(
  pai-dlc create-job
  --region "${REGION}"
  --resource-id "${RESOURCE_ID}"
  --workspace-id "${WORKSPACE_ID}"
  --display-name "${DISPLAY_NAME}"
  --job-type "${JOB_TYPE}"
  --user-command "${USER_COMMAND}"
  --job-specs "${JOB_SPECS}"
  --data-sources "${DATA_SOURCES}"
  --user-vpc "${USER_VPC}"
  --job-max-running-time-minutes "${MAX_RUNNING_MINUTES}"
  --priority "${PRIORITY}"
  --settings "${SETTINGS}"
  --accessibility "${ACCESSIBILITY}"
)

if [[ -n "${THIRD_PARTY_LIBS:-}" ]]; then
  # shellcheck disable=SC2086
  for lib in ${THIRD_PARTY_LIBS}; do
    ALIYUN_ARGS+=(--thirdparty-libs "${lib}")
  done
fi

if [[ -n "${CODE_SOURCE_ID:-}" ]]; then
  ALIYUN_ARGS+=(--code-source "{\"CodeSourceId\":\"${CODE_SOURCE_ID}\"}")
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="${WANDB_API_KEY#\'}"
  WANDB_API_KEY="${WANDB_API_KEY%\'}"
  WANDB_API_KEY="${WANDB_API_KEY#\"}"
  WANDB_API_KEY="${WANDB_API_KEY%\"}"
fi
DLC_ENVS="{\"NVIDIA_DRIVER_CAPABILITIES\":\"${DLC_GRAPHICS_CAPS}\""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  DLC_ENVS="${DLC_ENVS},\"WANDB_API_KEY\":\"${WANDB_API_KEY}\""
else
  echo "[warn] 未设置 WANDB_API_KEY。请写在 ${SCRIPT_DIR}/.env（不是 .env.example）" >&2
  echo "[warn] online 模式会登录失败；也可把 cmd.sh 里 WANDB_MODE 改成 offline" >&2
fi
DLC_ENVS="${DLC_ENVS}}"
ALIYUN_ARGS+=(--envs "${DLC_ENVS}")

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-}" == "true" ]]; then
  echo "[dry-run] aliyun ${ALIYUN_ARGS[*]}"
  echo "[dry-run] 未提交"
  exit 0
fi

# PAI API 在 VPC 内可达。本机若指向已挂掉的 127.0.0.1:7890 代理，create-job 会 connection refused。
_proxy="${HTTPS_PROXY:-${https_proxy:-}}"
if [[ "${_proxy}" == *"127.0.0.1:7890"* ]] && ! bash -c 'echo >/dev/tcp/127.0.0.1/7890' 2>/dev/null; then
  echo "[warn] 本地代理 127.0.0.1:7890 不可用，本次提交取消代理环境变量" >&2
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

if ! command -v aliyun >/dev/null 2>&1; then
  echo "[error] 未找到 aliyun CLI。安装: https://help.aliyun.com/zh/cli" >&2
  echo "        并执行: aliyun configure --region ${REGION}" >&2
  exit 1
fi

echo ""
echo "正在提交任务..."
set +e
RESP="$(aliyun "${ALIYUN_ARGS[@]}" 2>&1)"
STATUS=$?
set -e
echo "${RESP}"

if [[ "${STATUS}" -ne 0 ]]; then
  echo "[error] 提交失败" >&2
  exit "${STATUS}"
fi

JOB_ID="$(printf '%s\n' "${RESP}" | grep -oE '"JobId"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed 's/.*"\([^"]*\)".*/\1/')"
if [[ -z "${JOB_ID}" ]]; then
  echo "[error] 提交返回里没有 JobId" >&2
  exit 1
fi

CONSOLE_URL="https://pai.console.aliyun.com/ai-training/dlc/detail?jobId=${JOB_ID}&region=${REGION}&regionId=${REGION}&workspaceId=${WORKSPACE_ID}#/dlc/jobs/${JOB_ID}/overview"

echo ""
echo "任务提交成功"
echo "  JobId:   ${JOB_ID}"
echo "  控制台:  ${CONSOLE_URL}"
echo ""
echo "查看状态:"
echo "  aliyun pai-dlc get-job --job-id ${JOB_ID} --region ${REGION}"
echo "  aliyun pai-dlc list-ecs-specs --region ${REGION}"
echo "状态: Creating → Queuing → EnvPreparing → SanityChecking → Running → Succeeded/Failed/Stopped"
