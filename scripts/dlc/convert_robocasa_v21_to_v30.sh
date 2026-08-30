#!/usr/bin/env bash
# Copy a LeRobot v2.1 dataset and convert it to v3.0 without touching the original.
#
# Usage:
#   bash scripts/dlc/convert_robocasa_v21_to_v30.sh [src] [dst] [repo_id]
#
# Defaults match CloseDrawer. The official convert script is in-place, so this
# copies first and only converts the copy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SRC="${1:-${ROBOCASA_RAW_DATASET:-/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot}}"
DST="${2:-${ROBOCASA_V3_DATASET:-${CODE_DIR}/outputs/datasets/robocasa_closedrawer_v3}}"
REPO_ID="${3:-${ROBOCASA_REPO_ID:-robocasa/CloseDrawer}}"
PYTHON="${PYTHON:-${CODE_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python)"
fi
CONVERT_PY="${CODE_DIR}/src/lerobot/scripts/convert_dataset_v21_to_v30.py"

if [[ ! -d "${SRC}" ]]; then
  echo "[error] source dataset does not exist: ${SRC}" >&2
  exit 1
fi
if [[ "${SRC}" == /mnt/data/junbo/data/* && "${DST}" == "${SRC}" ]]; then
  echo "[error] refusing to convert in place under /mnt/data/junbo/data" >&2
  exit 1
fi
if [[ ! -f "${CONVERT_PY}" ]]; then
  echo "[error] missing convert script: ${CONVERT_PY}" >&2
  exit 1
fi

codebase_version() {
  local root="$1"
  if [[ ! -f "${root}/meta/info.json" ]]; then
    echo "missing"
    return 0
  fi
  "${PYTHON}" - "${root}/meta/info.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("codebase_version", "unknown"))
PY
}

SRC_VERSION="$(codebase_version "${SRC}")"
if [[ "${SRC_VERSION}" != "v2.1" && "${SRC_VERSION}" != "v3.0" ]]; then
  echo "[error] source codebase_version=${SRC_VERSION}, expected v2.1 or v3.0: ${SRC}" >&2
  exit 1
fi

if [[ -d "${DST}" ]]; then
  DST_VERSION="$(codebase_version "${DST}")"
  if [[ "${DST_VERSION}" == "v3.0" ]]; then
    echo "[convert] already v3.0, skip: ${DST}"
    exit 0
  fi
  echo "[error] destination exists but is not v3.0 (${DST_VERSION}): ${DST}" >&2
  echo "        remove it or pick another ROBOCASA_V3_DATASET" >&2
  exit 1
fi

if [[ "${SRC_VERSION}" == "v3.0" ]]; then
  echo "[convert] source is already v3.0; copying ${SRC} -> ${DST}"
  mkdir -p "$(dirname "${DST}")"
  cp -a "${SRC}" "${DST}"
  echo "[convert] done ${DST}"
  exit 0
fi

echo "[convert] copying v2.1 ${SRC} -> ${DST} (original is not modified)"
mkdir -p "$(dirname "${DST}")"
cp -a "${SRC}" "${DST}"

echo "[convert] running v2.1 -> v3.0 on ${DST}"
"${PYTHON}" "${CONVERT_PY}" --repo-id="${REPO_ID}" --root="${DST}" --push-to-hub=false

DST_VERSION="$(codebase_version "${DST}")"
if [[ "${DST_VERSION}" != "v3.0" ]]; then
  echo "[error] conversion left codebase_version=${DST_VERSION}" >&2
  exit 1
fi

OLD_COPY="$(dirname "${DST}")/$(basename "${DST}")_old"
if [[ -d "${OLD_COPY}" ]]; then
  echo "[convert] removing leftover v2.1 copy ${OLD_COPY}"
  rm -rf "${OLD_COPY}"
fi

echo "[convert] done ${DST} codebase_version=${DST_VERSION}"
