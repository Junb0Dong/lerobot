#!/usr/bin/env bash
# Convert four robocasa atomic tasks v2.1→v3.0 (copy first) and merge them
# with the official aggregate_datasets API so task_index is globally unique.
#
# Usage:
#   bash scripts/dlc/convert_robocasa_atomic4.sh
#   PYTHON=/path/to/python bash scripts/dlc/convert_robocasa_atomic4.sh
#
# Never writes under /mnt/data/junbo/data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ "${CODE_DIR}" == /mnt/workspace/* ]]; then
  CODE_DIR="/mnt/data/${CODE_DIR#/mnt/workspace/}"
fi

ATOMIC4_TASKS="${ATOMIC4_TASKS:-CloseDrawer StartCoffeeMachine TurnOffMicrowave TurnOffSinkFaucet}"
ATOMIC4_DATE="${ATOMIC4_DATE:-20250819}"
ATOMIC4_RAW_ROOT="${ATOMIC4_RAW_ROOT:-/mnt/data/junbo/data/robocasa/v1.0/pretrain/atomic}"
ATOMIC4_V3_ROOT="${ATOMIC4_V3_ROOT:-${CODE_DIR}/outputs/datasets}"
ATOMIC4_MERGED="${ATOMIC4_MERGED:-${ATOMIC4_V3_ROOT}/robocasa_atomic4_v3}"
ATOMIC4_REPO_ID="${ATOMIC4_REPO_ID:-robocasa/atomic4}"
CONVERT_ONE="${CODE_DIR}/scripts/dlc/convert_robocasa_v21_to_v30.sh"
PYTHON="${PYTHON:-${CODE_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python)"
fi

read -r -a TASKS <<< "${ATOMIC4_TASKS}"

v3_dst_for() {
  local task="$1"
  echo "${ATOMIC4_V3_ROOT}/robocasa_${task,,}_v3"
}

raw_src_for() {
  local task="$1"
  echo "${ATOMIC4_RAW_ROOT}/${task}/${ATOMIC4_DATE}/lerobot"
}

echo "[atomic4] validating source contracts (action_dim / cameras / fps)"
"${PYTHON}" - "${ATOMIC4_RAW_ROOT}" "${ATOMIC4_DATE}" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

raw_root = Path(sys.argv[1])
date = sys.argv[2]
tasks = sys.argv[3:]
expected_action = 12
rows = []
for task in tasks:
    info_path = raw_root / task / date / "lerobot" / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"[error] missing {info_path}")
    info = json.loads(info_path.read_text())
    features = info["features"]
    cams = [k for k in features if k.startswith("observation.images")]
    action_dim = int(features["action"]["shape"][0])
    state_dim = int(features["observation.state"]["shape"][0])
    row = {
        "task": task,
        "version": info.get("codebase_version"),
        "robot": info.get("robot_type"),
        "fps": info.get("fps"),
        "action_dim": action_dim,
        "state_dim": state_dim,
        "cams": cams,
        "feat_keys": sorted(features),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
    }
    rows.append(row)
    print(
        f"[atomic4] {task}: version={row['version']} robot={row['robot']} "
        f"fps={row['fps']} action_dim={action_dim} state_dim={state_dim} "
        f"cams={cams} episodes={row['episodes']} frames={row['frames']} "
        f"total_tasks={row['total_tasks']}"
    )

ref = rows[0]
errors = []
for row in rows:
    if row["action_dim"] != expected_action:
        errors.append(
            f"{row['task']} action_dim={row['action_dim']} (expected {expected_action} PandaOmron)"
        )
    for key in ("robot", "fps", "action_dim", "state_dim", "cams", "feat_keys"):
        if row[key] != ref[key]:
            errors.append(f"{row['task']} {key}={row[key]!r} != {ref['task']} {key}={ref[key]!r}")
if errors:
    print("[error] refusing to convert/merge; contracts do not match:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(1)
print("[atomic4] CONTRACT_OK action_dim=12 cameras/fps/state/keys match across all tasks")
PY

mkdir -p "${ATOMIC4_V3_ROOT}"
for TASK in "${TASKS[@]}"; do
  SRC="$(raw_src_for "${TASK}")"
  DST="$(v3_dst_for "${TASK}")"
  REPO_ID="robocasa/${TASK}"
  echo "[atomic4] convert ${TASK} -> ${DST}"
  PYTHON="${PYTHON}" CODE_DIR="${CODE_DIR}" bash "${CONVERT_ONE}" "${SRC}" "${DST}" "${REPO_ID}"
done

if [[ -d "${ATOMIC4_MERGED}" ]]; then
  MERGED_VERSION="$("${PYTHON}" - "${ATOMIC4_MERGED}/meta/info.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("missing")
else:
    print(json.load(open(p)).get("codebase_version", "unknown"))
PY
)"
  if [[ "${MERGED_VERSION}" == "v3.0" ]]; then
    echo "[atomic4] merged dataset already v3.0, skip: ${ATOMIC4_MERGED}"
  else
    echo "[error] merged dest exists but is not v3.0 (${MERGED_VERSION}): ${ATOMIC4_MERGED}" >&2
    exit 1
  fi
else
  echo "[atomic4] merging ${#TASKS[@]} v3 datasets -> ${ATOMIC4_MERGED}"
  ROOTS=()
  REPO_IDS=()
  for TASK in "${TASKS[@]}"; do
    ROOTS+=("$(v3_dst_for "${TASK}")")
    REPO_IDS+=("robocasa/${TASK}")
  done
  "${PYTHON}" - "${ATOMIC4_REPO_ID}" "${ATOMIC4_MERGED}" "${#TASKS[@]}" "${REPO_IDS[@]}" "${ROOTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

aggr_repo_id = sys.argv[1]
aggr_root = Path(sys.argv[2])
n = int(sys.argv[3])
repo_ids = sys.argv[4 : 4 + n]
roots = [Path(p) for p in sys.argv[4 + n : 4 + 2 * n]]
if len(roots) != n:
    raise SystemExit(f"repo/root length mismatch: {len(repo_ids)} vs {len(roots)}")

print("[atomic4] source task tables (official merge remaps by task string, not local 0-based ids):")
for repo_id, root in zip(repo_ids, roots, strict=True):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    print(f"  {repo_id}: info.total_tasks={meta.total_tasks} tasks={list(meta.tasks.index)}")

aggregate_datasets(
    repo_ids=repo_ids,
    aggr_repo_id=aggr_repo_id,
    roots=roots,
    aggr_root=aggr_root,
)

info = json.loads((aggr_root / "meta/info.json").read_text())
action_dim = int(info["features"]["action"]["shape"][0])
cams = [k for k in info["features"] if k.startswith("observation.images")]
total_tasks = int(info["total_tasks"])
if info.get("codebase_version") != "v3.0":
    raise SystemExit(f"merged codebase_version={info.get('codebase_version')}")
if action_dim != 12:
    raise SystemExit(f"merged action_dim={action_dim}, expected 12")
if total_tasks < 2:
    raise SystemExit(f"merged total_tasks={total_tasks}, policy requires >= 2")
print(
    f"[atomic4] MERGED_OK root={aggr_root} action_dim={action_dim} "
    f"state_dim={info['features']['observation.state']['shape'][0]} "
    f"fps={info['fps']} cams={cams} episodes={info['total_episodes']} "
    f"frames={info['total_frames']} num_tasks={total_tasks}"
)
PY
fi

echo "[atomic4] merged task table:"
"${PYTHON}" - "${ATOMIC4_MERGED}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
tasks = pd.read_parquet(root / "meta/tasks.parquet")
print(tasks.to_string())
print(
    f"num_tasks={info['total_tasks']} action_dim={info['features']['action']['shape'][0]} "
    f"episodes={info['total_episodes']} frames={info['total_frames']}"
)
PY
echo "[atomic4] CONVERT_OK ${ATOMIC4_MERGED}"
