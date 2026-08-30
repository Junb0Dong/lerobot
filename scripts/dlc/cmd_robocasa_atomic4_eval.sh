#!/usr/bin/env bash
# Fixed three-stage RoboCasa rollout: pretrain smoke, target smoke, target 20/task.
# This command never trains or mutates the pretrained checkpoint.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

CHECKPOINT="${CHECKPOINT:-/mnt/data/junbo/lerobot/outputs/dlc/atomic4_day_20260829_171919/policy/checkpoints/last/pretrained_model}"
ROBOCASA_EVAL_VENV="${ROBOCASA_EVAL_VENV:-${CODE_DIR}/.venv-robocasa-dlc}"
TASK_INDEX_MAP="${TASK_INDEX_MAP:-${CODE_DIR}/configs/robocasa/atomic4_actioncodec_task_indices.json}"
EVAL_RUN_ROOT="${EVAL_RUN_ROOT:-${CODE_DIR}/outputs/dlc/robocasa_atomic4_eval_${RUN_STAMP}}"
MAX_RUNNING_MINUTES="${MAX_RUNNING_MINUTES:-2880}"
INSTALL_LIBGLX=1
DISPLAY_NAME="lerobot-robocasa-atomic4-eval-$(date '+%Y%m%d-%H%M%S')"

USER_COMMAND=$(cat <<EOF
bash -euo pipefail <<INNER
export DEBIAN_FRONTEND=noninteractive
cd ${CODE_DIR}
export CODE_DIR=${CODE_DIR}
export INSTALL_LIBGLX=${INSTALL_LIBGLX}
source scripts/dlc/container_setup.sh
export PYTHON=${ROBOCASA_EVAL_VENV}/bin/python
test -x "\${PYTHON}"
"\${PYTHON}" - <<'PY'
import json
import numpy as np
import torch
from importlib.metadata import distribution, version
assert torch.cuda.is_available(), "CUDA is required"
assert torch.__version__.startswith("2.7"), torch.__version__
assert version("mujoco") == "3.3.1", version("mujoco")
expected = {
    "robocasa": "a07e365c958c4216cd6bbd5f30b47f09a65c6f00",
    "robosuite": "5ce6643f3092639d08f7b0f90ed1c6a84f50552c",
}
for package, commit in expected.items():
    direct_url = distribution(package).read_text("direct_url.json")
    assert direct_url is not None, f"{package} is missing direct_url.json"
    payload = json.loads(direct_url)
    assert payload.get("vcs_info", {}).get("commit_id") == commit, (package, payload)
torch.from_numpy(np.zeros((1,), dtype=np.float32))
print("ROBOCASA_EVAL_ENV_OK", torch.__version__, version("robocasa"), version("robosuite"), version("mujoco"))
PY

run_eval() {
  local name="\$1"
  local split="\$2"
  local episodes="\$3"
  local batch_size="\$4"
  local output_dir=${EVAL_RUN_ROOT}/"\${name}"
  "\${PYTHON}" scripts/dlc/finalize_robocasa_actioncodec_eval.py --phase before --checkpoint ${CHECKPOINT} --output-dir "\${output_dir}" --task-index-map ${TASK_INDEX_MAP} --split "\${split}" --episodes-per-task "\${episodes}" --batch-size "\${batch_size}" --seed 42
  "\${PYTHON}" -m lerobot.scripts.lerobot_eval --policy.path=${CHECKPOINT} --policy.device=cuda --env.type=robocasa --env.task=CloseDrawer,StartCoffeeMachine,TurnOffMicrowave,TurnOffSinkFaucet --env.split="\${split}" --env.task_index_map_path=${TASK_INDEX_MAP} --env.obj_registries='[lightwheel]' --eval.n_episodes="\${episodes}" --eval.batch_size="\${batch_size}" --eval.use_async_envs=true --seed=42 --output_dir="\${output_dir}"
  "\${PYTHON}" scripts/dlc/finalize_robocasa_actioncodec_eval.py --phase after --checkpoint ${CHECKPOINT} --output-dir "\${output_dir}" --task-index-map ${TASK_INDEX_MAP} --split "\${split}" --episodes-per-task "\${episodes}" --batch-size "\${batch_size}" --seed 42
}

run_eval pretrain_smoke pretrain 2 2
run_eval target_smoke target 2 2
run_eval target_20 target 20 4
echo ROBOCASA_ATOMIC4_EVAL_OK=${EVAL_RUN_ROOT}/target_20/summary.json
INNER
EOF
)
