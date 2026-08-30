#!/usr/bin/env bash
# Build a simulator-capable copy of the known-good DLC training venv.

set -euo pipefail

CODE_DIR="${CODE_DIR:?CODE_DIR is required}"
BASE_VENV="${BASE_VENV:-${CODE_DIR}/.venv-dlc}"
EVAL_VENV="${ROBOCASA_EVAL_VENV:-${CODE_DIR}/.venv-robocasa-dlc}"
ROBOCASA_COMMIT="a07e365c958c4216cd6bbd5f30b47f09a65c6f00"
ROBOSUITE_COMMIT="5ce6643f3092639d08f7b0f90ed1c6a84f50552c"

probe_eval_venv() {
  [[ -x "${EVAL_VENV}/bin/python" ]] || return 1
  "${EVAL_VENV}/bin/python" - <<'PY' >/dev/null 2>&1
import json
import numpy as np
import torch
from importlib.metadata import version
from importlib.metadata import distribution
import lerobot
import robosuite
assert torch.__version__.startswith("2.7")
assert version("mujoco") == "3.3.1"
assert np.__version__ == "1.26.4", np.__version__
actual_numpy_version = np.__version__
np.__version__ = "2.2.5"
try:
    import robocasa
finally:
    np.__version__ = actual_numpy_version
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
PY
}

if probe_eval_venv; then
  echo "[robocasa-eval] reusing ${EVAL_VENV}"
  exit 0
fi

if [[ -e "${EVAL_VENV}" ]]; then
  if [[ "${RESUME_ROBOCASA_EVAL_BOOTSTRAP:-0}" == "1" ]]; then
    echo "[robocasa-eval] resuming package/setup smoke in existing ${EVAL_VENV}"
  elif [[ "${FORCE_ROBOCASA_EVAL_BOOTSTRAP:-0}" != "1" ]]; then
    echo "[error] ${EVAL_VENV} exists but failed its probe; set FORCE_ROBOCASA_EVAL_BOOTSTRAP=1 to rebuild it" >&2
    exit 1
  else
    case "${EVAL_VENV}" in
      "${CODE_DIR}/.venv-robocasa-dlc") rm -rf "${EVAL_VENV}" ;;
      *) echo "[error] refusing to remove unexpected eval venv path: ${EVAL_VENV}" >&2; exit 1 ;;
    esac
  fi
fi

if [[ ! -x "${BASE_VENV}/bin/python" ]]; then
  echo "[error] known-good base venv is missing: ${BASE_VENV}" >&2
  exit 1
fi

if [[ "${RESUME_ROBOCASA_EVAL_BOOTSTRAP:-0}" != "1" ]]; then
  echo "[robocasa-eval] copying ${BASE_VENV} -> ${EVAL_VENV}"
  mkdir -p "${EVAL_VENV}"
  cp -a "${BASE_VENV}/." "${EVAL_VENV}/"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv is required to install the pinned simulator packages" >&2
  exit 1
fi

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
uv pip install --python "${EVAL_VENV}/bin/python" "numpy<2" "mujoco==3.3.1" "git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_COMMIT}"
# RoboCasa pins numpy==2.2.5 and lerobot==0.3.3 in package metadata. Neither
# can be installed into this checkpoint's Torch 2.7 / LeRobot 0.6 environment,
# so install its remaining declared runtime dependencies explicitly, then add
# RoboCasa itself with --no-deps below.
# Install simulator/import dependencies without resolving their dependency
# graph. In particular, tianshou is a training-only dependency whose resolver
# otherwise downloads a second Torch/CUDA stack; it is intentionally omitted
# from this rollout-only environment.
uv pip install --python "${EVAL_VENV}/bin/python" --no-deps pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml hidapi gymnasium
uv pip install --python "${EVAL_VENV}/bin/python" --no-deps "git+https://github.com/robocasa/robocasa.git@${ROBOCASA_COMMIT}"

"${EVAL_VENV}/bin/python" - <<'PY'
import ctypes

egl = ctypes.CDLL("libEGL.so.1")
assert getattr(egl, "eglQueryString") is not None
print("EGL_LOADER_OK", egl._name)
PY

SITE_PACKAGES=$("${EVAL_VENV}/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)
if [[ -f "${SITE_PACKAGES}/robosuite/macros_private.py" ]]; then
  echo "[robocasa-eval] robosuite macros already configured"
else
  "${EVAL_VENV}/bin/python" -m robosuite.scripts.setup_macros
fi
if [[ -f "${SITE_PACKAGES}/robocasa/macros_private.py" ]]; then
  echo "[robocasa-eval] robocasa macros already configured"
else
  cp "${SITE_PACKAGES}/robocasa/macros.py" "${SITE_PACKAGES}/robocasa/macros_private.py"
fi
"${EVAL_VENV}/bin/python" scripts/dlc/run_robocasa_numpy1_compat.py robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw

"${EVAL_VENV}/bin/python" - <<'PY'
import json
import numpy as np
import torch
from importlib.metadata import distribution, version
import robosuite

assert torch.cuda.is_available(), "CUDA is required for the EGL simulator smoke"
assert torch.__version__.startswith("2.7"), torch.__version__
assert version("mujoco") == "3.3.1", version("mujoco")
assert np.__version__ == "1.26.4", np.__version__
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
actual_numpy_version = np.__version__
np.__version__ = "2.2.5"
try:
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
finally:
    np.__version__ = actual_numpy_version
env = RoboCasaGymEnv(
    env_name="TurnOffSinkFaucet",
    split="target",
    camera_widths=256,
    camera_heights=256,
    obj_registries=("lightwheel",),
)
obs, _ = env.reset(seed=42)
for camera in ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"):
    assert obs[f"video.{camera}"].shape == (256, 256, 3)
env.close()
print(
    "ROBOCASA_EVAL_BOOTSTRAP_OK",
    "torch", torch.__version__,
    "robocasa", version("robocasa"),
    "robosuite", version("robosuite"),
    "mujoco", version("mujoco"),
)
PY
