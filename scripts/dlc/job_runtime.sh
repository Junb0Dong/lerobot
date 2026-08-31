#!/usr/bin/env bash
# Container-side helpers for ActionCodec tokenizer + semantic policy DLC jobs.
# Must be sourced after `cd $CODE_DIR`.
#
# Training CLI notes:
#   tokenizer has --action_dim but no --num_heads
#   policy requires --policy.push_to_hub=false
#   wandb is off unless WANDB_ENABLE=1
#   torchcodec missing libavutil.so.56 is expected; decoder falls back to pyav
#   Image torch is 2.7.0+cu128; uv.lock pins Linux torch 2.11.0+cu128.
#   Bootstrap reuses the image torch and must not download the lock CUDA stack.
#
# Venv constraint (do not ignore):
#   venv 的 shebang / bin/python symlink / pyvenv.cfg `home=` 绑死创建时的
#   Python 绝对路径。提交机 uv 装的解释器（例如
#   .../uv-python/cpython-3.12... 或 /root/.local/share/uv/python/...）在 DLC
#   官方镜像里通常不存在，`source .venv/bin/activate` 会失败。
#   即便解释器碰巧在 NAS 上、容器能启动，提交机 `.venv` 也是 torch 2.11
#   且 include-system-site-packages=false，不能当 DLC 环境用。
#   可行方案：镜像 python3.12 + `--system-site-packages` 的 `.venv-dlc`，
#   或 VIRTUAL_ENV/PYTHONPATH / `uv run --offline --frozen --no-sync`。
#   默认复用 NAS 上已有的 `.venv-dlc`：`import lerobot, torch`、torch 以
#   2.7 开头、且 `torch.from_numpy` 可用 → 零网络：不装 uv、不 sync、不
#   访问 PyPI。训练作业（pipeline / tokenizer / policy）默认 SKIP_UV_SYNC=1：
#   探测失败就失败，GPU 上不 curl uv、不访问 PyPI。FORCE_UV_SYNC=1 只给
#   cmd_bootstrap_venv.sh 用。不要每次 rm -rf .venv-dlc。
#   NGC 镜像 torch 按 NumPy 1.x 编译；uv.lock 的 numpy 2.2.6 会让
#   torch.from_numpy 报 "Numpy is not available"。sync 时跳过 numpy，
#   复用镜像自带的 1.x。LEROBOT_VENV 可指定已有环境；不要 uv sync 提交机 `.venv`。
#   policy 不要 mkdir output_dir：lerobot-train 在 resume=False 时若目录已存在
#   会 FileExistsError。tokenizer CLI 没有这项检查，可以预建输出目录。

lerobot_dlc_log() {
  echo "[lerobot-dlc $(date '+%H:%M:%S')] $*"
}

lerobot_dlc_nas_path() {
  local p="${1:-}"
  if [[ "${p}" == /mnt/workspace/* ]]; then
    echo "/mnt/data/${p#/mnt/workspace/}"
  else
    echo "${p}"
  fi
}

lerobot_dlc_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    echo "${PYTHON}"
    return 0
  fi
  if [[ -n "${LEROBOT_VENV:-}" && -x "${LEROBOT_VENV}/bin/python" ]]; then
    echo "${LEROBOT_VENV}/bin/python"
    return 0
  fi
  if [[ -x "${CODE_DIR}/.venv-dlc/bin/python" ]]; then
    echo "${CODE_DIR}/.venv-dlc/bin/python"
    return 0
  fi
  # 不要隐式回落到提交机 .venv：那是 torch 2.11，且 pyvenv.cfg 绑的解释器
  # 在镜像里经常不存在。需要时显式设 LEROBOT_VENV。
  command -v python3 || command -v python
}

lerobot_dlc_codebase_version() {
  local root="$1"
  local info="${root}/meta/info.json"
  if [[ ! -f "${info}" ]]; then
    echo "missing"
    return 0
  fi
  "$(lerobot_dlc_python)" - "${info}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("codebase_version", "unknown"))
PY
}

lerobot_dlc_feature_dim() {
  local root="$1"
  local feature="$2"
  "$(lerobot_dlc_python)" - "${root}/meta/info.json" "${feature}" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
shape = info["features"][sys.argv[2]]["shape"]
print(int(shape[0]))
PY
}

lerobot_dlc_total_tasks() {
  local root="$1"
  "$(lerobot_dlc_python)" - "${root}/meta/info.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1])).get("total_tasks", 0)))
PY
}

lerobot_dlc_write_uv_toml() {
  # DLC 容器是全新环境，~/.config/uv/uv.toml 不会随仓库过来。
  # 非 torch 依赖走阿里云 PyPI；torch/CUDA 由镜像提供，不要配 pytorch.org index。
  mkdir -p "${HOME}/.config/uv"
  cat > "${HOME}/.config/uv/uv.toml" <<'TOML'
[[index]]
name = "pypi"
url = "https://mirrors.aliyun.com/pypi/simple"
default = true
TOML
}

lerobot_dlc_image_python() {
  local candidate
  for candidate in python3 python /usr/bin/python3 /usr/bin/python /usr/local/bin/python3 /usr/local/bin/python; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c "import torch" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  command -v python3 || command -v python
}

lerobot_dlc_enable_system_site_packages() {
  local cfg="$1/pyvenv.cfg"
  if [[ ! -f "${cfg}" ]]; then
    echo "[error] missing ${cfg}" >&2
    return 1
  fi
  if grep -q '^include-system-site-packages =' "${cfg}"; then
    sed -i 's/^include-system-site-packages = .*/include-system-site-packages = true/' "${cfg}"
  else
    echo "include-system-site-packages = true" >> "${cfg}"
  fi
}

lerobot_dlc_uv_skip_torch_args() {
  # lock 的 Linux torch 是 2.11.0+cu128，连同 nvidia-/cuda-/triton wheel。
  # 镜像已带 torch 2.7.0+cu128，跳过整栈，避免再走 pytorch.org / 大 CUDA 包。
  # NGC torch 按 NumPy 1.x 编译；lock 的 numpy 2.2.6 会让 from_numpy 失败。
  local pkg
  local pkgs=(
    torch torchvision torchcodec triton numpy
    cuda-bindings cuda-pathfinder cuda-toolkit
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12
    nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12
  )
  for pkg in "${pkgs[@]}"; do
    printf -- '--no-install-package\n%s\n' "${pkg}"
  done
}

lerobot_dlc_describe_venv() {
  local venv="$1"
  local cfg="${venv}/pyvenv.cfg"
  local py="${venv}/bin/python"
  lerobot_dlc_log "probe ${venv}"
  if [[ -f "${cfg}" ]]; then
    lerobot_dlc_log "pyvenv.cfg $(tr '\n' ' | ' < "${cfg}")"
  else
    lerobot_dlc_log "pyvenv.cfg missing"
  fi
  if [[ -e "${py}" ]]; then
    lerobot_dlc_log "python ${py} -> $(readlink "${py}" 2>/dev/null || echo not-a-symlink) executable=$(if [[ -x "${py}" ]]; then echo yes; else echo no; fi)"
  else
    lerobot_dlc_log "python missing: ${py}"
  fi
}

lerobot_dlc_venv_has_lerobot_files() {
  local venv="$1"
  local sp
  sp="$(find "${venv}/lib" -maxdepth 3 -type d -name site-packages -print -quit 2>/dev/null || true)"
  [[ -n "${sp}" ]] || return 1
  [[ -n "$(find "${sp}" -maxdepth 1 \( -name 'lerobot*' -o -name '__editable__.lerobot*' \) -print -quit 2>/dev/null || true)" ]]
}

# Relink only when packages already live on NAS but bin/python points at a
# container-local interpreter that does not exist in this image.
lerobot_dlc_relink_venv_interpreter() {
  local venv="$1"
  local image_python="$2"
  local py="${venv}/bin/python"
  local home
  if [[ -x "${py}" ]]; then
    return 0
  fi
  if ! lerobot_dlc_venv_has_lerobot_files "${venv}"; then
    return 1
  fi
  lerobot_dlc_log "relinking ${py} -> ${image_python} (site-packages present, interpreter missing in this container)"
  mkdir -p "${venv}/bin"
  ln -sfn "${image_python}" "${py}"
  ln -sfn python "${venv}/bin/python3"
  home="$(dirname "${image_python}")"
  if [[ -f "${venv}/pyvenv.cfg" ]]; then
    if grep -q '^home =' "${venv}/pyvenv.cfg"; then
      sed -i "s|^home = .*|home = ${home}|" "${venv}/pyvenv.cfg"
    else
      echo "home = ${home}" >> "${venv}/pyvenv.cfg"
    fi
  fi
  lerobot_dlc_enable_system_site_packages "${venv}"
}

lerobot_dlc_venv_usable() {
  local venv="$1"
  local py="${venv}/bin/python"
  local out
  if [[ ! -d "${venv}" ]]; then
    lerobot_dlc_log "skip ${venv}: directory missing"
    return 1
  fi
  if [[ ! -x "${py}" ]]; then
    lerobot_dlc_log "skip ${venv}: interpreter not executable ($(readlink "${py}" 2>/dev/null || echo missing))"
    return 1
  fi
  if ! out="$("${py}" -c 'import numpy as np
import torch
from importlib.util import find_spec
if find_spec("lerobot") is None:
    raise SystemExit("lerobot missing")
version = torch.__version__
if not version.startswith("2.7"):
    raise SystemExit(f"torch {version} (need 2.7.x) from {torch.__file__}")
torch.from_numpy(np.zeros((1,), dtype=np.float32))
print(
    f"lerobot+torch ok {version} {torch.__file__} numpy {np.__version__} {np.__file__}"
)' 2>&1)"; then
    lerobot_dlc_log "skip ${venv}: ${out}"
    return 1
  fi
  lerobot_dlc_log "reusing ${venv}: ${out}"
  return 0
}

lerobot_dlc_strip_venv_numpy() {
  # Drop a venv-local NumPy 2.x so --system-site-packages sees the image 1.x.
  local venv="$1"
  local sp removed
  sp="$(find "${venv}/lib" -maxdepth 3 -type d -name site-packages -print -quit 2>/dev/null || true)"
  if [[ -z "${sp}" ]]; then
    return 1
  fi
  removed=0
  if [[ -d "${sp}/numpy" || -d "${sp}/numpy.libs" ]] || compgen -G "${sp}/numpy-*.dist-info" >/dev/null 2>&1; then
    lerobot_dlc_log "removing venv numpy from ${sp} (image torch needs NumPy 1.x)"
    rm -rf "${sp}/numpy" "${sp}/numpy.libs" "${sp}"/numpy-*.dist-info
    removed=1
  fi
  if [[ "${removed}" -eq 1 ]]; then
    return 0
  fi
  return 1
}

lerobot_dlc_repair_venv_numpy() {
  local venv="$1"
  local py="${venv}/bin/python"
  if [[ ! -x "${py}" ]]; then
    return 1
  fi
  if "${py}" -c 'import numpy as np
import torch
torch.from_numpy(np.zeros((1,), dtype=np.float32))' >/dev/null 2>&1; then
    return 0
  fi
  if lerobot_dlc_strip_venv_numpy "${venv}"; then
    if "${py}" -c 'import numpy as np
import torch
torch.from_numpy(np.zeros((1,), dtype=np.float32))
print(f"repaired numpy {np.__version__} {np.__file__}")'; then
      return 0
    fi
  fi
  return 1
}

lerobot_dlc_ensure_uv() {
  local nas_uv="${CODE_DIR:?CODE_DIR is required}/.cache/uv-dlc/bin/uv"
  export PATH="${CODE_DIR}/.cache/uv-dlc/bin:${HOME}/.local/bin:${PATH}"
  hash -r 2>/dev/null || true
  if command -v uv >/dev/null 2>&1; then
    lerobot_dlc_log "using uv $(command -v uv) ($(uv --version 2>/dev/null || echo version-unknown))"
    if [[ ! -x "${nas_uv}" ]]; then
      mkdir -p "$(dirname "${nas_uv}")"
      cp -a "$(command -v uv)" "${nas_uv}" 2>/dev/null || true
    fi
    return 0
  fi
  lerobot_dlc_log "installing uv (not on PATH or NAS ${nas_uv})"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  hash -r 2>/dev/null || true
  mkdir -p "$(dirname "${nas_uv}")"
  if command -v uv >/dev/null 2>&1; then
    cp -a "$(command -v uv)" "${nas_uv}"
  fi
}

lerobot_dlc_run_uv_sync() {
  env -u UV_NO_SYNC uv sync --frozen --inexact --extra dataset --extra training "$@"
}

lerobot_dlc_sync_venv_dlc() {
  local venv="$1"
  local image_python="$2"
  local attempt max_attempts
  local -a uv_skip_args=() uv_sync_extra=()
  local cache_has_wheels=0
  lerobot_dlc_ensure_uv
  lerobot_dlc_write_uv_toml
  export UV_PROJECT_ENVIRONMENT="${venv}"
  # copy-on-NAS took 175 min for 73 packages on dlc1myilu2h2pxn3. Same-FS hardlink is the default.
  export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${CODE_DIR}/.cache/uv-dlc}"
  export UV_PYTHON="${UV_PYTHON:-${image_python}}"
  export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
  export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
  export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
  export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  mkdir -p "${UV_CACHE_DIR}"
  mapfile -t uv_skip_args < <(lerobot_dlc_uv_skip_torch_args)
  if [[ -d "${UV_CACHE_DIR}/wheels-v6" ]]; then
    cache_has_wheels=1
  fi
  if [[ "${FORCE_UV_SYNC:-0}" == "1" ]]; then
    lerobot_dlc_log "FORCE_UV_SYNC=1, removing ${venv}"
    rm -rf "${venv}"
  fi
  if [[ ! -x "${venv}/bin/python" ]]; then
    lerobot_dlc_log "creating ${venv} with system-site-packages (python=${UV_PYTHON})"
    uv venv --python "${UV_PYTHON}" --system-site-packages "${venv}"
  else
    lerobot_dlc_log "keeping existing ${venv}; not rm -rf"
  fi
  lerobot_dlc_enable_system_site_packages "${venv}"
  lerobot_dlc_log "syncing because ${LEROBOT_DLC_SYNC_REASON:-venv missing or probe failed} -> ${venv} (link=${UV_LINK_MODE}, cache=${UV_CACHE_DIR}, skip torch/numpy/nvidia)"
  if [[ "${UV_OFFLINE:-0}" == "1" ]]; then
    uv_sync_extra+=(--offline)
    export UV_OFFLINE=1
    lerobot_dlc_log "UV_OFFLINE=1: uv sync will not touch the network"
  elif [[ "${cache_has_wheels}" -eq 1 ]]; then
    lerobot_dlc_log "trying uv sync --offline from ${UV_CACHE_DIR} (no PyPI)"
    if lerobot_dlc_run_uv_sync "${uv_skip_args[@]}" --offline; then
      lerobot_dlc_enable_system_site_packages "${venv}"
      lerobot_dlc_strip_venv_numpy "${venv}" || true
      return 0
    fi
    lerobot_dlc_log "offline cache incomplete; retrying with network (still skip torch/numpy/nvidia)"
  fi
  max_attempts="${UV_SYNC_ATTEMPTS:-3}"
  for attempt in $(seq 1 "${max_attempts}"); do
    if lerobot_dlc_run_uv_sync "${uv_skip_args[@]}" "${uv_sync_extra[@]}"; then
      lerobot_dlc_enable_system_site_packages "${venv}"
      lerobot_dlc_strip_venv_numpy "${venv}" || true
      return 0
    fi
    if [[ "${attempt}" -ge "${max_attempts}" ]]; then
      echo "[error] uv sync failed after ${max_attempts} attempts" >&2
      return 1
    fi
    lerobot_dlc_log "uv sync failed, retry $((attempt + 1))/${max_attempts} in 15s"
    sleep 15
  done
}

lerobot_dlc_activate_venv() {
  local venv="$1"
  export VIRTUAL_ENV="${venv}"
  export UV_PROJECT_ENVIRONMENT="${venv}"
  PYTHON="${venv}/bin/python"
  export PYTHON
  export PATH="${venv}/bin:${PATH}"
}

lerobot_dlc_bootstrap() {
  local venv_dlc selected candidate image_python
  local -a candidates=()
  export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
  # 提交机上的 127.0.0.1:7890 代理不要带进容器。
  if [[ "${HTTP_PROXY:-}${HTTPS_PROXY:-}${http_proxy:-}${https_proxy:-}" == *"127.0.0.1:7890"* ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  fi
  if [[ -n "${LEROBOT_VENV:-}" ]]; then
    LEROBOT_VENV="$(lerobot_dlc_nas_path "${LEROBOT_VENV}")"
    export LEROBOT_VENV
  fi
  venv_dlc="${CODE_DIR:?CODE_DIR is required}/.venv-dlc"
  image_python="$(lerobot_dlc_image_python)"
  lerobot_dlc_log "bootstrap FORCE_UV_SYNC=${FORCE_UV_SYNC:-0} SKIP_UV_SYNC=${SKIP_UV_SYNC:-0} UV_OFFLINE=${UV_OFFLINE:-0} LEROBOT_VENV=${LEROBOT_VENV:-}"
  lerobot_dlc_log "image interpreter ${image_python}"
  "${image_python}" - <<'PY'
import sys
try:
    import numpy as np
    import torch
    print("image python", sys.version.split()[0], "torch", torch.__version__, "file", torch.__file__, "numpy", np.__version__, np.__file__)
except Exception as exc:
    print("image python", sys.version.split()[0], "torch/numpy unavailable:", exc)
PY

  if [[ -n "${LEROBOT_VENV:-}" ]]; then
    candidates+=("${LEROBOT_VENV}")
  fi
  if [[ ${#candidates[@]} -eq 0 || "${candidates[0]}" != "${venv_dlc}" ]]; then
    candidates+=("${venv_dlc}")
  fi

  if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
    lerobot_dlc_log "SKIP_UV_SYNC=1: no curl uv, no uv sync, no PyPI (probe fail => job fail)"
  fi

  selected=""
  if [[ "${FORCE_UV_SYNC:-0}" == "1" && "${SKIP_UV_SYNC:-0}" == "1" ]]; then
    lerobot_dlc_log "FORCE_UV_SYNC=1 ignored because SKIP_UV_SYNC=1 (training jobs do not reinstall)"
  fi
  if [[ "${FORCE_UV_SYNC:-0}" == "1" && "${SKIP_UV_SYNC:-0}" != "1" ]]; then
    lerobot_dlc_log "FORCE_UV_SYNC=1, skip reuse (will recreate .venv-dlc)"
    LEROBOT_DLC_SYNC_REASON="FORCE_UV_SYNC=1"
  else
    for candidate in "${candidates[@]}"; do
      lerobot_dlc_describe_venv "${candidate}"
      if [[ "${candidate}" == "${venv_dlc}" && ! -x "${candidate}/bin/python" ]]; then
        lerobot_dlc_relink_venv_interpreter "${candidate}" "${image_python}" || true
      fi
      lerobot_dlc_repair_venv_numpy "${candidate}" || true
      if lerobot_dlc_venv_usable "${candidate}"; then
        selected="${candidate}"
        break
      fi
      LEROBOT_DLC_SYNC_REASON="probe failed for ${candidate}"
    done
  fi

  if [[ -n "${selected}" ]]; then
    lerobot_dlc_activate_venv "${selected}"
    export UV_NO_SYNC=1
    lerobot_dlc_log "reusing ${selected} (no uv install, no uv sync, no PyPI)"
  else
    if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
      echo "[error] no reusable venv (need import lerobot + torch 2.7.x + torch.from_numpy) and SKIP_UV_SYNC=1" >&2
      return 1
    fi
    LEROBOT_DLC_SYNC_REASON="${LEROBOT_DLC_SYNC_REASON:-venv missing or probe failed}"
    export LEROBOT_DLC_SYNC_REASON
    # 只往 .venv-dlc 写包，绝不 uv sync 提交机 .venv。
    lerobot_dlc_log "syncing because ${LEROBOT_DLC_SYNC_REASON}"
    lerobot_dlc_sync_venv_dlc "${venv_dlc}" "${image_python}"
    lerobot_dlc_repair_venv_numpy "${venv_dlc}" || true
    if ! lerobot_dlc_venv_usable "${venv_dlc}"; then
      echo "[error] ${venv_dlc} still unusable after sync" >&2
      return 1
    fi
    lerobot_dlc_activate_venv "${venv_dlc}"
  fi

  "${PYTHON}" - <<PY
import numpy as np
import torch
from importlib.util import find_spec

require_cuda = "${REQUIRE_CUDA:-1}" == "1"
if require_cuda:
    assert torch.cuda.is_available(), "CUDA is required"
    print("gpu", torch.cuda.get_device_name(), "torch", torch.__version__, "file", torch.__file__)
else:
    print("cuda check skipped", "torch", torch.__version__, "file", torch.__file__)
version = torch.__version__
assert version.startswith("2.7"), f"DLC must reuse image torch 2.7.x, got {version} from {torch.__file__}"
torch.from_numpy(np.zeros((1,), dtype=np.float32))
if find_spec("torchvision") is None:
    raise SystemExit("torchvision missing; image system-site-packages not visible")
if find_spec("lerobot") is None:
    raise SystemExit("lerobot missing after bootstrap")
print("torchvision ok, lerobot ok, numpy", np.__version__, np.__file__)
PY
}

lerobot_dlc_main_bootstrap() {
  lerobot_dlc_bootstrap
  lerobot_dlc_log "BOOTSTRAP_OK python=${PYTHON} venv=${VIRTUAL_ENV:-}"
}

lerobot_dlc_prepare_v3() {
  local raw="${ROBOCASA_RAW_DATASET:?ROBOCASA_RAW_DATASET is required}"
  local dst="${ROBOCASA_V3_DATASET:?ROBOCASA_V3_DATASET is required}"
  local convert_script="${CODE_DIR}/scripts/dlc/convert_robocasa_v21_to_v30.sh"
  lerobot_dlc_log "prepare v3 dataset dst=${dst}"
  PYTHON="$(lerobot_dlc_python)" bash "${convert_script}" "${raw}" "${dst}" "${ROBOCASA_REPO_ID}"
  local version
  version="$(lerobot_dlc_codebase_version "${dst}")"
  if [[ "${version}" != "v3.0" ]]; then
    echo "[error] converted dataset is ${version}, expected v3.0: ${dst}" >&2
    return 1
  fi
}

lerobot_dlc_resolve_action_dim() {
  local root="${ROBOCASA_V3_DATASET:-${ROBOCASA_RAW_DATASET}}"
  if [[ -n "${ACTION_DIM:-}" ]]; then
    echo "${ACTION_DIM}"
    return 0
  fi
  lerobot_dlc_feature_dim "${root}" action
}

lerobot_dlc_resolve_num_tasks() {
  local root="${ROBOCASA_V3_DATASET:-${ROBOCASA_RAW_DATASET}}"
  local total
  total="$(lerobot_dlc_total_tasks "${root}")"
  if [[ "${total}" -lt 2 ]]; then
    echo 2
  else
    echo "${total}"
  fi
}

lerobot_dlc_train_tokenizer() {
  local python output_dir action_dim
  python="$(lerobot_dlc_python)"
  output_dir="${TOKENIZER_PATH:-${RUN_ROOT}/tokenizer}"
  action_dim="$(lerobot_dlc_resolve_action_dim)"
  # tokenizer CLI 没有 FileExists 检查；保存时 mkdir(parents=True, exist_ok=True)。
  mkdir -p "${output_dir}" "${RUN_ROOT}/logs"
  lerobot_dlc_log "tokenizer repo=${ROBOCASA_REPO_ID} root=${ROBOCASA_V3_DATASET} action_dim=${action_dim} steps=${TOKENIZER_STEPS} batch=${TOKENIZER_BATCH_SIZE:-${BATCH_SIZE}} alignment=${ALIGNMENT_WEIGHT} decoder=${DECODER_TYPE}"
  "${python}" -m lerobot.scripts.lerobot_train_actioncodec_tokenizer --repo_id="${ROBOCASA_REPO_ID}" --root="${ROBOCASA_V3_DATASET}" --output_dir="${output_dir}" --action_dim="${action_dim}" --action_horizon=20 --latent_horizon=16 --codebook_size=1024 --num_codebooks=1 --alignment_weight="${ALIGNMENT_WEIGHT}" --decoder_type="${DECODER_TYPE}" --device=cuda --steps="${TOKENIZER_STEPS}" --batch_size="${TOKENIZER_BATCH_SIZE:-${BATCH_SIZE}}" --num_workers="${NUM_WORKERS}" --log_freq=10
  test -f "${output_dir}/model.safetensors"
  test -f "${output_dir}/model_config.json"
  test -f "${output_dir}/action_stats.json"
  test -f "${output_dir}/dataset_contract.json"
  "${python}" - "${output_dir}" "${action_dim}" "${DECODER_TYPE}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
want = int(sys.argv[2])
decoder = sys.argv[3]
config = json.loads((root / "model_config.json").read_text())
contract = json.loads((root / "dataset_contract.json").read_text())
assert config["action_dim"] == want, config["action_dim"]
assert contract["action_dim"] == want, contract["action_dim"]
assert config["horizon"] == 20 and config["latent_horizon"] == 16
assert config["codebook_size"] == 1024 and config["num_codebooks"] == 1
assert config["decoder_type"] == decoder, config["decoder_type"]
print("TOKENIZER_OK", root)
PY
}

lerobot_dlc_wandb_flags() {
  if [[ "${WANDB_ENABLE:-0}" == "1" ]]; then
    echo "--wandb.enable=true --wandb.mode=online --wandb.project=${WANDB_PROJECT:-lerobot-actioncodec}"
  else
    echo "--wandb.enable=false --wandb.mode=disabled"
  fi
}

lerobot_dlc_train_policy() {
  local python output_dir action_dim num_tasks tokenizer_path wandb_flags
  python="$(lerobot_dlc_python)"
  output_dir="${POLICY_OUTPUT_DIR:-${RUN_ROOT}/policy}"
  tokenizer_path="${TOKENIZER_PATH:?TOKENIZER_PATH is required}"
  action_dim="$(lerobot_dlc_resolve_action_dim)"
  num_tasks="$(lerobot_dlc_resolve_num_tasks)"
  wandb_flags="$(lerobot_dlc_wandb_flags)"
  # 只建父目录和 logs。不要 mkdir "${output_dir}"：空的 policy/ 也会让
  # lerobot-train 在 resume=False 时抛 FileExistsError（dlc1kwqe4qhqshsn）。
  mkdir -p "$(dirname "${output_dir}")" "${RUN_ROOT}/logs"
  test -f "${tokenizer_path}/model.safetensors"
  lerobot_dlc_log "policy repo=${ROBOCASA_REPO_ID} root=${ROBOCASA_V3_DATASET} action_dim=${action_dim} num_tasks=${num_tasks} tokenizer=${tokenizer_path} steps=${POLICY_STEPS}"
  # UserCommand 里不要用反斜杠续行；这里已经是单行。
  # shellcheck disable=SC2086
  "${python}" -m lerobot.scripts.lerobot_train --dataset.repo_id="${ROBOCASA_REPO_ID}" --dataset.root="${ROBOCASA_V3_DATASET}" --policy.type=actioncodec --policy.action_dim="${action_dim}" --policy.num_tasks="${num_tasks}" --policy.tokenizer_path="${tokenizer_path}" --policy.push_to_hub=false --policy.device=cuda --output_dir="${output_dir}" --job_name="actioncodec-${ROBOCASA_TASK:-CloseDrawer}" --steps="${POLICY_STEPS}" --batch_size="${BATCH_SIZE}" --num_workers="${NUM_WORKERS}" --log_freq=10 --save_freq="${POLICY_STEPS}" --env_eval_freq=0 --save_checkpoint=true --ema.enable=true --ema.power=0.75 --ema.max_decay=0.9999 --accelerator.mixed_precision=bf16 ${wandb_flags}
  test -d "${output_dir}/checkpoints/last/pretrained_model"
  lerobot_dlc_log "POLICY_OK ${output_dir}/checkpoints/last/pretrained_model"
}

lerobot_dlc_main_convert() {
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_v3
  lerobot_dlc_log "CONVERT_OK $(lerobot_dlc_codebase_version "${ROBOCASA_V3_DATASET}") action_dim=$(lerobot_dlc_resolve_action_dim)"
}

lerobot_dlc_main_tokenizer() {
  export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_v3
  lerobot_dlc_train_tokenizer
}

lerobot_dlc_main_policy() {
  export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_v3
  lerobot_dlc_train_policy
}

lerobot_dlc_main_pipeline() {
  export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_v3
  lerobot_dlc_train_tokenizer
  lerobot_dlc_train_policy
}

lerobot_dlc_bind_atomic4() {
  ATOMIC4_MERGED="${ATOMIC4_MERGED:-${CODE_DIR}/outputs/datasets/robocasa_atomic4_v3}"
  ATOMIC4_REPO_ID="${ATOMIC4_REPO_ID:-robocasa/atomic4}"
  ROBOCASA_TASK="${ROBOCASA_TASK:-atomic4}"
  ROBOCASA_REPO_ID="${ATOMIC4_REPO_ID}"
  ROBOCASA_V3_DATASET="${ATOMIC4_MERGED}"
  export ATOMIC4_MERGED ATOMIC4_REPO_ID ROBOCASA_TASK ROBOCASA_REPO_ID ROBOCASA_V3_DATASET
}

lerobot_dlc_prepare_atomic4() {
  local convert_script="${CODE_DIR}/scripts/dlc/convert_robocasa_atomic4.sh"
  lerobot_dlc_bind_atomic4
  lerobot_dlc_log "prepare atomic4 merged dst=${ATOMIC4_MERGED}"
  PYTHON="$(lerobot_dlc_python)" CODE_DIR="${CODE_DIR}" bash "${convert_script}"
  local version
  version="$(lerobot_dlc_codebase_version "${ATOMIC4_MERGED}")"
  if [[ "${version}" != "v3.0" ]]; then
    echo "[error] merged dataset is ${version}, expected v3.0: ${ATOMIC4_MERGED}" >&2
    return 1
  fi
  lerobot_dlc_bind_atomic4
}

lerobot_dlc_main_atomic4_convert() {
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_atomic4
  lerobot_dlc_log "CONVERT_OK $(lerobot_dlc_codebase_version "${ATOMIC4_MERGED}") action_dim=$(lerobot_dlc_resolve_action_dim) num_tasks=$(lerobot_dlc_resolve_num_tasks)"
}

lerobot_dlc_main_atomic4_tokenizer() {
  export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_atomic4
  lerobot_dlc_train_tokenizer
}

lerobot_dlc_main_atomic4_policy() {
  export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_atomic4
  lerobot_dlc_train_policy
}

lerobot_dlc_main_atomic4_pipeline() {
  export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
  lerobot_dlc_bootstrap
  lerobot_dlc_prepare_atomic4
  lerobot_dlc_train_tokenizer
  lerobot_dlc_train_policy
}
