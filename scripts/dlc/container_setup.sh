#!/usr/bin/env bash
# DLC 容器内启动：设置 MuJoCo headless EGL。必须 source：
#   source scripts/dlc/container_setup.sh
#
# 当前镜像已带 Python 3.12 + torch 2.7.0 + cu128。不要在这里 pip/uv 装 torch。
#
# NVIDIA 图形库靠任务创建时的 NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# 挂进容器。这些 .so 往往是绑定挂载（或落在镜像只读层上），与容器可写层不在同一文件系统。
# 再 apt-get install libegl1/libgles2 时 dpkg 无法做跨设备硬链接备份，报
# "Invalid cross-device link"。不要用 --force-overwrite / purge 去覆盖 NVIDIA 挂载。

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

egl_lib_present() {
  local libdir=/usr/lib/x86_64-linux-gnu
  [[ -e "$libdir/libEGL.so.1" ]] \
    || [[ -e "$libdir/libEGL.so.1.1.0" ]] \
    || [[ -e "$libdir/libEGL_nvidia.so.0" ]] \
    || [[ -e "$libdir/libGLESv2.so.2" ]] \
    || [[ -e "$libdir/libGLESv2.so.2.1.0" ]]
}

glx_lib_present() {
  [[ -e /usr/lib/x86_64-linux-gnu/libGLX.so.0 ]] \
    || [[ -e /lib/x86_64-linux-gnu/libGLX.so.0 ]]
}

gl_dispatch_present() {
  [[ -e /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 ]] \
    || [[ -e /lib/x86_64-linux-gnu/libGLdispatch.so.0 ]]
}

install_cmake_if_needed() {
  if command -v cmake >/dev/null 2>&1; then
    echo "[egl] cmake already present: $(command -v cmake)"
    return 0
  fi
  echo "[egl] installing cmake"
  apt-get update
  apt-get install -y --no-install-recommends cmake
}

if egl_lib_present; then
  echo "[egl] libEGL/libGLESv2 already present (NVIDIA mount or image layer); skip apt libegl*"
  echo "[egl] existing EGL/GLES libs:"
  ls -l /usr/lib/x86_64-linux-gnu/libEGL* /usr/lib/x86_64-linux-gnu/libGLES* 2>/dev/null || true
  install_cmake_if_needed
else
  echo "[egl] libEGL missing; installing GLVND/EGL loader packages"
  apt-get update
  apt-get install -y --no-install-recommends \
    cmake \
    libegl1 \
    libegl-dev \
    libgles2 \
    libgl1 \
    libglvnd0
fi

# A mounted libEGL.so loader is not sufficient by itself: GLVND's loader has
# a runtime dependency on libGLdispatch.so.0. Without it, ctypes cannot load
# libEGL and PyOpenGL reports PLATFORM.EGL=None / missing eglQueryString.
# Install only libglvnd0 here; unlike libegl1, it does not overwrite the
# NVIDIA-mounted EGL libraries and therefore avoids dpkg cross-device errors.
if gl_dispatch_present; then
  echo "[egl] libGLdispatch.so.0 already present"
else
  echo "[egl] libGLdispatch.so.0 missing; installing libglvnd0"
  apt-get update
  apt-get install -y --no-install-recommends libglvnd0
  gl_dispatch_present || {
    echo "[egl] libGLdispatch.so.0 is still missing after setup" >&2
    return 1
  }
fi

# OpenCV's GUI wheel imports libGLX; this pipeline uses opencv-python-headless
# so GLX is not required. Installing libglx0 pulls mesa/libllvm (~40MB) on every
# fresh container. Set INSTALL_LIBGLX=1 only if you need the GUI OpenCV stack.
if glx_lib_present; then
  echo "[egl] libGLX.so.0 already present"
elif [[ "${INSTALL_LIBGLX:-0}" == "1" ]]; then
  echo "[egl] INSTALL_LIBGLX=1; installing libglx0"
  apt-get update
  apt-get install -y --no-install-recommends libglx0
  glx_lib_present || {
    echo "[egl] libGLX.so.0 is still missing after setup" >&2
    return 1
  }
else
  echo "[egl] libGLX.so.0 missing; skip apt (opencv-python-headless). Set INSTALL_LIBGLX=1 to install."
fi

nvidia_icd="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [[ ! -f "$nvidia_icd" && -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]]; then
  mkdir -p "$(dirname "$nvidia_icd")"
  cat > "$nvidia_icd" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
JSON
  echo "[egl] wrote $nvidia_icd"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -f "$nvidia_icd" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$nvidia_icd}"
fi
if [[ -n "${DISPLAY:-}" ]]; then
  echo "[egl] unsetting DISPLAY=${DISPLAY} for headless EGL"
  unset DISPLAY
fi

echo "[egl] MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "[egl] NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
echo "[egl] __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-unset}"
if [[ -d /usr/share/glvnd/egl_vendor.d ]]; then
  echo "[egl] glvnd vendors:"
  ls -l /usr/share/glvnd/egl_vendor.d
else
  echo "[egl] warning: /usr/share/glvnd/egl_vendor.d is missing"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi
