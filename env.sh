# Source from anywhere: activates the robocasa venv and sets headless GPU rendering.
# The container has no libEGL_nvidia; NVGL points at the user-space GL libs extracted from the
# driver .run that matches `nvidia-smi` (same layout robot-ood uses).
RC_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RC_HOME
source "$RC_HOME/.venv/bin/activate"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
_NV_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
export NVGL="${NVGL:-$(cd "$RC_HOME/.." && pwd)/nvidia-gl-${_NV_DRIVER:-none}}"
if [ -f "$NVGL/lib/libEGL_nvidia.so.0" ]; then
  export __EGL_VENDOR_LIBRARY_DIRS="$NVGL/egl_vendor.d"
  export LD_LIBRARY_PATH="$NVGL/lib:${LD_LIBRARY_PATH:-}"
fi
# robosuite wants MUJOCO_EGL_DEVICE_ID inside CUDA_VISIBLE_DEVICES and then uses it as the EGL
# index; EGL order differs from CUDA order, so leave CUDA_VISIBLE_DEVICES unset and pick by EGL index.
unset CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID="${EGL_DEV:-0}"
export PYTHONUNBUFFERED=1
