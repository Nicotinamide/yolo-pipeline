"""Project environment bootstrap — auto-detects CUDA and Python version."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "bin/python"


def _detect_cuda_home() -> Path | None:
    """Detect CUDA installation directory, preferring the latest version."""
    env_cuda = os.environ.get("CUDA_HOME")
    if env_cuda and Path(env_cuda).is_dir():
        return Path(env_cuda)

    if Path("/usr/local/cuda").is_dir():
        return Path("/usr/local/cuda")

    candidates = sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    if candidates:
        return Path(candidates[0])

    return None


def _detect_nvidia_venv_lib() -> Path | None:
    """Find nvidia CUDA lib inside the venv (any Python version, any cu version)."""
    pattern = str(VENV / "lib/python*/site-packages/nvidia/cu*/lib")
    matches = glob.glob(pattern)
    if matches:
        return Path(sorted(matches, reverse=True)[0])
    return None


def ensure_project_env() -> None:
    """Re-run the current script under the project venv and set up CUDA paths."""
    os.environ.setdefault("VIRTUAL_ENV", str(VENV))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

    # CUDA setup
    cuda_home = _detect_cuda_home()
    if cuda_home:
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        ld_paths: list[str] = []

        # 按架构选 CUDA 目标库：aarch64 (Jetson) 或 x86_64
        import platform
        arch = platform.machine()
        arch_to_subdir = {"aarch64": "aarch64-linux", "x86_64": "x86_64-linux"}
        arch_subdir = arch_to_subdir.get(arch)
        if arch_subdir:
            cuda_arch_lib = cuda_home / f"targets/{arch_subdir}/lib"
            if cuda_arch_lib.is_dir():
                ld_paths.append(str(cuda_arch_lib))

        cuda_lib64 = cuda_home / "lib64"
        if cuda_lib64.is_dir():
            ld_paths.append(str(cuda_lib64))

        # x86 PyTorch wheels ship CUDA libs in site-packages/nvidia.  Jetson
        # uses system CUDA/L4T libraries, so ignore stale pip CUDA libs there.
        if arch == "x86_64":
            nvidia_venv_lib = _detect_nvidia_venv_lib()
            if nvidia_venv_lib and nvidia_venv_lib.is_dir():
                ld_paths.append(str(nvidia_venv_lib))

        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        current_parts = [part for part in current_ld.split(":") if part]
        for lib_path in reversed(ld_paths):
            if lib_path not in current_parts:
                current_parts.insert(0, lib_path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(current_parts)

    # PATH setup
    venv_bin = str(VENV / "bin")
    path_parts = [part for part in os.environ.get("PATH", "").split(":") if part]
    os.environ["PATH"] = ":".join([venv_bin, *[part for part in path_parts if part != venv_bin]])

    # Re-exec under project venv if needed
    in_project_venv = Path(sys.prefix).resolve() == VENV.resolve()
    if VENV_PYTHON.exists() and not in_project_venv:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])
