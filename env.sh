#!/usr/bin/env bash
# 项目根目录自动检测：支持 source env.sh 和 bash env.sh 两种方式。
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    _ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
export PROJECT_ROOT="$_ENV_DIR"

export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# CUDA：优先使用已有的 CUDA_HOME，否则尝试自动检测。
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME="/usr/local/cuda"
    elif compgen -G "/usr/local/cuda-*" > /dev/null 2>&1; then
        export CUDA_HOME="$(ls -d /usr/local/cuda-* | sort -V | tail -1)"
    fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    _CUDA_LIBS="$CUDA_HOME/targets/aarch64-linux/lib:$CUDA_HOME/lib64"
    # 检测 venv 中的 nvidia 库（不绑定具体 Python 版本）
    _NVIDIA_LIB="$(find "$VIRTUAL_ENV/lib" -path '*/nvidia/cu*/lib' -type d 2>/dev/null | head -1)"
    if [[ -n "$_NVIDIA_LIB" ]]; then
        _CUDA_LIBS="$_CUDA_LIBS:$_NVIDIA_LIB"
    fi
    export LD_LIBRARY_PATH="$_CUDA_LIBS:${LD_LIBRARY_PATH:-}"
fi

export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
