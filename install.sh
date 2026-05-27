#!/usr/bin/env bash
# ================================================================
#  YOLO Pipeline 一键安装
# ================================================================
#
#  用法:
#      bash install.sh                    # 自动检测平台并安装
#      bash install.sh --jetson           # 强制 Jetson
#      bash install.sh --cuda12           # 强制 x86_64 CUDA 12
#      bash install.sh --cuda118          # 强制 x86_64 CUDA 11.8
#      bash install.sh --cpu              # 强制 CPU
#      bash install.sh --skip-torch       # 只装通用依赖，不装 torch
#      bash install.sh --torch-wheel /path/to/torch-2.x.whl   # 用本地 torch wheel
#      bash install.sh --offline          # 完全离线安装（所有包都从 cache）
#
#  网络不好怎么办:
#      1. 先在能联网的机器上跑一遍 install.sh，让 uv cache 缓存好
#      2. 把 ~/.cache/uv 复制到目标机器同样位置
#      3. 在目标机器跑 bash install.sh --offline
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============= 解析参数 =============
PLATFORM=""
SKIP_TORCH=0
OFFLINE=0
TORCH_WHEEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jetson)        PLATFORM="jetson"; shift ;;
        --cuda12)        PLATFORM="cuda12"; shift ;;
        --cuda118)       PLATFORM="cuda118"; shift ;;
        --cpu)           PLATFORM="cpu"; shift ;;
        --skip-torch)    SKIP_TORCH=1; shift ;;
        --offline)       OFFLINE=1; shift ;;
        --torch-wheel)   TORCH_WHEEL="$2"; shift 2 ;;
        --help|-h)       grep "^#" "$0" | head -25; exit 0 ;;
        *)               echo "未知参数: $1"; exit 1 ;;
    esac
done

# ============= 检查 uv =============
if ! command -v uv > /dev/null; then
    echo "❌ uv 未安装。安装方法："
    echo ""
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "   国内无法访问 astral.sh 的话，可以用 pipx 或 pip:"
    echo "   pip install --user uv"
    echo ""
    echo "   装完后重启终端或 source ~/.bashrc 让 uv 进 PATH"
    exit 1
fi

# ============= 自动检测平台 =============
detect_platform() {
    local arch="$(uname -m)"

    if [[ "$arch" == "aarch64" ]]; then
        if [[ -f /etc/nv_tegra_release ]]; then
            echo "jetson"; return
        fi
        echo "cpu"; return
    fi

    if [[ "$arch" == "x86_64" ]]; then
        if command -v nvidia-smi > /dev/null 2>&1; then
            local cuda_version="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)"
            if [[ -n "$cuda_version" ]]; then
                local major="$(echo "$cuda_version" | cut -d. -f1)"
                if [[ "$major" -ge 12 ]]; then echo "cuda12"; return
                elif [[ "$major" == "11" ]]; then echo "cuda118"; return
                fi
            fi
        fi
        echo "cpu"; return
    fi

    echo "cpu"
}

if [[ -z "$PLATFORM" ]]; then
    PLATFORM="$(detect_platform)"
    echo "🔍 自动检测平台: $PLATFORM"
else
    echo "▶ 使用指定平台: $PLATFORM"
fi

# ============= 平台 → torch 安装源（多个 fallback）=============
# 每个平台都列多个备选源，自动尝试直到成功
declare -a TORCH_SOURCES
declare -a TORCH_PACKAGES
case "$PLATFORM" in
    jetson)
        # JetPack 6 / L4T R36.5 on this project targets CUDA 12.6.  Keep
        # torch pinned, otherwise PyPI may resolve a newer aarch64 cu13 wheel.
        TORCH_PACKAGES=("torch==2.8.0" "torchvision==0.23.0")
        TORCH_SOURCES=(
            "https://pypi.jetson-ai-lab.io/jp6/cu126"
        )
        ;;
    cuda12)
        TORCH_PACKAGES=("torch" "torchvision")
        TORCH_SOURCES=(
            "https://download.pytorch.org/whl/cu124"
            "https://mirror.sjtu.edu.cn/pytorch-wheels/cu124"
            "https://mirrors.aliyun.com/pytorch-wheels/cu124"
        )
        ;;
    cuda118)
        TORCH_PACKAGES=("torch" "torchvision")
        TORCH_SOURCES=(
            "https://download.pytorch.org/whl/cu118"
            "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118"
            "https://mirrors.aliyun.com/pytorch-wheels/cu118"
        )
        ;;
    cpu)
        TORCH_PACKAGES=("torch" "torchvision")
        TORCH_SOURCES=(
            "https://download.pytorch.org/whl/cpu"
            "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"
            "https://mirrors.aliyun.com/pytorch-wheels/cpu"
        )
        ;;
    *)
        echo "❌ 未知平台: $PLATFORM"; exit 1 ;;
esac

# ============= 创建 venv =============
if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo ""
    echo "📦 创建虚拟环境 (.venv)..."
    uv venv --python 3.10
else
    echo "✓ .venv 已存在，跳过创建"
fi

PIP_OFFLINE=""
if [[ "$OFFLINE" == "1" ]]; then
    PIP_OFFLINE="--offline"
fi

CONSTRAINT_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINT_FILE"' EXIT
if [[ "$SKIP_TORCH" != "1" && -z "$TORCH_WHEEL" ]]; then
    printf "%s\n" "${TORCH_PACKAGES[@]}" "numpy<2.0" > "$CONSTRAINT_FILE"
else
    printf "numpy<2.0\n" > "$CONSTRAINT_FILE"
fi

# ============= 装 torch =============
if [[ "$SKIP_TORCH" == "1" ]]; then
    echo ""
    echo "⏭  --skip-torch: 跳过 torch"
elif [[ -n "$TORCH_WHEEL" ]]; then
    if [[ ! -f "$TORCH_WHEEL" ]]; then
        echo "❌ wheel 文件不存在: $TORCH_WHEEL"; exit 1
    fi
    echo ""
    echo "📦 从本地 wheel 安装 torch..."
    echo "   文件: $TORCH_WHEEL"
    uv pip install "$TORCH_WHEEL"
elif [[ "$OFFLINE" == "1" ]]; then
    echo ""
    echo "📦 离线安装 torch（从 cache）..."
    uv pip install --offline -c "$CONSTRAINT_FILE" "${TORCH_PACKAGES[@]}"
else
    echo ""
    echo "📦 安装 PyTorch ($PLATFORM)..."
    if [[ "$PLATFORM" == "jetson" ]]; then
        # 清理之前误装的 PyPI CUDA 13 包，避免 env.sh/project_env.py 加载到不匹配库。
        uv pip uninstall -y \
            cuda-bindings cuda-pathfinder cuda-toolkit triton \
            nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
            nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
            nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
            nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
            > /dev/null 2>&1 || true
    fi

    INSTALLED_TORCH=0
    for src in "${TORCH_SOURCES[@]}"; do
        echo "   尝试: $src"
        if uv pip install --index "$src" --index-strategy first-index -c "$CONSTRAINT_FILE" "${TORCH_PACKAGES[@]}" 2>&1 | tee /tmp/uv_install.log; then
            INSTALLED_TORCH=1
            break
        fi
        echo "   ✗ 失败，尝试下一个镜像..."
        echo ""
    done

    if [[ "$INSTALLED_TORCH" == "0" ]]; then
        echo ""
        echo "❌ 所有 torch 镜像源都失败。可能原因："
        echo "   1. 网络不通 / 代理问题（最常见）"
        echo "   2. DNS 劫持（试试 ping pypi.jetson-ai-lab.io 看 IP 是否真实）"
        echo "   3. TLS 中间人证书问题"
        echo ""
        echo "解决方案（任选其一）："
        echo ""
        echo "  A) 找另一台能联网的机器，从浏览器下载 torch wheel:"
        case "$PLATFORM" in
            jetson)
                echo "     https://pypi.jetson-ai-lab.io/jp6/cu126/torch/"
                echo "     https://pypi.jetson-ai-lab.io/jp6/cu126/torchvision/"
                echo "     选 torch-2.8.0 和 torchvision-0.23.0 的 cp310 linux_aarch64 wheel" ;;
            cuda12)
                echo "     https://download.pytorch.org/whl/cu124/torch/" ;;
            cuda118)
                echo "     https://download.pytorch.org/whl/cu118/torch/" ;;
            cpu)
                echo "     https://download.pytorch.org/whl/cpu/torch/" ;;
        esac
        echo "     然后: bash install.sh --torch-wheel /path/to/torch-*.whl"
        echo ""
        echo "  B) 如果 ~/.cache/uv 之前装过 torch，离线安装："
        echo "     bash install.sh --offline"
        echo ""
        echo "  C) 跳过 torch，先用项目其他部分："
        echo "     bash install.sh --skip-torch"
        exit 1
    fi
fi

# ============= 通用依赖 =============
echo ""
echo "📦 安装通用依赖 (requirements.txt)..."
if [[ "$OFFLINE" == "1" ]]; then
    echo "   离线模式：仅从 ~/.cache/uv 读取"
fi

uv pip install $PIP_OFFLINE -c "$CONSTRAINT_FILE" -r requirements.txt

# ============= 验证 =============
echo ""
echo "🧪 验证安装..."
"$SCRIPT_DIR/.venv/bin/python" - <<'EOF'
import sys
ok = True

def check(name, fn):
    global ok
    try: fn(); print(f"  ✓ {name}")
    except Exception as e: print(f"  ✗ {name}: {e}"); ok = False

check("ultralytics", lambda: __import__("ultralytics"))
check("opencv-python", lambda: __import__("cv2"))
check("numpy", lambda: __import__("numpy"))
check("pyyaml", lambda: __import__("yaml"))

try:
    import torch
    cuda = "✓ CUDA" if torch.cuda.is_available() else "CPU only"
    print(f"  ✓ torch {torch.__version__}  ({cuda})")
    if torch.cuda.is_available():
        print(f"    设备: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  ⚠ torch 未安装（用了 --skip-torch？）")
except Exception as e:
    print(f"  ✗ torch: {e}"); ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "========================================"
echo "✅ 安装完成"
echo "========================================"
echo ""
echo "下一步:"
echo "  source env.sh                           # 激活环境"
echo "  bash weights/download.sh                # 下载预训练权重"
echo "  # 编辑 train.sh 顶部 5 个变量"
echo "  bash train.sh                           # 开始训练"
