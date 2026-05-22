#!/usr/bin/env bash
# ================================================================
#  YOLO 预训练权重下载脚本
# ================================================================
#
#  交互模式（直接运行）:
#      bash weights/download.sh
#      → 弹出菜单选择系列、规模、任务
#
#  命令行模式:
#      bash weights/download.sh yolo26n.pt              # 指定权重
#      bash weights/download.sh yolo26-detect           # 系列别名
#      bash weights/download.sh yolo26s.pt yolo11n.pt   # 多个
#
#  别名:
#      yolo26-detect / yolo26-segment
#      yolo11-detect / yolo11-segment
#      yolov8-detect / yolov8-segment
#      detect-all / segment-all / all
#
#  来源: https://github.com/ultralytics/assets/releases/tag/v8.4.0
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ULTRALYTICS_BASE="https://github.com/ultralytics/assets/releases/download/v8.4.0"

# 权重分组
YOLO26_DETECT=(yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt yolo26x.pt)
YOLO26_SEGMENT=(yolo26n-seg.pt yolo26s-seg.pt yolo26m-seg.pt yolo26l-seg.pt yolo26x-seg.pt)
YOLO11_DETECT=(yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt)
YOLO11_SEGMENT=(yolo11n-seg.pt yolo11s-seg.pt yolo11m-seg.pt yolo11l-seg.pt yolo11x-seg.pt)
YOLOV8_DETECT=(yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt)
YOLOV8_SEGMENT=(yolov8n-seg.pt yolov8s-seg.pt yolov8m-seg.pt yolov8l-seg.pt yolov8x-seg.pt)

# ============= 交互菜单 =============
interactive_menu() {
    echo "================================================"
    echo "  YOLO 预训练权重下载"
    echo "================================================"
    echo ""
    echo "请选择 YOLO 系列:"
    echo "  1) YOLO26 (推荐，2026 最新)"
    echo "  2) YOLO11"
    echo "  3) YOLOv8"
    echo ""
    read -rp "选择 [1-3，默认 1]: " series_choice
    series_choice="${series_choice:-1}"

    case "$series_choice" in
        1) series="yolo26" ;;
        2) series="yolo11" ;;
        3) series="yolov8" ;;
        *) echo "无效选择: $series_choice"; exit 1 ;;
    esac

    echo ""
    echo "请选择任务类型:"
    echo "  1) 检测 (detect, 边界框)"
    echo "  2) 分割 (segment, 掩码)"
    echo ""
    read -rp "选择 [1-2，默认 1]: " task_choice
    task_choice="${task_choice:-1}"

    case "$task_choice" in
        1) task="detect" ;;
        2) task="segment" ;;
        *) echo "无效选择: $task_choice"; exit 1 ;;
    esac

    echo ""
    echo "请选择模型规模 (从小到大):"
    echo "  1) n - nano   (最小，速度优先)"
    echo "  2) s - small  (默认推荐)"
    echo "  3) m - medium"
    echo "  4) l - large"
    echo "  5) x - xlarge (最大，精度优先)"
    echo "  6) 全部下载（n/s/m/l/x）"
    echo ""
    read -rp "选择 [1-6，默认 1]: " size_choice
    size_choice="${size_choice:-1}"

    case "$size_choice" in
        1) sizes=(n) ;;
        2) sizes=(s) ;;
        3) sizes=(m) ;;
        4) sizes=(l) ;;
        5) sizes=(x) ;;
        6) sizes=(n s m l x) ;;
        *) echo "无效选择: $size_choice"; exit 1 ;;
    esac

    # 拼装文件名
    suffix=""
    if [[ "$task" == "segment" ]]; then
        suffix="-seg"
    fi

    SELECTED=()
    for sz in "${sizes[@]}"; do
        SELECTED+=("${series}${sz}${suffix}.pt")
    done

    echo ""
    echo "================================================"
    echo "将下载以下权重:"
    for w in "${SELECTED[@]}"; do
        echo "  • $w"
    done
    echo "================================================"
    read -rp "确认下载？[Y/n]: " confirm
    confirm="${confirm:-Y}"
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "取消下载"
        exit 0
    fi
}

# ============= 别名展开 =============
expand_aliases() {
    local result=()
    for arg in "$@"; do
        case "$arg" in
            yolo26-detect)   result+=("${YOLO26_DETECT[@]}") ;;
            yolo26-segment)  result+=("${YOLO26_SEGMENT[@]}") ;;
            yolo11-detect)   result+=("${YOLO11_DETECT[@]}") ;;
            yolo11-segment)  result+=("${YOLO11_SEGMENT[@]}") ;;
            yolov8-detect)   result+=("${YOLOV8_DETECT[@]}") ;;
            yolov8-segment)  result+=("${YOLOV8_SEGMENT[@]}") ;;
            detect-all)      result+=("${YOLO26_DETECT[@]}" "${YOLO11_DETECT[@]}" "${YOLOV8_DETECT[@]}") ;;
            segment-all)     result+=("${YOLO26_SEGMENT[@]}" "${YOLO11_SEGMENT[@]}" "${YOLOV8_SEGMENT[@]}") ;;
            all)             result+=("${YOLO26_DETECT[@]}" "${YOLO26_SEGMENT[@]}" "${YOLO11_DETECT[@]}" "${YOLO11_SEGMENT[@]}" "${YOLOV8_DETECT[@]}" "${YOLOV8_SEGMENT[@]}") ;;
            *)               result+=("$arg") ;;
        esac
    done
    SELECTED=("${result[@]}")
}

# ============= 下载工具 =============
if command -v wget > /dev/null; then
    DOWNLOAD_CMD() { wget -q --show-progress -O "$1" "$2"; }
elif command -v curl > /dev/null; then
    DOWNLOAD_CMD() { curl -L --progress-bar -o "$1" "$2"; }
else
    echo "错误: 需要 wget 或 curl"
    exit 1
fi

# ============= 入口 =============
SELECTED=()
if [[ $# -eq 0 ]]; then
    interactive_menu
else
    expand_aliases "$@"
fi

echo ""
echo "下载到: $SCRIPT_DIR"
echo "权重数: ${#SELECTED[@]}"
echo ""

# ============= 执行下载 =============
FAILED=()
for weight in "${SELECTED[@]}"; do
    if [[ -f "$weight" ]]; then
        size=$(du -h "$weight" | cut -f1)
        echo "✓ $weight 已存在 ($size)"
        continue
    fi

    url="$ULTRALYTICS_BASE/$weight"
    echo "↓ $weight"
    echo "  $url"

    if DOWNLOAD_CMD "$weight" "$url"; then
        if [[ -s "$weight" ]]; then
            size=$(du -h "$weight" | cut -f1)
            echo "✓ $weight ($size)"
        else
            echo "✗ $weight 下载为空文件"
            rm -f "$weight"
            FAILED+=("$weight")
        fi
    else
        echo "✗ $weight 下载失败"
        rm -f "$weight"
        FAILED+=("$weight")
    fi
    echo ""
done

echo "========================================"
echo "已有权重:"
ls -lh "$SCRIPT_DIR"/*.pt 2>/dev/null | awk '{print "  " $5 "  " $9}'

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "失败的权重:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
