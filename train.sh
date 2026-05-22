#!/usr/bin/env bash
# ================================================================
#  YOLO 训练脚本 - 改下面的配置就能跑
# ================================================================
#
#  用法:
#      bash train.sh
#
#  切换数据/模型/任务时，只需修改下方"配置区"的几个变量。
#  无关紧要的参数已经设好了智能默认值，按需打开即可。
#
#  对比实验：复制此文件，改 NAME 和其他变量后再运行。
#      cp train.sh train_v8.sh
#      # 编辑 train_v8.sh：PRETRAINED=yolo8n.pt, NAME=my_run_v8
#      bash train_v8.sh
# ================================================================

# ==================== 配置区 - 改这里 ====================

# 1. 数据来源（必填）
IMAGES="/path/to/your/images"          # 图片文件夹
LABELS="/path/to/your/labels"          # YOLO txt 标签文件夹

# 2. 任务类型: detect (检测框) 或 segment (分割掩码)
TASK="segment"

# 3. 运行名（决定 datasets/<NAME>、runs/<NAME> 目录名）
#    留空时自动取 IMAGES 文件夹的最后一级目录名
NAME=""

# 4. 预训练权重
PRETRAINED="weights/yolo26s-seg.pt"

# 5. 训练轮数
EPOCHS=150

# ==================== 可选配置（按需修改） ====================

IMGSZ=960          # 训练分辨率；小目标多就调大到 1280
BATCH=8            # 批大小；显存不够就调小
VAL_RATIO=0.2      # 验证集比例
DEVICE=""          # 留空=自动选 GPU/CPU；或填 "0" / "cpu"
AUTO_PREDICT=1     # 1=训练完自动跑 val 集预测；0=不跑
KEEP_HISTORY=0     # 0=同 NAME 覆盖（节省空间）；1=每次训练保留历史（NAME 自动加时间戳后缀）
RESUME=""          # 续训：从某个 last.pt 继续训练，例如 "runs/my_run/weights/last.pt"；留空=从头开始

# ==================== 配置结束，下面不用改 ====================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 自动激活环境
source "$SCRIPT_DIR/env.sh"

# 自动取 NAME
if [[ -z "${NAME:-}" ]]; then
    NAME="$(basename "$IMAGES")"
fi

# 保留历史：NAME 加时间戳后缀
if [[ "$KEEP_HISTORY" == "1" ]]; then
    NAME="${NAME}_$(date +%Y%m%d_%H%M%S)"
fi

# 准备 device 参数
DEVICE_ARG=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG="--device $DEVICE"
fi

# AUTO_PREDICT
PREDICT_ARG=""
if [[ "$AUTO_PREDICT" != "1" ]]; then
    PREDICT_ARG="--no-predict"
fi

# RESUME
RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
    RESUME_ARG="--resume $RESUME"
fi

echo "========================================================"
echo " 运行名:   $NAME"
echo " 任务:     $TASK"
echo " 图片:     $IMAGES"
echo " 标签:     $LABELS"
echo " 预训练:   $PRETRAINED"
echo " 参数:     epochs=$EPOCHS, imgsz=$IMGSZ, batch=$BATCH"
if [[ -n "$RESUME" ]]; then
    echo " 续训自:   $RESUME"
fi
echo " 输出:     runs/$NAME/"
echo "========================================================"

python tools/train.py \
    --images "$IMAGES" \
    --labels "$LABELS" \
    --task "$TASK" \
    --name "$NAME" \
    --pretrained "$PRETRAINED" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --val-ratio "$VAL_RATIO" \
    $DEVICE_ARG \
    $PREDICT_ARG \
    $RESUME_ARG
