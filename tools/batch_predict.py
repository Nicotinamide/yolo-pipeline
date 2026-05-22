#!/usr/bin/env python3
"""批量推理：对一个文件夹的所有图片跑推理，输出统计摘要。

用法:
    python tools/batch_predict.py --model best_v26_1.pt --source /path/to/images/
    python tools/batch_predict.py --model best_v8_2.pt --source datasets/d19_seg/images/val
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量推理与统计工具")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--source", required=True, help="图片目录")
    parser.add_argument("--imgsz", type=int, default=960, help="推理分辨率")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default="0", help="推理设备")
    parser.add_argument("--save", action="store_true", help="保存带标注的结果图片")
    parser.add_argument("--save-txt", action="store_true", help="保存 YOLO txt 预测")
    return parser.parse_args()


def main() -> None:
    ensure_project_env()
    from ultralytics import YOLO

    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        sys.exit(f"模型不存在: {model_path}")

    source = args.source
    if not Path(source).is_absolute():
        candidate = ROOT / source
        if candidate.exists():
            source = str(candidate)

    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(ROOT / "runs"),
        name=f"batch_{model_path.stem}",
        save=args.save,
        save_txt=args.save_txt,
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    total_images = 0
    total_objects = 0
    class_counts: Counter[str] = Counter()
    conf_sum = 0.0
    empty_images = 0

    for r in results:
        total_images += 1
        n = len(r.boxes)
        if n == 0:
            empty_images += 1
            continue
        total_objects += n
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            class_counts[name] += 1
            conf_sum += conf

    print(f"\n{'=' * 50}")
    print(f"模型: {model_path.name}")
    print(f"数据: {source}")
    print(f"分辨率: {args.imgsz}, 置信度阈值: {args.conf}")
    print(f"{'=' * 50}")
    print(f"图片总数: {total_images}")
    print(f"有检测结果的图片: {total_images - empty_images}")
    print(f"无检测结果的图片: {empty_images}")
    print(f"检测目标总数: {total_objects}")
    if total_objects > 0:
        print(f"平均置信度: {conf_sum / total_objects:.3f}")
        print(f"每张图平均目标数: {total_objects / total_images:.2f}")
    print(f"\n类别统计:")
    for name, count in class_counts.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
