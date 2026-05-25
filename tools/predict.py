#!/usr/bin/env python3
"""通用推理脚本：指定模型和图片/视频，直接出结果。

用法:
    python tools/predict.py --model best_v8_1.pt --source image.png
    python tools/predict.py --model best_v26_1.pt --source video.mp4 --imgsz 1280
    python tools/predict.py --model best_v8_2.pt --source 0 --show  # 摄像头
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env
from pipeline.config import detect_default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通用 YOLO 推理工具")
    parser.add_argument("--model", required=True, help="模型文件路径 (.pt)")
    parser.add_argument("--source", required=True, help="图片/目录/视频/摄像头编号")
    parser.add_argument("--imgsz", type=int, default=960, help="推理分辨率 (默认 960)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认 0.25)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值 (默认 0.7)")
    parser.add_argument("--device", default=None, help="推理设备: 0=GPU, cpu=CPU（默认自动检测）")
    parser.add_argument("--name", default=None, help="输出目录名 (默认按模型名生成)")
    parser.add_argument("--show", action="store_true", help="弹窗显示结果")
    parser.add_argument("--save-txt", action="store_true", help="保存 YOLO txt 标签")
    parser.add_argument("--save-conf", action="store_true", help="txt 中包含置信度")
    parser.add_argument("--retina-masks", action="store_true", help="高分辨率 mask (分割模型)")
    parser.add_argument("--no-save", action="store_true", help="不保存结果图片")
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

    device = args.device if args.device is not None else detect_default_device()

    source = args.source
    if source.isdigit():
        source = int(source)
    elif not Path(str(source)).is_absolute():
        candidate = ROOT / source
        if candidate.exists():
            source = str(candidate)

    run_name = args.name or f"predict_{model_path.stem}"

    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=str(ROOT / "runs"),
        name=run_name,
        save=not args.no_save,
        show=args.show,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        retina_masks=args.retina_masks,
        exist_ok=True,
        stream=False,
    )

    # 打印检测摘要
    for i, r in enumerate(results):
        n = len(r.boxes)
        if n == 0:
            print(f"[{i}] 无检测结果")
            continue
        items = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            items.append(f"{name}({conf:.3f})")
        print(f"[{i}] {n} 个目标: {', '.join(items)}")

    if results and not args.no_save:
        print(f"\n结果保存到: {results[0].save_dir}")


if __name__ == "__main__":
    main()
