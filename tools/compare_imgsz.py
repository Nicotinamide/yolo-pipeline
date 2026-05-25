#!/usr/bin/env python3
"""分辨率对比测试：同一模型 + 同一图片，不同 imgsz 的效果对比。

用法:
    python tools/compare_imgsz.py --model best_v26_1.pt --source image.png
    python tools/compare_imgsz.py --model best_v8_2.pt --source img.png --sizes 640 960 1280 1920
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env
from pipeline.config import detect_default_device


DEFAULT_SIZES = [640, 960, 1280, 1920]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分辨率对比推理工具")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--source", required=True, help="测试图片路径")
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help=f"要测试的 imgsz 列表 (默认 {DEFAULT_SIZES})")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default=None, help="推理设备（默认自动检测 GPU/CPU）")
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
    if not Path(source).is_absolute():
        candidate = ROOT / source
        if candidate.exists():
            source = str(candidate)

    model = YOLO(str(model_path))
    print(f"模型: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.1f}M)")
    print(f"图片: {source}")
    print(f"测试分辨率: {args.sizes}")
    print("=" * 70)
    print(f"{'imgsz':>6} | {'检测数':>4} | {'推理耗时':>8} | 检测结果")
    print("-" * 70)

    for sz in args.sizes:
        results = model.predict(
            source=source,
            imgsz=sz,
            conf=args.conf,
            device=device,
            project=str(ROOT / "runs"),
            name=f"imgsz_{model_path.stem}_{sz}",
            save=True,
            exist_ok=True,
            stream=False,
            verbose=False,
        )

        r = results[0]
        n = len(r.boxes)
        speed = f"{r.speed['inference']:.1f}ms" if hasattr(r, 'speed') and r.speed else "N/A"

        if n == 0:
            print(f"{sz:>6} | {n:>4} | {speed:>8} | 无检测")
        else:
            items = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                items.append(f"{name}({conf:.3f})")
            print(f"{sz:>6} | {n:>4} | {speed:>8} | {', '.join(items)}")

    print("=" * 70)
    print(f"结果图片: {ROOT / 'runs'}/imgsz_{model_path.stem}_*/")


if __name__ == "__main__":
    main()
