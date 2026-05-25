#!/usr/bin/env python3
"""多模型对比测试：同一张图片，不同模型的检测结果并排对比。

用法:
    python tools/compare_models.py --source image.png
    python tools/compare_models.py --models best_v8_1.pt best_v26_1.pt --source img.png
    python tools/compare_models.py --models "*.pt" --source image.png --imgsz 1280
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env
from pipeline.config import detect_default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多模型对比推理工具")
    parser.add_argument("--models", nargs="*", default=None,
                        help="模型列表 (支持 glob，如 '*.pt')；默认使用项目根目录下所有 best*.pt")
    parser.add_argument("--source", required=True, help="测试图片/视频路径")
    parser.add_argument("--imgsz", type=int, default=960, help="推理分辨率")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default=None, help="推理设备（默认自动检测 GPU/CPU）")
    return parser.parse_args()


def resolve_models(patterns: list[str] | None) -> list[Path]:
    if patterns is None:
        patterns = ["best*.pt"]

    models: list[Path] = []
    for pattern in patterns:
        p = Path(pattern).expanduser()
        if p.is_absolute():
            matches = glob.glob(str(p))
        else:
            matches = glob.glob(str(ROOT / pattern))
        for m in sorted(matches):
            mp = Path(m)
            if mp.is_file() and mp.suffix == ".pt":
                models.append(mp)

    if not models:
        # 尝试 runs 下的 best.pt
        for bp in sorted(ROOT.glob("runs/*/weights/best.pt")):
            models.append(bp)

    return models


def main() -> None:
    ensure_project_env()
    from ultralytics import YOLO

    args = parse_args()
    models = resolve_models(args.models)
    if not models:
        sys.exit("未找到任何模型文件。用 --models 指定路径或 glob 模式。")

    device = args.device if args.device is not None else detect_default_device()

    source = args.source
    if not Path(source).is_absolute():
        candidate = ROOT / source
        if candidate.exists():
            source = str(candidate)

    print(f"测试图片: {source}")
    print(f"推理分辨率: {args.imgsz}")
    print(f"模型数量: {len(models)}")
    print("=" * 70)

    for model_path in models:
        model = YOLO(str(model_path))
        results = model.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            project=str(ROOT / "runs"),
            name=f"compare_{model_path.stem}",
            save=True,
            exist_ok=True,
            stream=False,
            verbose=False,
        )

        r = results[0]
        n = len(r.boxes)
        speed = f"{r.speed['inference']:.1f}ms" if hasattr(r, 'speed') and r.speed else "N/A"
        model_size = f"{model_path.stat().st_size / 1024 / 1024:.1f}M"

        print(f"\n{model_path.name} ({model_size}, {speed})")
        if n == 0:
            print("  无检测结果")
        else:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                name = model.names[cls_id]
                print(f"  {name}: {conf:.3f}  bbox=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]")

    print("\n" + "=" * 70)
    print(f"结果图片保存在: {ROOT / 'runs'}/compare_*/")


if __name__ == "__main__":
    main()
