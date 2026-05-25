#!/usr/bin/env python3
"""模型导出工具：将 .pt 导出为 TensorRT / ONNX 等格式用于部署。

用法:
    python tools/export_model.py --model best_v26_1.pt --format engine          # TensorRT FP16
    python tools/export_model.py --model best_v8_2.pt --format onnx             # ONNX
    python tools/export_model.py --model best_v26_1.pt --format engine --imgsz 1280 --no-half  # TensorRT FP32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env
from pipeline.config import detect_default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 模型导出工具")
    parser.add_argument("--model", required=True, help="源模型路径 (.pt)")
    parser.add_argument("--format", default="engine",
                        choices=["engine", "onnx", "torchscript", "openvino", "ncnn"],
                        help="导出格式 (默认 engine=TensorRT)")
    parser.add_argument("--imgsz", type=int, default=960, help="导出分辨率")
    parser.add_argument("--device", default=None, help="导出设备（默认自动检测 GPU/CPU）")
    parser.add_argument("--no-half", action="store_true", help="禁用 FP16 (默认开启半精度)")
    parser.add_argument("--dynamic", action="store_true", help="动态 batch size (ONNX)")
    parser.add_argument("--simplify", action="store_true", help="简化 ONNX 图")
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

    model = YOLO(str(model_path))

    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": device,
        "half": not args.no_half,
    }
    if args.format == "onnx":
        export_args["dynamic"] = args.dynamic
        export_args["simplify"] = args.simplify

    print(f"导出模型: {model_path.name}")
    print(f"格式: {args.format}, 分辨率: {args.imgsz}, FP16: {not args.no_half}")
    print("导出中...")

    result = model.export(**export_args)
    print(f"导出完成: {result}")


if __name__ == "__main__":
    main()
