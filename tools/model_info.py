#!/usr/bin/env python3
"""模型信息查看：显示模型结构、类别、参数量等。

用法:
    python tools/model_info.py --model best_v26_1.pt
    python tools/model_info.py --model best_v8_2.pt --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 模型信息查看工具")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细层信息")
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

    model = YOLO(str(model_path))
    file_size = model_path.stat().st_size / 1024 / 1024

    print(f"{'=' * 50}")
    print(f"模型文件: {model_path.name} ({file_size:.1f} MB)")
    print(f"任务类型: {model.task}")
    print(f"类别数: {len(model.names)}")
    print(f"类别名: {model.names}")
    print(f"{'=' * 50}")

    model.info(verbose=args.verbose)


if __name__ == "__main__":
    main()
