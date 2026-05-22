#!/usr/bin/env python3
"""一行命令训练 YOLO：给图片文件夹 + 标签文件夹，自动准备数据、训练、预测。

最常用形态:
    python tools/train.py --images /path/to/images --labels /path/to/labels
        # 自动取最后一级目录名作为运行名
        # 自动 80/20 划分 train/val
        # 自动用 yolo26n.pt 预训练
        # 训练完自动在 val 集上预测可视化结果

更换数据/模型对比效果:
    python tools/train.py --images /data/A --labels /data/A_lbl --pretrained yolo26n.pt --name A_n
    python tools/train.py --images /data/A --labels /data/A_lbl --pretrained yolo26s.pt --name A_s
    python tools/train.py --images /data/B --labels /data/B_lbl --task segment       --name B_seg

已经有 YOLO 数据集（含 data.yaml）跳过准备阶段:
    python tools/train.py --data datasets/foo/data.yaml --name foo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_env import ROOT, ensure_project_env
from pipeline.config import normalize_pipeline_config
from pipeline.yolo_dataset import prepare_yolo_dataset
from pipeline.yolo_runner import predict_yolo, train_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="给图片+标签文件夹，一行命令训练 YOLO 并自动预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据来源（二选一）
    parser.add_argument("--images", help="图片文件夹（与 --labels 配套使用）")
    parser.add_argument("--labels", help="标签文件夹（YOLO txt 格式）")
    parser.add_argument("--classes", help="类别名 txt 文件路径；不填会自动找 images 或 labels 同级的 classes.txt")
    parser.add_argument("--data", help="已有 YOLO data.yaml 路径（跳过 prepare 阶段）")

    # 任务和模型
    parser.add_argument("--task", default="detect", choices=["detect", "segment"], help="任务类型")
    parser.add_argument("--pretrained", default=None,
                        help="预训练权重，如 yolo26n.pt / yolo8s.pt / yolo11n.pt（默认项目根目录的 yolo26n.pt）")
    parser.add_argument("--model", default=None,
                        help="模型架构 yaml；不填则按 task+pretrained 自动推导")

    # 运行名
    parser.add_argument("--name", default=None,
                        help="运行名（决定 datasets/<name> 和 runs/<name> 目录名）；不填则取 images 文件夹名")

    # 划分
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")

    # 训练参数（其他全用智能默认值）
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="CUDA 设备（默认自动检测）")

    # 流程控制
    parser.add_argument("--no-predict", action="store_true", help="训练后不在 val 集上跑预测")
    parser.add_argument("--prepare-only", action="store_true", help="只准备数据集，不训练")
    parser.add_argument("--dry-run", action="store_true", help="只打印解析后的配置，不执行")
    parser.add_argument("--resume", default=None,
                        help="从 checkpoint 续训，例如 runs/my_run/weights/last.pt")
    return parser.parse_args()


def auto_run_name(args: argparse.Namespace) -> str:
    """从 images 路径自动取最后一级目录名当运行名。"""
    if args.name:
        return args.name
    if args.images:
        return Path(args.images).expanduser().resolve().name
    if args.data:
        return Path(args.data).expanduser().resolve().parent.name
    return "yolo_run"


def auto_classes_file(args: argparse.Namespace) -> str | None:
    """自动定位 classes.txt：优先用户指定，其次 images/classes.txt，最后 labels/classes.txt 或上级。"""
    if args.classes:
        return args.classes
    if not args.images:
        return None
    images_dir = Path(args.images).expanduser().resolve()
    candidates = [
        images_dir / "classes.txt",
        images_dir.parent / "classes.txt",
    ]
    if args.labels:
        labels_dir = Path(args.labels).expanduser().resolve()
        candidates.extend([labels_dir / "classes.txt", labels_dir.parent / "classes.txt"])
    for c in candidates:
        if c.exists():
            return str(c)
    return None  # 没有也没关系，pipeline 会用默认数字类名


def build_config(args: argparse.Namespace) -> dict:
    if not args.images and not args.data:
        sys.exit("错误：必须指定 --images（搭配 --labels），或 --data 指向已有 data.yaml")
    if args.images and not args.labels:
        sys.exit("错误：指定 --images 时必须同时指定 --labels（YOLO txt 标签目录）")

    name = auto_run_name(args)
    config: dict = {
        "name": name,
        "task": args.task,
        "train": {
            "params": {
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
            },
        },
    }

    # 训练输入
    train_input = {}
    if args.pretrained is not None:
        train_input["pretrained"] = args.pretrained
    if args.model is not None:
        train_input["model"] = args.model
    if train_input:
        config["train"]["input"] = train_input
    if args.device is not None:
        config["train"]["params"]["device"] = args.device
    if args.resume is not None:
        config["train"]["params"]["resume"] = args.resume

    # 数据
    if args.data:
        config["dataset"] = {"data_yaml": args.data}
    else:
        prepare_input = {
            "images": str(Path(args.images).expanduser().resolve()),
            "labels": str(Path(args.labels).expanduser().resolve()),
        }
        classes_file = auto_classes_file(args)
        if classes_file:
            prepare_input["names_file"] = classes_file
        config["prepare"] = {
            "input": prepare_input,
            "output": {"overwrite": True},
            "split": {"val_ratio": args.val_ratio, "seed": args.seed},
        }

    return config


def main() -> None:
    args = parse_args()
    raw_config = build_config(args)
    raw_config["config_path"] = str(ROOT)
    config = normalize_pipeline_config(raw_config)

    if args.dry_run:
        from pipeline.yolo_runner import summarize_config
        print(summarize_config(config))
        return

    ensure_project_env()

    # 1. 准备数据集
    if args.images:
        print("=" * 60)
        print(f"[1/3] 准备数据集 → datasets/{config['name']}/")
        print("=" * 60)
        prepare_yolo_dataset(config, overwrite=True)

    if args.prepare_only:
        return

    # 2. 训练
    print("\n" + "=" * 60)
    step = 2 if args.images else 1
    total = 3 if not args.no_predict else 2
    if args.images is None:
        total -= 1
    print(f"[{step}/{total}] 训练 → runs/{config['name']}/")
    print(f"        epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
    print(f"        pretrained={config['train'].get('pretrained', 'none')}")
    print("=" * 60)
    train_yolo(config)

    # 3. val 集预测
    if not args.no_predict:
        print("\n" + "=" * 60)
        print(f"[{step + 1}/{total}] val 集预测 → runs/{config['name']}_predict/")
        print("=" * 60)
        predict_yolo(config)

    print("\n" + "=" * 60)
    print(f"完成。模型: runs/{config['name']}/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
