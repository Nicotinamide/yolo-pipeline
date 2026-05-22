"""YOLO dataset preparation for detection and segmentation tasks."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path | None


@dataclass(frozen=True)
class LabelScan:
    object_count: int
    class_counts: Counter[int]
    vertex_counts: list[int]


def prepare_yolo_dataset(config: dict[str, Any], overwrite: bool | None = None) -> dict[str, Any]:
    dataset = config.get("dataset") or {}
    task = str(config.get("task") or dataset.get("task") or "detect")
    if not dataset.get("source"):
        data_yaml = dataset.get("data_yaml")
        if data_yaml and Path(str(data_yaml)).expanduser().exists():
            print(f"Dataset already prepared: {Path(str(data_yaml)).expanduser().resolve()}")
            return {"data_yaml": Path(str(data_yaml)).expanduser(), "prepared": False}
        raise ValueError("Prepare needs dataset.source/prepare.input.images, or an existing dataset.data_yaml")

    source_dir = Path(str(dataset["source"])).expanduser().resolve()
    label_dir = Path(str(dataset.get("label_dir") or dataset.get("label_subdir", "labels"))).expanduser()
    if not label_dir.is_absolute():
        label_dir = source_dir / label_dir
    output_dir = Path(str(dataset["output"])).expanduser()
    allow_missing_labels = bool(dataset.get("allow_missing_labels", False))

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    if not label_dir.exists() and not allow_missing_labels:
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    items, class_counts, vertex_counts, object_count = collect_items(
        source_dir=source_dir,
        label_dir=label_dir,
        task=task,
        allow_missing_labels=allow_missing_labels,
    )
    class_names = choose_class_names(dataset, default_classes_path=source_dir / "classes.txt", class_counts=class_counts)
    if class_counts and max(class_counts) >= len(class_names):
        raise ValueError(f"Labels use class {max(class_counts)}, but only {len(class_names)} names are defined")

    splits = split_items(
        items=items,
        val_ratio=float(dataset.get("val_ratio", 0.2)),
        test_ratio=float(dataset.get("test_ratio", 0.0)),
        seed=int(dataset.get("seed", 42)),
    )
    should_overwrite = bool(dataset.get("overwrite", False)) if overwrite is None else overwrite
    prepare_output(output_dir, should_overwrite)
    write_dataset_files(output_dir, splits, str(dataset.get("mode", "copy")))
    data_yaml = write_data_yaml(output_dir, splits, class_names)
    summary_path = write_summary(
        output_dir=output_dir,
        source_dir=source_dir,
        label_dir=label_dir,
        task=task,
        splits=splits,
        class_counts=class_counts,
        vertex_counts=vertex_counts,
        object_count=object_count,
        class_names=class_names,
    )

    split_text = ", ".join(f"{split_name}={len(split_items)}" for split_name, split_items in splits.items())
    print(f"Prepared YOLO {task} dataset: {output_dir.resolve()}")
    print(f"Images: {len(items)}, objects: {object_count}, splits: {split_text}")
    print(f"Class counts: {dict(sorted(class_counts.items()))}")
    print(f"data.yaml: {data_yaml.resolve()}")
    print(f"summary: {summary_path.resolve()}")

    return {
        "output": output_dir,
        "data_yaml": data_yaml,
        "summary": summary_path,
        "splits": {split_name: len(split_items) for split_name, split_items in splits.items()},
        "class_counts": dict(sorted(class_counts.items())),
        "object_count": object_count,
    }


def list_images(source_dir: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in source_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS
    )


def read_class_names(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        return []
    return [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]


def normalize_config_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            normalized[int(key)] = str(value)
        if not normalized:
            return []
        max_class_id = max(normalized)
        missing = [class_id for class_id in range(max_class_id + 1) if class_id not in normalized]
        if missing:
            raise ValueError(f"names mapping must be contiguous from 0, missing: {missing}")
        return [normalized[class_id] for class_id in range(max_class_id + 1)]
    raise ValueError("dataset.names must be a list or mapping")


def scan_label(label_path: Path, task: str) -> LabelScan:
    class_counts: Counter[int] = Counter()
    vertex_counts: list[int] = []
    object_count = 0

    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{label_path}:{line_number} contains a non-numeric YOLO label") from exc

        validate_label_coords(label_path, line_number, task, class_id, coords)
        object_count += 1
        class_counts[class_id] += 1
        if task == "segment":
            vertex_counts.append(len(coords) // 2)

    return LabelScan(object_count=object_count, class_counts=class_counts, vertex_counts=vertex_counts)


def validate_label_coords(label_path: Path, line_number: int, task: str, class_id: int, coords: list[float]) -> None:
    if class_id < 0:
        raise ValueError(f"{label_path}:{line_number} has negative class id {class_id}")
    if any(coord < 0.0 or coord > 1.0 for coord in coords):
        raise ValueError(f"{label_path}:{line_number} has coordinates outside [0, 1]")

    if task == "detect":
        if len(coords) != 4:
            raise ValueError(f"{label_path}:{line_number} must be YOLO detect: class x_center y_center width height")
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            raise ValueError(f"{label_path}:{line_number} has non-positive width or height")
        return

    if task == "segment":
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"{label_path}:{line_number} must be YOLO segment: class x1 y1 x2 y2 x3 y3 ...")
        return

    raise ValueError(f"Unsupported YOLO task for dataset preparation: {task}")


def collect_items(
    source_dir: Path, label_dir: Path, task: str, allow_missing_labels: bool = False
) -> tuple[list[DatasetItem], Counter[int], list[int], int]:
    image_paths = list_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    items: list[DatasetItem] = []
    class_counts: Counter[int] = Counter()
    vertex_counts: list[int] = []
    object_count = 0
    missing_labels: list[Path] = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            if allow_missing_labels:
                items.append(DatasetItem(image_path=image_path, label_path=None))
                continue
            missing_labels.append(label_path)
            continue

        label_scan = scan_label(label_path, task)
        object_count += label_scan.object_count
        class_counts.update(label_scan.class_counts)
        vertex_counts.extend(label_scan.vertex_counts)
        items.append(DatasetItem(image_path=image_path, label_path=label_path))

    if missing_labels:
        examples = ", ".join(str(label_path) for label_path in missing_labels[:5])
        raise FileNotFoundError(f"Missing {len(missing_labels)} label files, examples: {examples}")
    if not items:
        raise RuntimeError("No image/label pairs were collected")

    return items, class_counts, vertex_counts, object_count


def choose_class_names(dataset: dict[str, Any], default_classes_path: Path, class_counts: Counter[int]) -> list[str]:
    config_names = normalize_config_names(dataset.get("names"))
    if config_names:
        return config_names

    max_class_id = max(class_counts) if class_counts else 0
    single_class_name = dataset.get("single_class_name")
    if max_class_id == 0 and single_class_name:
        return [str(single_class_name)]

    classes_path = Path(str(dataset.get("names_file") or default_classes_path)).expanduser()
    if not classes_path.is_absolute():
        classes_path = default_classes_path.parent / classes_path
    source_class_names = read_class_names(classes_path)
    required_count = max_class_id + 1
    if source_class_names:
        if len(source_class_names) < required_count:
            raise ValueError(f"classes.txt has {len(source_class_names)} names, but labels require {required_count}")
        return source_class_names if dataset.get("keep_all_classes", False) else source_class_names[:required_count]

    return [str(class_index) for class_index in range(required_count)]


def split_items(
    items: list[DatasetItem], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[DatasetItem]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Split ratios must be non-negative and sum to less than 1")

    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    total_count = len(shuffled)
    test_count = round(total_count * test_ratio)
    val_count = round(total_count * val_ratio)

    if test_ratio > 0 and total_count > 2:
        test_count = max(1, test_count)
    if val_ratio > 0 and total_count > 1:
        val_count = max(1, val_count)

    while total_count - test_count - val_count < 1:
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    test_items = shuffled[:test_count]
    val_items = shuffled[test_count : test_count + val_count]
    train_items = shuffled[test_count + val_count :]
    splits = {"train": train_items, "val": val_items}
    if test_items:
        splits["test"] = test_items
    return splits


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Re-run with --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_or_link(source_path: Path, dest_path: Path, mode: str) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dest_path.symlink_to(source_path.resolve())
    elif mode == "copy":
        shutil.copy2(source_path, dest_path)
    else:
        raise ValueError(f"Unsupported dataset file mode: {mode}")


def write_dataset_files(output_dir: Path, splits: dict[str, list[DatasetItem]], mode: str) -> None:
    for split_name, split_items_for_subset in splits.items():
        for item in split_items_for_subset:
            image_dest = output_dir / "images" / split_name / item.image_path.name
            label_dest = output_dir / "labels" / split_name / f"{item.image_path.stem}.txt"
            copy_or_link(item.image_path, image_dest, mode)
            if item.label_path is None:
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                label_dest.write_text("")
            else:
                copy_or_link(item.label_path, label_dest, mode)


def yaml_quote(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def write_data_yaml(output_dir: Path, splits: dict[str, list[DatasetItem]], class_names: list[str]) -> Path:
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        lines.append("test: images/test")
    lines.extend([f"nc: {len(class_names)}", "names:"])
    for class_index, class_name in enumerate(class_names):
        lines.append(f"  {class_index}: {yaml_quote(class_name)}")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n")
    return data_yaml


def write_summary(
    output_dir: Path,
    source_dir: Path,
    label_dir: Path,
    task: str,
    splits: dict[str, list[DatasetItem]],
    class_counts: Counter[int],
    vertex_counts: list[int],
    object_count: int,
    class_names: list[str],
) -> Path:
    summary_lines = [
        f"task: {task}",
        f"source: {source_dir.resolve()}",
        f"label_dir: {label_dir.resolve()}",
        f"images: {sum(len(split_items_for_subset) for split_items_for_subset in splits.values())}",
        f"objects: {object_count}",
        "splits:",
    ]
    for split_name, split_items_for_subset in splits.items():
        summary_lines.append(f"  {split_name}: {len(split_items_for_subset)}")
    summary_lines.extend(["classes:", *[f"  {class_index}: {class_name}" for class_index, class_name in enumerate(class_names)]])
    summary_lines.append("class_object_counts:")
    for class_id, count in sorted(class_counts.items()):
        summary_lines.append(f"  {class_id}: {count}")
    if vertex_counts:
        sorted_vertices = sorted(vertex_counts)
        summary_lines.extend(
            [
                f"vertices_min: {sorted_vertices[0]}",
                f"vertices_median: {sorted_vertices[len(sorted_vertices) // 2]}",
                f"vertices_max: {sorted_vertices[-1]}",
            ]
        )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return summary_path
