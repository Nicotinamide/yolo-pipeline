"""Configuration loading helpers for YOLO pipelines."""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from typing import Any

import yaml

from project_env import ROOT


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    return normalize_pipeline_config(load_raw_pipeline_config(config_path))


def load_raw_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping: {path}")

    context = {
        "root": str(ROOT),
        "config_dir": str(path.parent),
        "cwd": str(Path.cwd()),
        "name": str(raw.get("name") or path.stem),
        "task": str(raw.get("task") or "detect"),
    }
    config = expand_templates(raw, context)
    config["config_path"] = str(path)
    return config


def normalize_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize concise or legacy config schemas into the internal flat schema."""
    normalized = copy.deepcopy(config)
    normalized["name"] = str(normalized.get("name") or "yolo")
    normalized["task"] = str(normalized.get("task") or "detect")
    if normalized["task"] not in {"detect", "segment"}:
        raise ValueError(f"Unsupported task: {normalized['task']}. Use 'detect' or 'segment'.")

    normalized["dataset"] = normalize_dataset_section(normalized)
    normalized["train"] = normalize_train_section(normalized)
    normalized["predict"] = normalize_predict_section(normalized)
    return expand_templates(normalized, template_context(normalized))


def template_context(config: dict[str, Any]) -> dict[str, str]:
    dataset = config.get("dataset") or {}
    train = config.get("train") or {}
    return {
        "root": str(ROOT),
        "cwd": str(Path.cwd()),
        "config_dir": str(Path(config.get("config_path", ".")).parent),
        "name": str(config.get("name") or "yolo"),
        "task": str(config.get("task") or "detect"),
        "dataset": str(dataset.get("output") or ""),
        "data": str(configured_data_yaml(config)) if dataset.get("output") or dataset.get("data_yaml") else "",
        "run": str(Path(str(train.get("project") or ROOT / "runs")) / str(train.get("name") or config.get("name") or "yolo")),
    }


def normalize_dataset_section(config: dict[str, Any]) -> dict[str, Any]:
    name = str(config.get("name") or "yolo")
    dataset = as_mapping(config.get("dataset"))
    prepare = as_mapping(config.get("prepare"))
    prepare_input = as_mapping(prepare.get("input"))
    prepare_output = as_mapping(prepare.get("output"))
    prepare_split = as_mapping(prepare.get("split"))
    prepare_params = as_mapping(prepare.get("params"))

    normalized: dict[str, Any] = {}
    normalized["source"] = first_defined(dataset.get("source"), prepare_input.get("images"), prepare_input.get("source"))
    normalized["label_dir"] = first_defined(dataset.get("label_dir"), prepare_input.get("labels"), prepare_input.get("label_dir"))
    normalized["label_subdir"] = first_defined(dataset.get("label_subdir"), prepare_input.get("label_subdir"), "labels")

    input_names = first_defined(prepare_input.get("names"), prepare_input.get("classes"))
    if "names" in dataset:
        normalized["names"] = dataset.get("names")
    elif isinstance(input_names, (dict, list)):
        normalized["names"] = input_names

    normalized["names_file"] = first_defined(
        dataset.get("names_file"),
        prepare_input.get("names_file"),
        prepare_input.get("classes_file"),
        input_names if isinstance(input_names, str) else None,
    )
    normalized["output"] = first_defined(
        dataset.get("output"),
        prepare_output.get("dataset"),
        prepare_output.get("output"),
        str(ROOT / "datasets" / name),
    )
    normalized["data_yaml"] = first_defined(dataset.get("data_yaml"), str(Path(str(normalized["output"])) / "data.yaml"))
    normalized["mode"] = first_defined(dataset.get("mode"), prepare_output.get("mode"), prepare_params.get("mode"), "copy")
    normalized["overwrite"] = first_defined(
        dataset.get("overwrite"), prepare_output.get("overwrite"), prepare_params.get("overwrite"), False
    )
    normalized["val_ratio"] = first_defined(dataset.get("val_ratio"), prepare_split.get("val_ratio"), 0.2)
    normalized["test_ratio"] = first_defined(dataset.get("test_ratio"), prepare_split.get("test_ratio"), 0.0)
    normalized["seed"] = first_defined(dataset.get("seed"), prepare_split.get("seed"), 42)
    normalized["keep_all_classes"] = first_defined(
        dataset.get("keep_all_classes"), prepare_params.get("keep_all_classes"), False
    )
    normalized["allow_missing_labels"] = first_defined(
        dataset.get("allow_missing_labels"), prepare_params.get("allow_missing_labels"), False
    )
    return remove_none(normalized)


def normalize_train_section(config: dict[str, Any]) -> dict[str, Any]:
    train = as_mapping(config.get("train"))
    train_input = as_mapping(train.get("input"))
    train_output = as_mapping(train.get("output"))
    train_params = as_mapping(train.get("params"))
    flat = {key: value for key, value in train.items() if key not in {"input", "output", "params"}}

    normalized: dict[str, Any] = {}
    normalized.update(train_params)
    normalized.update({"data": train_input.get("data"), "model": train_input.get("model"), "pretrained": train_input.get("pretrained")})
    normalized.update(
        {
            "project": train_output.get("project"),
            "name": first_defined(train_output.get("name"), train_output.get("run")),
        }
    )
    normalized.update(flat)

    # 智能默认值：未指定 pretrained 时按顺序找常用预训练权重
    if normalized.get("pretrained") is None:
        candidates = [
            ROOT / "weights" / "yolo26n.pt",
            ROOT / "weights" / "yolov8n.pt",
            ROOT / "weights" / "yolo11n.pt",
            ROOT / "yolo26n.pt",  # 兼容旧版本（根目录）
        ]
        for candidate in candidates:
            if candidate.exists():
                normalized["pretrained"] = str(candidate)
                break

    normalized["model"] = first_defined(
        normalized.get("model"), infer_model_from_pretrained(config.get("task"), normalized.get("pretrained"))
    )
    normalized["data"] = first_defined(normalized.get("data"), str(configured_data_yaml({**config, "train": normalized})))
    normalized["project"] = first_defined(normalized.get("project"), str(ROOT / "runs"))
    normalized["name"] = first_defined(normalized.get("name"), str(config.get("name") or "yolo_train"))

    # 训练超参的合理默认值（用户显式指定的值优先）
    apply_train_defaults(normalized)
    return remove_none(normalized)


def apply_train_defaults(train: dict[str, Any]) -> None:
    """为常用训练超参填充合理默认值，未显式配置时生效。"""
    defaults = {
        "epochs": 150,
        "imgsz": 960,
        "batch": 8,
        "workers": 4,
        "cache": True,
        "amp": True,
        "deterministic": False,
        "exist_ok": True,        # 默认允许覆盖；避免新手训练完一次就卡住
        "patience": 30,
        "close_mosaic": 15,
    }
    for key, value in defaults.items():
        train.setdefault(key, value)

    # 设备自动选择：未指定时根据 CUDA 可用性决定
    if train.get("device") is None:
        train["device"] = detect_default_device()


def detect_default_device() -> str:
    """检测训练设备：有 CUDA 就用 GPU 0，否则 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def normalize_predict_section(config: dict[str, Any]) -> dict[str, Any]:
    predict = as_mapping(config.get("predict"))
    predict_input = as_mapping(predict.get("input"))
    predict_output = as_mapping(predict.get("output"))
    predict_params = as_mapping(predict.get("params"))
    flat = {key: value for key, value in predict.items() if key not in {"input", "output", "params"}}

    train = as_mapping(config.get("train"))
    dataset = as_mapping(config.get("dataset"))
    train_project = first_defined(train.get("project"), ROOT / "runs")
    train_name = first_defined(train.get("name"), config.get("name"), "yolo_train")
    default_model = Path(str(train_project)) / str(train_name) / "weights" / "best.pt"
    default_source = Path(str(dataset.get("output") or ROOT / "datasets" / str(config.get("name") or "yolo"))) / "images" / "val"

    normalized: dict[str, Any] = {}
    normalized.update(predict_params)
    normalized.update(
        {
            "model": first_defined(predict_input.get("model"), predict_input.get("weights")),
            "source": predict_input.get("source"),
            "names": predict_input.get("names"),
            "use_config_names": first_defined(predict_input.get("use_config_names"), predict_input.get("names_from") != "model"),
        }
    )
    normalized.update(
        {
            "project": predict_output.get("project"),
            "name": first_defined(predict_output.get("name"), predict_output.get("run")),
        }
    )
    normalized.update(flat)
    normalized["model"] = first_defined(normalized.get("model"), str(default_model))
    normalized["source"] = first_defined(normalized.get("source"), str(default_source))
    normalized["project"] = first_defined(normalized.get("project"), str(ROOT / "runs"))
    normalized["name"] = first_defined(normalized.get("name"), f"{config.get('name') or 'yolo'}_predict")
    normalized["imgsz"] = first_defined(normalized.get("imgsz"), train.get("imgsz"))
    normalized["device"] = first_defined(normalized.get("device"), train.get("device"))
    normalized["conf"] = first_defined(normalized.get("conf"), 0.25)
    normalized["iou"] = first_defined(normalized.get("iou"), 0.7)
    normalized["save"] = first_defined(normalized.get("save"), True)
    normalized["exist_ok"] = first_defined(normalized.get("exist_ok"), True)
    return remove_none(normalized)


def infer_model_from_pretrained(task: Any, pretrained: Any) -> str | None:
    """Infer a reasonable model argument when only task and pretrained weights are configured."""
    if not pretrained:
        return None

    pretrained_path = Path(str(pretrained))
    if pretrained_path.suffix != ".pt":
        return None

    stem = pretrained_path.stem
    task_name = str(task or "detect")
    if task_name == "segment":
        if stem.endswith("-seg"):
            return str(pretrained)
        return f"{stem}-seg.yaml"
    if task_name == "detect":
        return str(pretrained)
    return None


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def remove_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def expand_templates(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        formatted = value.format_map(_SafeFormatDict(context))
        expanded = os.path.expandvars(os.path.expanduser(formatted))
        return expanded
    if isinstance(value, list):
        return [expand_templates(item, context) for item in value]
    if isinstance(value, dict):
        return {key: expand_templates(item, context) for key, item in value.items()}
    return value


def parse_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_dotted_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"Invalid override key: {dotted_key}")

    current: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
        current = next_value
    current[parts[-1]] = value


def apply_dotted_overrides(config: dict[str, Any], overrides: list[str] | None) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value: {override}")
        key, value = override.split("=", 1)
        apply_dotted_override(config, key, parse_cli_value(value))


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' is required and must be a mapping")
    return value


def configured_data_yaml(config: dict[str, Any]) -> Path:
    train = config.get("train") or {}
    dataset = config.get("dataset") or {}
    data_yaml = train.get("data") or dataset.get("data_yaml")
    if data_yaml:
        return Path(str(data_yaml)).expanduser()

    output = dataset.get("output")
    if not output:
        raise ValueError("Config needs train.data, dataset.data_yaml, or dataset.output")
    return Path(str(output)).expanduser() / "data.yaml"
