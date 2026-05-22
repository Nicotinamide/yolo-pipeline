"""Generic Ultralytics YOLO train and predict runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from project_env import ROOT, ensure_project_env

from .config import configured_data_yaml
from .yolo_dataset import normalize_config_names, read_class_names


TRAIN_WRAPPER_KEYS = {"model", "pretrained", "resume"}
PREDICT_WRAPPER_KEYS = {"model", "names", "use_config_names"}


def train_yolo(config: dict[str, Any]):
    ensure_project_env()
    from ultralytics import YOLO

    train = config.get("train") or {}
    if not train:
        raise ValueError("Config section 'train' is required for training")

    resume = train.get("resume")
    if resume:
        checkpoint = Path(str(resume)).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        return model.train(resume=True)

    model_arg = train.get("model")
    if not model_arg:
        raise ValueError("Config needs train.model")

    data_path = configured_data_yaml(config)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}. Run prepare first or fix train.data.")

    model = YOLO(str(model_arg))
    pretrained = train.get("pretrained")
    if pretrained:
        pretrained_path = Path(str(pretrained)).expanduser()
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
        model_path = Path(str(model_arg)).expanduser()
        if model_path.resolve() != pretrained_path.resolve() if model_path.exists() else True:
            model.load(str(pretrained_path))

    train_args = {key: value for key, value in train.items() if key not in TRAIN_WRAPPER_KEYS and value is not None}
    train_args.setdefault("data", str(data_path))
    train_args.setdefault("project", str(ROOT / "runs"))
    train_args.setdefault("name", str(config.get("name") or "yolo_train"))
    return model.train(**train_args)


def predict_yolo(config: dict[str, Any], source: str | None = None):
    ensure_project_env()
    from ultralytics import YOLO

    predict = config.get("predict") or {}
    if not predict:
        raise ValueError("Config section 'predict' is required for prediction")

    model_arg = predict.get("model")
    if not model_arg:
        raise ValueError("Config needs predict.model")
    model_path = Path(str(model_arg)).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    predict_source = source or predict.get("source")
    if predict_source is None:
        raise ValueError("Config needs predict.source, or pass --predict-source")

    model = YOLO(str(model_path))
    names = resolve_predict_names(config)
    if names:
        model.model.names = names
    predict_args = {key: value for key, value in predict.items() if key not in PREDICT_WRAPPER_KEYS and value is not None}
    predict_args["source"] = normalize_source(str(predict_source))
    predict_args.setdefault("project", str(ROOT / "runs"))
    predict_args.setdefault("name", str(config.get("name") or "yolo_predict"))
    predict_args.setdefault("save", True)
    predict_args.setdefault("exist_ok", True)
    predict_args.setdefault("stream", False)

    results = model.predict(**predict_args)
    save_dir = results[0].save_dir if results else Path(str(predict_args["project"])) / str(predict_args["name"])
    print(f"Results saved to {save_dir}")
    return results


def normalize_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def resolve_predict_names(config: dict[str, Any]) -> dict[int, str]:
    predict = config.get("predict") or {}
    if predict.get("use_config_names", True) is False:
        return {}

    predict_names = normalize_config_names(predict.get("names"))
    if predict_names:
        return names_list_to_dict(predict_names)

    dataset = config.get("dataset") or {}
    dataset_names = normalize_config_names(dataset.get("names"))
    if dataset_names:
        return names_list_to_dict(dataset_names)

    data_yaml = configured_data_yaml(config)
    if data_yaml.exists():
        data = yaml.safe_load(data_yaml.read_text()) or {}
        data_names = normalize_config_names(data.get("names"))
        if data_names:
            return names_list_to_dict(data_names)

    names_file = dataset.get("names_file")
    if names_file:
        file_names = read_class_names(Path(str(names_file)).expanduser())
        if file_names:
            return names_list_to_dict(file_names)

    return {}


def names_list_to_dict(names: list[str]) -> dict[int, str]:
    return {class_index: class_name for class_index, class_name in enumerate(names)}


def summarize_config(config: dict[str, Any]) -> str:
    dataset = config.get("dataset") or {}
    train = config.get("train") or {}
    predict = config.get("predict") or {}
    lines = [
        f"name: {config.get('name', '')}",
        f"task: {config.get('task', 'detect')}",
        "prepare:",
        f"  input.images: {dataset.get('source', '<existing YOLO dataset>')}",
        f"  input.labels: {dataset.get('label_dir', '')}",
        f"  input.names_file: {dataset.get('names_file', '')}",
        f"  output.dataset: {dataset.get('output', '')}",
        f"  output.data_yaml: {configured_data_yaml(config)}",
        f"  split.val_ratio: {dataset.get('val_ratio', '')}",
        "train:",
        f"  input.data: {train.get('data') or configured_data_yaml(config)}",
        f"  input.model: {train.get('model', '')}",
        f"  input.pretrained: {train.get('pretrained', '')}",
        f"  output.run: {Path(str(train.get('project') or ROOT / 'runs')) / str(train.get('name') or config.get('name') or 'yolo')}",
        f"  params.imgsz: {train.get('imgsz', '')}",
        f"  params.batch: {train.get('batch', '')}",
        f"  params.epochs: {train.get('epochs', '')}",
        "predict:",
        f"  input.model: {predict.get('model', '')}",
        f"  input.source: {predict.get('source', '')}",
        f"  output.run: {Path(str(predict.get('project') or ROOT / 'runs')) / str(predict.get('name') or config.get('name') or 'yolo_predict')}",
        f"  params.imgsz: {predict.get('imgsz', '')}",
        f"  params.conf: {predict.get('conf', '')}",
    ]
    return "\n".join(lines)
