"""统一检测检查点的配置契约与读写，供训练、推理、导出和 ONNX 元数据共用。"""
from pathlib import Path

import torch

from detection.components.transforms import STRIDE, normalize_input_size
from detection.models.east_model import create_model


MODEL_FORMAT = "unified-east-resnet18-v1"  # 统一检测模型契约版本，拒绝历史切片或文档专用配置。
ONNX_CONFIG_KEY = "ocrdet_config"  # ONNX 元数据中的统一检测配置字段名。


def checkpoint_config(input_size, dynamic_spatial=False):
    """记录输入尺寸上限及预处理/几何契约；检查点本身不限制推理尺寸。"""
    return {
        "format": MODEL_FORMAT,
        "input_size": list(normalize_input_size(input_size)),
        "stride": STRIDE,
        "geometry": "log1p-trbl-feature-pixels",
        "preprocess": "rgb-imagenet-center-auto-v1",
        "dynamic_spatial": bool(dynamic_spatial),
    }


def validate_config(config):
    """拒绝格式不符或改过尺寸、预处理、几何契约的检查点/ONNX 配置。"""
    if not isinstance(config, dict) or config.get("format") != MODEL_FORMAT:
        raise ValueError(f"Expected a unified EAST checkpoint: {config!r}")
    try:
        expected = checkpoint_config(config["input_size"], config.get("dynamic_spatial", False))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid unified detection config: {config!r}: {error}") from error
    if config != expected:
        raise ValueError(f"Incompatible unified detection config: {config!r}; expected {expected}")
    return config


def save_checkpoint(model, path, input_size, dynamic_spatial=False):
    """按契约写入 config 和 model；先写临时文件再替换，避免留下半个权重文件。"""
    path = Path(path)
    payload = {"config": checkpoint_config(input_size, dynamic_spatial),
               "model": model.state_dict()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    """校验检查点契约后构建模型并切到评估模式；返回模型和原始检查点字典。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"Expected a unified detection checkpoint with config and model: {path}")
    validate_config(checkpoint["config"])
    model = create_model()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint

