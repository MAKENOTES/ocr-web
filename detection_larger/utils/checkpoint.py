"""大图检测检查点的读取与写入；配置字段和校验规则定义在 utils/config.py。"""
from pathlib import Path

import torch

from detection_larger.models.east_model import create_model
from detection_larger.utils.config import (
    MODEL_FORMAT,
    ONNX_CONFIG_KEY,
    checkpoint_config,
    validate_config,
)


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
        raise ValueError(f"Expected a document detection checkpoint with config and model: {path}")
    validate_config(checkpoint["config"])
    model = create_model()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
