"""大图检测检查点的配置契约与读写，供训练、推理、导出和 ONNX 元数据共用。"""
import torch

from detection_larger.components.transforms import STRIDE, normalize_input_size
from detection_larger.models.east_model import create_model

MODEL_FORMAT = "document-east-resnet18-full-image-v2"  # 整图模型契约版本，保持已有文档检测权重兼容。
ONNX_CONFIG_KEY = "docdet_config"  # ONNX 元数据中的文档检测配置字段名。


def checkpoint_config(input_size):
    """记录调用方尺寸及预处理/几何契约；加载时校验格式，推理尺寸仍由入口常量决定。"""
    return {"format": MODEL_FORMAT, "input_size": list(normalize_input_size(input_size)),
            "stride": STRIDE, "geometry": "log1p-trbl-feature-pixels",
            "preprocess": "rgb-imagenet-center-auto-v1"}


def validate_config(config):
    """拒绝旧切片检查点和改过尺寸、预处理或几何契约的整图检查点。"""
    if config.get("format") != MODEL_FORMAT:
        raise ValueError("Expected full-image EAST checkpoint; old tiled checkpoints require their original inference code")
    expected = checkpoint_config(config["input_size"])
    if config != expected:
        raise ValueError(f"Incompatible document model config: {config}; expected {expected}")


def save_checkpoint(model, path, input_size):
    """按检查点契约写入 config 和 model；先写临时文件再替换，避免留下半个权重文件。"""
    payload = {"config": checkpoint_config(input_size), "model": model.state_dict()}
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    """校验检查点契约后构建模型并切到评估模式；返回模型和原始检查点字典。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"Expected a document detection checkpoint with config: {path}")
    validate_config(checkpoint["config"])
    model = create_model()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
