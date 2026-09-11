"""网络直接复用两个参考项目共有的 ResNet18 EAST，权重注明整图输入契约。"""
import torch

from detection.models.east_model import EASTModel
from detection_larger.components.transforms import DEFAULT_INPUT_SIZE, STRIDE, normalize_input_size

MODEL_FORMAT = "document-east-resnet18-full-image-v2"
ONNX_CONFIG_KEY = "docdet_config"


def create_model(pretrained=False):
    return EASTModel(pretrained=pretrained)


def checkpoint_config(input_size=DEFAULT_INPUT_SIZE):
    return {"format": MODEL_FORMAT, "input_size": list(normalize_input_size(input_size)),
            "stride": STRIDE, "geometry": "log1p-trbl-feature-pixels",
            "preprocess": "rgb-imagenet-center-auto-v1"}


def validate_config(config):
    if config.get("format") != MODEL_FORMAT:
        raise ValueError("Expected full-image EAST checkpoint; old tiled checkpoints require their original inference code")
    expected = checkpoint_config(config["input_size"])
    if config != expected:
        raise ValueError(f"Incompatible document model config: {config}; expected {expected}")


def save_checkpoint(model, path, input_size):
    payload = {"config": checkpoint_config(input_size), "model": model.state_dict()}
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"Expected a document detection checkpoint with config: {path}")
    validate_config(checkpoint["config"])
    model = create_model()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
