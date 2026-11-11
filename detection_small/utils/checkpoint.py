"""小图检测检查点读写；保持纯 state_dict 兼容根目录历史和部署入口。"""
from pathlib import Path

import torch

from detection_small.models.east_model import EASTModel


def save_checkpoint(model, path):
    """原子写入纯模型参数；不包含优化器、调度器或训练状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def extract_state_dict(payload):
    """兼容纯 state_dict 和带 config/model 字段的检查点。"""
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    return payload


def load_state_dict(path):
    """读取本地检查点的模型参数，兼容历史纯权重格式。"""
    return extract_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def load_model(path, device="cpu"):
    """构建 ResNet18 EAST 并加载权重，返回评估模式的模型。"""
    model = EASTModel(pretrained=False)
    model.load_state_dict(load_state_dict(path))
    return model.to(device).eval()
