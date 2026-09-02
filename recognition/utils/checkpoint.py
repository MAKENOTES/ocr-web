"""Recognition checkpoint compatibility helpers."""

from collections.abc import Mapping


def extract_state_dict(checkpoint) -> Mapping:
    """兼容纯 state_dict 和包含 ``state_dict`` 的 checkpoint。"""
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint 中未找到有效的 state_dict")
    return state_dict


def infer_num_channels(state_dict: Mapping, default: int | None = None) -> int:
    """从第一层卷积权重推断模型输入通道，兼容旧灰度与新 RGB 权重。"""
    preferred_suffixes = (
        "cnn.stem.0.weight",
        "cnn.conv1.weight",
    )
    for suffix in preferred_suffixes:
        for name, tensor in state_dict.items():
            if name.endswith(suffix) and getattr(tensor, "ndim", 0) == 4:
                return int(tensor.shape[1])

    if default is not None:
        return default
    raise ValueError("无法从 checkpoint 推断 recognition 输入通道数")
