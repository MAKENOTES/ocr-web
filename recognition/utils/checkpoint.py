"""读取纯参数或封装检查点，并推断识别输入通道和 BiGRU 层数。"""

import re
from collections.abc import Mapping

# 匹配 BiGRU 每层的输入权重键（如 rnn.weight_ih_l0），用于推断递归层数。
_RNN_LAYER_KEY = re.compile(r"\.weight_ih_l(\d+)$")


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


def infer_num_layers(state_dict: Mapping, default: int | None = None) -> int:
    """从 BiGRU 权重键的最大层号推断层数，兼容单层与多层权重。

    只统计不带宽度的 ``weight_ih_l{k}``，反向键 ``_reverse`` 不参与计数；
    用该结果构建 CRNN 才能与 checkpoint 的参数形状一致。
    """
    indices = [
        int(match.group(1))
        for name in state_dict
        if isinstance(name, str) and (match := _RNN_LAYER_KEY.search(name))
    ]
    if indices:
        return max(indices) + 1

    if default is not None:
        return default
    raise ValueError("无法从 checkpoint 推断 recognition BiGRU 层数")
