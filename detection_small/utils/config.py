"""小图检测的配置契约：检查点和 ONNX 元数据共用同一份格式定义。

部署 runtime 由 tools/sync_detection_small_runtime.py 从本模块生成一份只含
格式常量和校验函数的副本，避免训练、PyTorch 推理和 ONNX/OpenVINO 部署对
配置字段的理解出现偏移。
"""
from detection_small.components.preprocessing import STRIDE, normalize_input_size


MODEL_FORMAT = "small-east-resnet18-v1"  # 小图检测模型契约版本，拒绝统一检测或其他尺寸配置。
ONNX_CONFIG_KEY = "ocrsmall_config"  # ONNX 元数据中的小图检测配置字段名。


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
        raise ValueError(f"Expected a small detection config: {config!r}")
    try:
        expected = checkpoint_config(config["input_size"], config.get("dynamic_spatial", False))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid small detection config: {config!r}: {error}") from error
    if config != expected:
        raise ValueError(f"Incompatible small detection config: {config!r}; expected {expected}")
    return config
