"""统一检测的配置契约：检查点和 ONNX 元数据共用同一份格式定义。

部署 runtime 由本模块生成一份只含格式常量和校验函数的副本，避免训练、
PyTorch 推理和 ONNX/OpenVINO 部署对配置字段的理解出现偏移。
"""

# 本文件由 tools/sync_detection_runtime.py 从 detection/utils/config.py 生成。
# 请勿手改；修改源文件后重新运行同步脚本。
from .preprocessing import STRIDE, normalize_input_size


# 骨干名称 → 模型格式。runtime 快照必须自包含，这里不导入 models/backbones.py；
# 名称与 detection/models/backbones.py 的注册表一致，由测试保证同步。
MODEL_FORMATS = {
    "resnet18": "unified-east-resnet18-v1",
    "mobilenet_v3_large": "unified-east-mobilenetv3-v1",
}
LEGACY_BACKBONE = "resnet18"  # 旧检查点/ONNX 元数据缺少 backbone 字段时的兼容兜底，不是训练默认骨干。
MODEL_FORMAT = MODEL_FORMATS[LEGACY_BACKBONE]  # 旧格式版本，保持历史导入兼容。
ONNX_CONFIG_KEY = "ocrdet_config"  # ONNX 元数据中的统一检测配置字段名。


def checkpoint_config(input_size, dynamic_spatial=False, backbone=LEGACY_BACKBONE):
    """记录骨干、输入尺寸上限及预处理/几何契约；检查点本身不限制推理尺寸。"""
    try:
        model_format = MODEL_FORMATS[backbone]
    except KeyError as error:
        raise ValueError(f"未知检测骨干: {backbone!r}; 可选: {tuple(MODEL_FORMATS)}") from error
    return {
        "format": model_format,
        "backbone": backbone,
        "input_size": list(normalize_input_size(input_size)),
        "stride": STRIDE,
        "geometry": "log1p-trbl-feature-pixels",
        "preprocess": "rgb-imagenet-center-auto-v1",
        "dynamic_spatial": bool(dynamic_spatial),
    }


def _backbone_from_format(model_format):
    """按格式版本反查骨干；旧检查点不含 backbone 字段时用于补齐。"""
    for backbone, candidate in MODEL_FORMATS.items():
        if model_format == candidate:
            return backbone
    raise ValueError(f"Expected a unified EAST checkpoint: {model_format!r}")


def validate_config(config):
    """拒绝格式不符或改过骨干、尺寸、预处理、几何契约的检查点/ONNX 配置。

    返回补齐 backbone 后的配置副本；不修改传入字典，便于调用方按统一字段读取。
    """
    if not isinstance(config, dict):
        raise ValueError(f"Expected a unified EAST checkpoint: {config!r}")
    config = dict(config)
    if "backbone" not in config:
        config["backbone"] = _backbone_from_format(config.get("format"))
    if config["backbone"] not in MODEL_FORMATS:
        raise ValueError(f"未知检测骨干: {config['backbone']!r}; 可选: {tuple(MODEL_FORMATS)}")
    if config["format"] != MODEL_FORMATS[config["backbone"]]:
        raise ValueError(f"Expected a unified EAST checkpoint: {config!r}")
    try:
        expected = checkpoint_config(config["input_size"], config.get("dynamic_spatial", False),
                                     config["backbone"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid unified detection config: {config!r}: {error}") from error
    if config != expected:
        raise ValueError(f"Incompatible unified detection config: {config!r}; expected {expected}")
    return config
