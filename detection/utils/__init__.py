"""统一检测的通用工具；配置契约和检查点读写由本文件汇总导出，其余模块按需直接导入。"""
from detection.utils.checkpoint import load_checkpoint, save_checkpoint
from detection.utils.config import (
    MODEL_FORMAT,
    ONNX_CONFIG_KEY,
    checkpoint_config,
    validate_config,
)

__all__ = [
    "MODEL_FORMAT",
    "ONNX_CONFIG_KEY",
    "checkpoint_config",
    "load_checkpoint",
    "save_checkpoint",
    "validate_config",
]
