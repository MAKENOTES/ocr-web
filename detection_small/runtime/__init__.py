"""小图检测推理运行包：部署方唯一需要调用的稳定入口。

模型和图片路径必须显式传入；ONNX/OpenVINO 后端、预处理和解码由本包负责。
runtime 快照由 tools/sync_detection_small_runtime.py 从仓库源文件生成。
"""
from .backends import (
    BACKEND_CHOICES,
    OnnxRuntimeBackend,
    OpenVINOBackend,
    exported_shape,
    load_backend,
)
from .engine import collect_images, predict_image, run_inference, warmup_backend

__all__ = [
    "BACKEND_CHOICES",
    "OnnxRuntimeBackend",
    "OpenVINOBackend",
    "collect_images",
    "exported_shape",
    "load_backend",
    "predict_image",
    "run_inference",
    "warmup_backend",
]
