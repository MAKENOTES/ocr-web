"""识别推理运行包：部署方唯一需要调用的稳定入口。

模型与字典路径必须显式传入；具体后端选择和预处理细节由 RecognitionEngine 负责。
"""

from .engine import (
    BACKEND_CHOICES,
    RecognitionEngine,
    RecognitionReport,
    RecognitionResult,
)
from .loader import collect_image_paths

__all__ = [
    "BACKEND_CHOICES",
    "RecognitionEngine",
    "RecognitionReport",
    "RecognitionResult",
    "collect_image_paths",
]
