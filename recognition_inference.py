"""兼容入口：保留历史参数和默认值，推理由 recognition.runtime 提供。

多后端推理、图片收集和输出统计均已收敛到 runtime；本文件只固定旧默认值，
新部署请直接使用 ``python -m recognition.runtime`` 或部署 bundle。
"""

from recognition.runtime.backends import (
    DEFAULT_NUM_CHANNELS,
    IMG_HEIGHT,
    ONNXBackend,
    OpenVINOBackend,
    PyTorchBackend,
    load_backend,
    normalize_output_to_tbc,
)
from recognition.runtime.cli import main as _cli_main

# 历史单层 BiGRU 导出产物；多层模型请使用 recognition_inference_layer.py。
DEFAULT_MODEL_PATH = "best_recognition.onnx"
DEFAULT_IMAGE_PATH = "recognition/data/test_images"
DEFAULT_KEYS_PATH = "recognition/keys.txt"


def main(argv=None):
    return _cli_main(
        argv,
        description="CRNN 优化推理 (修复版)",
        default_model_path=DEFAULT_MODEL_PATH,
        default_image_path=DEFAULT_IMAGE_PATH,
        default_keys_path=DEFAULT_KEYS_PATH,
        default_backend="auto",
    )


if __name__ == "__main__":
    main()
