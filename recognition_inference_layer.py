"""兼容入口：保留逐层 BiGRU 的旧命令行默认值，推理由 recognition.runtime 提供。

本文件只负责 CLI 兼容，不再导出后端类；Python 调用请直接使用 recognition.runtime。
与 recognition_inference.py 的差异仅在于默认模型和默认后端。
"""

from recognition.runtime.cli import main as _cli_main

# 多层 BiGRU 的导出产物；旧单层模型请使用 recognition_inference.py。
DEFAULT_MODEL_PATH = "best_recognition_layer.onnx"
DEFAULT_IMAGE_PATH = "recognition/data/test_images"
DEFAULT_KEYS_PATH = "recognition/keys.txt"


def main(argv=None):
    return _cli_main(
        argv,
        description="CRNN 优化推理 (逐层 BiGRU)",
        default_model_path=DEFAULT_MODEL_PATH,
        default_image_path=DEFAULT_IMAGE_PATH,
        default_keys_path=DEFAULT_KEYS_PATH,
        default_backend="openvino",
    )


if __name__ == "__main__":
    main()
