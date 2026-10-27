"""识别引擎：对外唯一的推理入口。

部署方只需构造 RecognitionEngine 并调用 recognize，不需要了解后端和 DataLoader 细节。
"""

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image

from recognition.components.converter import LabelConverter
from recognition.runtime.backends import load_backend
from recognition.runtime.preprocess import build_inference_loader

# 可用的后端名称；auto 按模型扩展名选择。
BACKEND_CHOICES = ("auto", "onnx", "openvino", "pytorch")


@dataclass(frozen=True)
class RecognitionResult:
    """单张图片的识别结果；path 是预处理阶段校验后的图片路径。"""

    path: str
    text: str


@dataclass(frozen=True)
class RecognitionReport:
    """一次 recognize 的结果与耗时统计。

    results 按调用方传入的图片顺序排列；迭代 report 等价于迭代 results。
    """

    results: list[RecognitionResult]
    infer_ms: float  # 所有 batch 的推理+解码耗时之和（毫秒）
    wall_seconds: float  # 含图片加载、预处理和推理的总耗时（秒）

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


class RecognitionEngine:
    """加载一次模型，重复识别多张图片。

    模型和字典路径必须显式传入；ONNX 会按导出时写入的 keys_sha256 校验字典，
    避免类别顺序不一致导致静默错字。
    """

    def __init__(self, model_path: str | os.PathLike, keys_path: str | os.PathLike,
                 backend: str = "auto", num_threads: int = 1,
                 allow_spinning: bool = False):
        self.model_path = os.fspath(model_path)
        self.keys_path = os.fspath(keys_path)

        # 尽早给出带具体路径的错误，避免部署时只看到第三方库的模糊异常。
        for label, path in (("模型", self.model_path), ("字典", self.keys_path)):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label}文件不存在: {path}")
        if backend not in BACKEND_CHOICES:
            raise ValueError(
                f"backend 仅支持 {', '.join(BACKEND_CHOICES)}，当前为 {backend!r}"
            )

        self.converter = LabelConverter(self.keys_path)
        self.backend = load_backend(
            self.model_path, backend, self.keys_path, num_threads, allow_spinning
        )
        self.num_channels = self.backend.num_channels

    def recognize(self, image_paths: Iterable[str | os.PathLike], *, batch_size: int = 1,
                  num_workers: int = 0,
                  sort_by_width: bool = False) -> RecognitionReport:
        """识别图片列表，结果恢复为传入顺序。

        image_paths 可以是路径字符串、Path 或两者混合；sort_by_width=True 时
        按宽高比把相近图片组 batch，仅改变推理顺序，不改变输出顺序。
        """
        paths = [os.fspath(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths 不能为空")
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")
        if num_workers < 0:
            raise ValueError("num_workers 不能为负")
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"图片不存在: {path}")

        # 排序只作用于推理顺序；order[position] 记录每个结果应回到的输入下标。
        order = list(range(len(paths)))
        if sort_by_width:
            order.sort(key=lambda index: self._aspect_ratio(paths[index]))
        ordered_paths = [paths[index] for index in order]

        loader = build_inference_loader(
            ordered_paths, self.num_channels, batch_size, num_workers
        )
        results = []
        infer_ms = 0.0
        wall_start = time.perf_counter()

        for images, batch_paths, img_widths in loader:
            infer_start = time.perf_counter()
            # 统一接口：所有后端都接收真实宽度，BiGRU 通过 sequence_lens 跳过 padding。
            preds = self.backend.infer(images, img_widths)
            texts = self.converter.decode(preds, layout="TBC")
            infer_ms += (time.perf_counter() - infer_start) * 1000
            results.extend(
                RecognitionResult(path=path, text=text)
                for path, text in zip(batch_paths, texts)
            )

        wall_seconds = time.perf_counter() - wall_start
        restored = [None] * len(paths)
        for position, result in enumerate(results):
            restored[order[position]] = result
        return RecognitionReport(
            results=restored,
            infer_ms=infer_ms,
            wall_seconds=wall_seconds,
        )

    @staticmethod
    def _aspect_ratio(path: str) -> float:
        with Image.open(path) as image:
            return image.width / max(1, image.height)


__all__ = [
    "BACKEND_CHOICES",
    "RecognitionEngine",
    "RecognitionReport",
    "RecognitionResult",
]
