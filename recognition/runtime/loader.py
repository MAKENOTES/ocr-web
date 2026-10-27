"""推理输入准备：收集图片并构建共享预处理 DataLoader。

图片读取、透明合成、去章、等比缩放和动态补边由 runtime 内部模块提供，
保证训练、评估、ONNX、OpenVINO 与 PyTorch 看到相同的输入。
"""

import os
from pathlib import Path

from torch.utils.data import DataLoader

from .backends import IMG_HEIGHT
from .dataset import InferenceDataset, dynamic_padding_collate_fn

# 支持的图片扩展名；目录扫描时会同时匹配小写和大写形式。
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def collect_image_paths(image_path: str | os.PathLike) -> list[str]:
    """返回单张图片或目录下所有受支持图片的排序去重路径。"""
    path = Path(image_path)
    if path.is_dir():
        paths = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(str(candidate) for candidate in path.glob(f"*{extension}"))
            paths.extend(str(candidate) for candidate in path.glob(f"*{extension.upper()}"))
        if not paths:
            raise FileNotFoundError(f"目录中没有受支持的图片: {image_path}")
        return sorted(set(paths))
    if path.is_file():
        return [str(path)]
    raise FileNotFoundError(f"未找到图像文件: {image_path}")


def build_inference_loader(image_paths: list[str], num_channels: int, batch_size: int,
                           num_workers: int) -> DataLoader:
    """构建不打乱顺序的推理 DataLoader，返回 (图片, 路径, 真实宽度)。"""
    dataset = InferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=num_channels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dynamic_padding_collate_fn,
    )
