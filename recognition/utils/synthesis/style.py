"""从真实 detection 框估计颜色和留白，不复用含字像素。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np

from recognition.components.preprocessing import load_rgb_image


@dataclass(frozen=True)
class Style:
    background: tuple[int, int, int]
    foreground: tuple[int, int, int]
    height: int
    ink_fraction: float
    side_fraction: float
    length: int
    reference: str

    def to_dict(self) -> dict:
        return asdict(self)


def sample_styles(directories: list[Path], count: int, seed: int) -> list[Style]:
    """按图像数量比例抽样 train/val；坏图带路径报错，纯色框不作参考。"""
    paths = []
    for directory in directories:
        items = sorted(directory.rglob("*.png"))
        if not items:
            raise ValueError(f"No reference PNG images found: {directory}")
        paths.extend(items)
    paths = sorted(set(paths))
    paths = random.Random(seed).sample(paths, min(count, len(paths)))
    styles = []
    for path in paths:
        try:
            image = load_rgb_image(path)
        except Exception as exc:
            raise ValueError(f"Cannot load reference image: {path}: {exc}") from exc
        original_width, original_height = image.size
        image.thumbnail((1024, 128))
        pixels = np.asarray(image).astype(np.float32)
        flat = pixels.reshape(-1, 3)
        # 最常见的量化颜色比边缘中位数更适合文字贴边的小图。
        quantized = (flat // 16).astype(np.int32)
        codes = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
        dominant = np.bincount(codes, minlength=4096).argmax()
        background = np.median(flat[codes == dominant], axis=0)
        distances = np.linalg.norm(pixels - background, axis=2)
        contrast = float(np.percentile(distances, 98))
        if contrast < 35:
            continue
        ink = distances > max(25, contrast * 0.40)
        foreground = np.median(pixels[distances >= contrast * 0.85], axis=0)
        ys, xs = np.where(ink)
        ink_height = int(ys.max() - ys.min() + 1)
        side = min(int(xs.min()), image.width - 1 - int(xs.max())) / image.height
        styles.append(Style(
            tuple(int(v) for v in background),
            tuple(int(v) for v in foreground),
            int(np.clip(original_height, 16, 64)),
            float(np.clip(ink_height / image.height, 0.65, 0.95)),
            float(np.clip(side, 0.02, 0.35)),
            int(np.clip(round(original_width / original_height / 0.85), 1, 24)),
            str(path),
        ))
    if not styles:
        raise ValueError(f"No readable text styles found in: {directories}")
    return styles
