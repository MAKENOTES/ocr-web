#!/usr/bin/env python3
"""参考低清错误框合成模糊、笔画粘连、破碎和扫描带，默认 1000 张。"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录，由当前文件位置推导，不依赖运行目录。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.synthesis import error_common as common

PATTERNS = ('low_resolution', 'joined_strokes', 'broken_strokes', 'scan_band')  # 四种低清干扰类别，按类别均分样本。


def degrade(image, style, pattern, rng):
    """按字号控制退化；不裁字、不改标签，局部破碎只减淡少量墨迹。"""
    if pattern not in PATTERNS:
        raise ValueError(f'Unknown degradation pattern: {pattern}')
    pixels = np.asarray(image).astype(np.float32)
    background = np.array(style.background, dtype=np.float32)
    detail = {'pattern': pattern}
    if pattern == 'low_resolution':
        scale = rng.uniform(.65, .85)
        radius = rng.uniform(.25, .45)
        small = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        output = small.resize(image.size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius))
        detail.update(scale=scale, blur_radius=radius)
    elif pattern == 'joined_strokes':
        # 2 像素核只混入部分扩展墨迹，避免完整 3×3 膨胀吞掉小字号内部空隙。
        axis = rng.choice(('horizontal', 'vertical'))
        kernel = np.ones((1, 2) if axis == 'horizontal' else (2, 1), np.uint8)
        expanded = cv2.erode(pixels, kernel)
        strength = rng.uniform(.3, .55)
        output = Image.fromarray(np.clip(pixels * (1 - strength) + expanded * strength, 0, 255).astype(np.uint8))
        detail.update(axis=axis, strength=strength)
    else:
        noise = np.random.default_rng(rng.getrandbits(64))
        if pattern == 'broken_strokes':
            ink = np.linalg.norm(pixels - background, axis=2) > 35
            probability = rng.uniform(.035, .07)
            mask = ink & (noise.random(ink.shape) < probability)
            strength = rng.uniform(.45, .75)
            pixels[mask] = pixels[mask] * (1 - strength) + background * strength
            detail.update(dropout_probability=probability, fade_strength=strength)
        else:
            top = rng.randrange(max(1, image.height // 4), max(2, image.height * 3 // 4))
            height = max(1, round(image.height * rng.uniform(.06, .13)))
            strength = rng.uniform(.15, .3)
            # 灰色扫描带同时轻微压暗纸面、减淡正文，模拟扫描横带。
            pixels[top:top + height] = pixels[top:top + height] * (1 - strength) + 175 * strength
            detail.update(band_top=top, band_height=height, band_strength=strength)
        pixels += noise.normal(0, rng.uniform(.6, 1.8), pixels.shape)
        output = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return output, detail


def make_sample(inputs, style, rng, index):
    pattern = PATTERNS[index % len(PATTERNS)]
    body, text, face, detail = inputs.body(style, rng, bold=pattern == 'joined_strokes')
    output, effect = degrade(body, style, pattern, rng)
    return output, text, face, dict(detail, **effect)


def parse_args(argv=None):
    return common.parse_args('degraded', argv)


def generate(args):
    return common.run(args, make_sample)


def main():
    try:
        report = generate(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    print(f'{"Validated" if report.get("dry_run") else "Saved"} {report["images"]} images; {report["annotation"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
