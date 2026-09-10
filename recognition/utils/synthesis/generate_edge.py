#!/usr/bin/env python3
"""合成贴边、轻微裁切和相邻字残片，默认 1000 张；残片不计入正文标注。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.synthesis import error_common as common
from recognition.utils.synthesis.generate import render_image

# 参考图以右边缘问题为主；每十张分配四张右贴边，两张右轻裁，其余四种各一张。
PATTERNS = ('tight_right', 'tight_right', 'tight_right', 'tight_right',
            'clip_right', 'clip_right', 'tight_left', 'clip_left', 'clip_vertical', 'fragment')


def ink_box(image, style):
    pixels = np.asarray(image).astype(float)
    distance = np.linalg.norm(pixels - np.array(style.background), axis=2)
    threshold = max(12, float(distance.max()) * .3)
    ys, xs = np.where(distance > threshold)
    if not len(xs):
        raise ValueError('No measurable ink in synthetic image')
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_edge(image, text, style, pattern, rng, fragment=None):
    """轻裁最多一像素；窄字、标点仅贴边，避免删除识别所需的主要笔画。"""
    if pattern not in PATTERNS:
        raise ValueError(f'Unknown edge pattern: {pattern}')
    x0, y0, x1, y1 = ink_box(image, style)
    box = [0, 0, image.width, image.height]
    cut = 0
    if pattern.endswith('right'):
        cut = int(pattern == 'clip_right' and '\u4e00' <= text[-1] <= '\u9fff'
                  and y1 - y0 >= 20)
        box[2] = x1 - cut
    elif pattern.endswith('left'):
        cut = int(pattern == 'clip_left' and '\u4e00' <= text[0] <= '\u9fff'
                  and y1 - y0 >= 20)
        box[0] = x0 + cut
    elif pattern == 'clip_vertical':
        cut = int(y1 - y0 >= 24)
        if rng.random() < .5:
            box[1] = y0 + cut
        else:
            box[3] = y1 - cut
    elif pattern == 'fragment':
        if fragment is None:
            raise ValueError('fragment image required for edge fragment pattern')
        # 只混入相邻字最外侧 1 列墨迹，保留间隙和完整目标文字。
        fx0, fy0, fx1, fy1 = ink_box(fragment, style)
        left = rng.random() < .5
        sliver = fragment.crop((fx1 - 1, 0, fx1, fragment.height) if left
                               else (fx0, 0, fx0 + 1, fragment.height))
        gap = rng.randint(1, 3)
        output = Image.new('RGB', (image.width + 1 + gap, image.height), style.background)
        offset = 1 + gap if left else 0
        output.paste(image, (offset, 0))
        output.paste(sliver, (0 if left else output.width - 1, 0))
        return output, {'pattern': pattern, 'side': 'left' if left else 'right',
                        'fragment_width': 1, 'gap': gap, 'body_box': [offset, 0, offset + image.width, image.height]}
    # 一像素也可能承载完整竖笔/横笔；按受影响字区墨迹复核，超限退回贴边。
    if cut:
        pixels = np.asarray(image).astype(float)
        distance = np.linalg.norm(pixels - np.array(style.background), axis=2)
        ink = distance > max(12, float(distance.max()) * .3)
        kept = np.zeros(ink.shape, dtype=bool)
        kept[box[1]:box[3], box[0]:box[2]] = True
        region = np.zeros(ink.shape, dtype=bool)
        if pattern == 'clip_left':
            region[:, x0:min(x1, x0 + y1 - y0)] = True
        elif pattern == 'clip_right':
            region[:, max(x0, x1 - (y1 - y0)):x1] = True
        else:
            region[:] = True
        lost = int((ink & region & ~kept).sum())
        limit = .05 if pattern == 'clip_vertical' else .10
        if lost > max(1, int((ink & region).sum() * limit)):
            if pattern == 'clip_left':
                box[0] -= cut
            elif pattern == 'clip_right':
                box[2] += cut
            elif box[1]:
                box[1] -= cut
            else:
                box[3] += cut
            cut = 0
    return image.crop(box), {'pattern': pattern, 'crop_box': box, 'cut_ink_pixels': cut}


def make_sample(inputs, style, rng, index):
    body, text, face, detail = inputs.body(style, rng)
    pattern = PATTERNS[index % len(PATTERNS)]
    fragment = None
    if pattern == 'fragment':
        chars = sorted(c for c in inputs.keys & face.chars if '\u4e00' <= c <= '\u9fff')
        if not chars:
            raise ValueError(f'No Chinese fragment glyph available: {face.path}')
        char = rng.choice(chars)
        fragment = render_image(char, face, style, rng)
        detail['fragment_character'] = char
    output, effect = apply_edge(body, text, style, pattern, rng, fragment)
    return output, text, face, dict(detail, **effect)


def parse_args(argv=None):
    return common.parse_args('edge', argv)


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
