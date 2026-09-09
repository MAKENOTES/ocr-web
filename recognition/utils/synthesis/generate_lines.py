#!/usr/bin/env python3
"""从 keys.txt 随机取字，参考验证框合成四个方向的表格边线干扰。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.components.dataset import load_rgb_image
from recognition.utils.annotation import format_annotation_line
from recognition.utils.character_stats import load_keys
from recognition.utils.synthesis.generate import (
    DATA, default_font_paths, ensure_empty_outputs, load_fonts, render_image, sha256,
)
from recognition.utils.synthesis.generate_red_stamp import reference_paths
from recognition.utils.synthesis.naming import REPORT_ROOT
from recognition.utils.synthesis.style import Style

PATTERNS = {
    "top": ("top",), "bottom": ("bottom",),
    "left": ("left",), "right": ("right",),
    "all": ("top", "bottom", "left", "right"),
}


def measure_style(path: Path) -> Style:
    """估计纸面、墨色和字号；长边线不计入正文高度和留白统计。"""
    image = load_rgb_image(path)
    pixels = np.asarray(image)
    gray = pixels.mean(axis=2)
    background = np.median(pixels[gray >= np.percentile(gray, 65)], axis=0)
    ink = (gray < background.mean() - 45).astype(np.uint8)
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, max(8, round(image.width * 0.65))), np.uint8))
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((max(8, round(image.height * 0.8)), 1), np.uint8))
    body = (ink > 0) & ~(horizontal.astype(bool) | vertical.astype(bool))
    if body.sum() < 5:
        raise ValueError(f"No measurable body text in reference: {path}")
    ys, xs = np.where(body)
    foreground = np.median(pixels[body & (gray <= np.percentile(gray[body], 40))], axis=0)
    fraction = float(np.clip((ys.max() - ys.min() + 1) / image.height, 0.45, 0.85))
    return Style(
        tuple(int(v) for v in background), tuple(int(v) for v in foreground),
        int(np.clip(image.height, 16, 64)), fraction,
        float(np.clip(min(xs.min(), image.width - 1 - xs.max()) / image.height, 0.02, 0.3)),
        int(np.clip(round((xs.max() - xs.min() + 1) / (image.height * fraction)), 2, 24)),
        str(path),
    )


def add_lines(image: Image.Image, style: Style, pattern: str,
              rng: random.Random) -> tuple[Image.Image, dict]:
    """在完整正文外添加细边线：上下水平，左右垂直，不裁字或覆盖正文。"""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown line pattern: {pattern}")
    sides = PATTERNS[pattern]
    thickness = rng.choices((1, 2), weights=(9, 1), k=1)[0]
    gap = rng.randint(0, 2)
    pad = thickness + gap + 1
    left, right, top, bottom = [pad if side in sides else 0
                                for side in ("left", "right", "top", "bottom")]
    output = Image.new("RGB", (image.width + left + right, image.height + top + bottom),
                       style.background)
    output.paste(image, (left, top))
    # 独立透明线层先轻微模糊，再叠加，保持原正文像素不变。
    mask = Image.new("L", output.size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = output.size
    boxes = {
        "top": (0, 0, w - 1, thickness - 1),
        "bottom": (0, h - thickness, w - 1, h - 1),
        "left": (0, 0, thickness - 1, h - 1),
        "right": (w - thickness, 0, w - 1, h - 1),
    }
    for side in sides:
        draw.rectangle(boxes[side], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.3)))
    strength = rng.uniform(0.65, 1.0)
    color = tuple(round(bg + strength * (fg - bg))
                  for bg, fg in zip(style.background, style.foreground))
    output.paste(color, (0, 0, w, h), mask)
    return output, {"pattern": pattern, "sides": list(sides), "thickness": thickness,
                    "gap": gap, "color": color, "body_box": [left, top, left + image.width, top + image.height]}


def generate(args: argparse.Namespace) -> dict:
    if args.images <= 0:
        raise ValueError("--images must be positive")
    ensure_empty_outputs(args.output_dir, args.annotation, args.report_dir)
    training = args.annotation.with_name(args.annotation.stem + "_training.txt")
    partials = [p.with_name(p.name + ".partial") for p in (args.annotation, training)]
    for path in (training, *partials):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    paths = reference_paths(args.references, args.data_dir)
    styles = [measure_style(path) for path in paths]
    keys = load_keys(args.keys)
    visible_keys = [char for char in keys if char.isprintable() and not char.isspace()]
    if not visible_keys:
        raise ValueError(f"No visible characters in dictionary: {args.keys}")
    fonts = load_fonts(args.fonts or default_font_paths(), visible_keys)
    pools = [[char for char in visible_keys if char in face.chars] for face in fonts]
    # 参考主要为细宋体，仍保留少量黑体及粗体变化。
    weights = [(6 if "Serif" in f.name or "UMing" in f.name else 1)
               * (0.15 if "Bold" in f.path.stem else 1) for f in fonts]
    rng = random.Random(args.seed)
    patterns = [list(PATTERNS)[i % len(PATTERNS)] for i in range(args.images)]
    references = [i % len(styles) for i in range(args.images)]
    rng.shuffle(patterns)
    rng.shuffle(references)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.annotation.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    counts, character_counts, reference_counts = Counter(), Counter(), Counter()
    preview = Image.new("RGB", (1200, 1050), "#dedede")
    draw = ImageDraw.Draw(preview)
    preview_count = 0
    interval = max(1, args.images // 60)
    with partials[0].open("x", encoding="utf-8") as labels, \
            partials[1].open("x", encoding="utf-8") as train_labels, \
            (args.report_dir / "samples.jsonl").open("x", encoding="utf-8") as records:
        for i, (pattern, reference) in enumerate(zip(patterns, references)):
            style = styles[reference]
            face_index = rng.choices(range(len(fonts)), weights=weights, k=1)[0]
            face = fonts[face_index]
            length = max(2, min(24, style.length + rng.choice((-1, 0, 0, 1))))
            text = "".join(rng.choices(pools[face_index], k=length))
            body = render_image(text, face, style, rng)
            image, lines = add_lines(body, style, pattern, rng)
            filename = f"synthesis_lines_{args.date}_{i + 1:05d}.png"
            path = args.output_dir / filename
            with path.open("xb") as handle:
                image.save(handle, format="PNG")
            relative = Path(os.path.relpath(path, args.annotation.parent)).as_posix()
            # 同时满足仓库指南的逗号引号格式，以及当前训练加载器的 Tab 格式。
            quoted = text.replace('"', '""')
            labels.write(f'{relative}, "{quoted}"\n')
            train_labels.write(format_annotation_line(relative, text))
            counts[pattern] += 1
            character_counts.update(text)
            reference_counts[style.reference] += 1
            records.write(json.dumps({"file": relative, "text": text, "font": str(face.path),
                                      "font_index": face.index, "style": style.to_dict(),
                                      "lines": lines}, ensure_ascii=False) + "\n")
            if i % interval == 0 and preview_count < 60:
                thumb = image.copy()
                thumb.thumbnail((288, 48))
                x, y = (preview_count % 4) * 300 + 4, (preview_count // 4) * 70
                draw.text((x, y + 2), f"{i + 1:05d} {pattern}", fill="black")
                preview.paste(thumb, (x, y + 20))
                preview_count += 1
            if (i + 1) % 200 == 0:
                print(f"Generated {i + 1}/{args.images}", flush=True)
    preview.save(args.report_dir / "preview.png")
    summary = {"images": args.images, "seed": args.seed, "patterns": dict(counts),
               "output_dir": str(args.output_dir.resolve()), "annotation": str(args.annotation.resolve()),
               "training_annotation": str(training.resolve()), "references": dict(reference_counts),
               "keys": str(args.keys.resolve()), "keys_sha256": sha256(args.keys),
               "reference_sha256": {str(p): sha256(p) for p in paths},
               "characters": dict(character_counts),
               "styles": [s.to_dict() for s in styles]}
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for partial, destination in zip(partials, (args.annotation, training)):
        partial.rename(destination)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--keys", type=Path, default=ROOT / "recognition/keys.txt")
    parser.add_argument("--references", type=Path, default=Path(__file__).with_name("line_references.txt"))
    parser.add_argument("--fonts", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    args.date = datetime.now().strftime("%Y%m%d")
    prefix = f"synthesis_lines_{args.date}"
    args.output_dir = args.output_dir or args.data_dir / f"{prefix}_images"
    args.annotation = args.annotation or args.data_dir / f"{prefix}_annotation.txt"
    args.report_dir = args.report_dir or REPORT_ROOT / prefix
    return args


def main() -> int:
    try:
        summary = generate(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {summary['images']} images; training labels: {summary['training_annotation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
