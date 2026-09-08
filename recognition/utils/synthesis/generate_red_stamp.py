#!/usr/bin/env python3
"""参考指定 train/val 红章框，重绘正文并叠加局部红章，默认生成 1000 张。"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.components.dataset import load_rgb_image
from recognition.utils.annotation import format_annotation_line, split_annotation_fields
from recognition.utils.character_stats import load_keys
from recognition.utils.synthesis.generate import (
    DATA, FontFace, default_font_paths, ensure_empty_outputs, get_font,
    load_fonts, render_image, sha256,
)
from recognition.utils.synthesis.style import Style
from recognition.utils.synthesis.naming import REPORT_ROOT


@dataclass(frozen=True)
class StampStyle:
    body: Style
    aspect: float
    color: tuple[int, int, int]
    opacity: float
    red_fraction: float
    diameter_ratio: float
    red_measured: bool = True


def reference_paths(manifest: Path, data_dir: Path) -> list[Path]:
    paths = []
    for number, raw in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        path = data_dir / entry
        if path.suffix != ".png" or not path.is_file():
            raise FileNotFoundError(f"Reference PNG does not exist: {manifest}:{number}: {path}")
        paths.append(path.resolve())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(f"Reference manifest must be nonempty with unique paths: {manifest}")
    return paths


def red_mask(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float32)
    return ((values[..., 0] - np.maximum(values[..., 1], values[..., 2]) > 20)
            & (values[..., 0] > 70))


def measure_style(path: Path) -> StampStyle:
    """只保留颜色、尺度和覆盖率等数值；正文颜色估计排除红章像素。"""
    image = load_rgb_image(path)
    width, height = image.size
    image.thumbnail((2048, 128))
    pixels = np.asarray(image).astype(np.float32)
    red = red_mask(pixels)
    gray = pixels.mean(axis=2)
    background = np.median(pixels[gray >= np.percentile(gray, 75)], axis=0)
    ink = (~red) & (gray < background.mean() - 45)
    if ink.sum() < 8:
        raise ValueError(f"No measurable body text in reference: {path}")
    dark = ink & (gray <= np.percentile(gray[ink], 30))
    foreground = np.median(pixels[dark], axis=0)
    ys, xs = np.where(ink)
    ink_fraction = float(np.clip((ys.max() - ys.min() + 1) / image.height, 0.65, 0.95))
    side = min(xs.min(), image.width - 1 - xs.max()) / image.height
    strength = pixels[..., 0] - np.maximum(pixels[..., 1], pixels[..., 2])
    red_measured = bool(red.sum() >= 8)
    if red_measured:
        strong = red & (strength >= np.percentile(strength[red], 80))
        color = np.median(pixels[strong], axis=0)
        color[1:] = np.minimum(color[1:], (25, 45))
        observed = np.median(pixels[red], axis=0)
        opacity = float(np.clip((background[1:].mean() - observed[1:].mean())
                               / max(1, background[1:].mean() - color[1:].mean()), 0.45, 0.98))
        red_x = np.where(red)[1]
        red_span = np.percentile(red_x, 98) - np.percentile(red_x, 2)
    else:
        color, opacity, red_span = (0, 0, 0), 0.0, 0.0
    return StampStyle(
        Style(tuple(int(v) for v in background), tuple(int(v) for v in foreground),
              int(np.clip(height, 16, 64)), ink_fraction, float(np.clip(side, 0.02, 0.35)),
              max(1, round(width / height / ink_fraction)), str(path)),
        width / height, tuple(int(v) for v in color), opacity,
        float(red.mean()), float(np.clip(red_span / image.height, 3.5, 12)), red_measured,
    )


def measure_styles(paths: list[Path]) -> list[StampStyle]:
    styles = [measure_style(path) for path in paths]
    measured = [style for style in styles if style.red_measured]
    if not measured:
        raise ValueError(f"No measurable red stamps in references: {paths}")
    for index, style in enumerate(styles):
        if not style.red_measured:
            # 有的指定框只有灰色水印：保留其正文风格，红章参数取其余参考的中位数。
            styles[index] = replace(
                style, color=tuple(int(v) for v in np.median([s.color for s in measured], axis=0)),
                opacity=float(np.median([s.opacity for s in measured])),
                red_fraction=float(np.median([s.red_fraction for s in measured])),
                diameter_ratio=float(np.median([s.diameter_ratio for s in measured])),
            )
            print(f"Body-only reference (no red stamp); using median stamp style: {style.body.reference}", flush=True)
    return styles


def text_units(text: str) -> float:
    return sum(1.0 if unicodedata.east_asian_width(char) in ("W", "F") else 0.55 for char in text)


def load_corpus(path: Path, keys: set[str], fonts: list[FontFace]) -> list[tuple[float, str, int, list[int]]]:
    """正文只取 train 标注文本，重新渲染；严格检查字典及整行字体覆盖。"""
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            _, text = split_annotation_fields(line, path, number)
            text = text.strip()
            if not text:
                raise ValueError(f"Empty corpus label: {path}:{number}")
            missing = set(text) - keys
            if missing:
                raise ValueError(f"Corpus characters absent from keys: {path}:{number}: {sorted(missing)!r}")
            choices = [i for i, face in enumerate(fonts) if set(text) <= face.chars]
            if not choices:
                raise ValueError(f"No font covers the whole corpus label: {path}:{number}: {text!r}")
            records.append((text_units(text), text, number, choices))
    if not records:
        raise ValueError(f"Empty corpus: {path}")
    return sorted(records)


def stamp_mask(diameter: int, face: FontFace, text: str, rng: random.Random) -> Image.Image:
    """程序绘制圆环、沿弧章字和五角星；章字是干扰，不属于正文标注。"""
    if not set(text) <= face.chars:
        raise ValueError(f"Stamp font does not cover text: {face.path}: {text!r}")
    mask = Image.new("L", (diameter, diameter), 0)
    draw = ImageDraw.Draw(mask)
    center = diameter / 2
    margin = max(2, round(diameter * 0.025))
    draw.ellipse((margin, margin, diameter - margin - 1, diameter - margin - 1),
                 outline=255, width=max(1, round(diameter * rng.uniform(0.009, 0.016))))
    font_size = max(10, round(diameter * min(0.115, 1.6 / len(text))))
    font = get_font(face.path, face.index, font_size)
    rotation = rng.uniform(-25, 25)
    for index, char in enumerate(text):
        angle = -160 + 140 * index / max(1, len(text) - 1) + rotation
        radius = diameter * 0.355
        x = center + radius * math.cos(math.radians(angle))
        y = center + radius * math.sin(math.radians(angle))
        glyph = Image.new("L", (font_size * 2, font_size * 2), 0)
        ImageDraw.Draw(glyph).text((font_size, font_size), char, font=font, fill=255, anchor="mm")
        glyph = glyph.rotate(-angle - 90, resample=Image.Resampling.BICUBIC, expand=True)
        mask.paste(glyph, (round(x - glyph.width / 2), round(y - glyph.height / 2)), glyph)
    points = []
    for index in range(10):
        angle = math.radians(-90 + index * 36 + rotation)
        radius = diameter * (0.15 if index % 2 == 0 else 0.06)
        points.append((center + radius * math.cos(angle), center + radius * math.sin(angle)))
    draw.polygon(points, fill=255)
    # 下弧增加同一合成章文字，避免中心截面只剩两个圆环边缘。
    bottom_font = get_font(face.path, face.index, max(10, round(diameter * 0.07)))
    draw.text((center, diameter * 0.76), text[-4:], font=bottom_font, fill=255, anchor="mm")
    return mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))


def overlay_stamp(image: Image.Image, style: StampStyle, face: FontFace, text: str,
                  rng: random.Random) -> tuple[Image.Image, dict]:
    """裁出圆章局部叠入正文，有限次搜索覆盖率接近参考且确实与正文相交的位置。"""
    pixels = np.asarray(image).astype(np.float32)
    ink = pixels.mean(axis=2) < np.mean(style.body.background) - 35
    if not ink.any():
        raise ValueError("Rendered body has no measurable text ink")
    diameter = max(48, round(image.height * style.diameter_ratio * rng.uniform(0.85, 1.15)))
    mask = stamp_mask(diameter, face, text, rng)
    best = None
    target = float(np.clip(style.red_fraction * rng.uniform(0.8, 1.2), 0.015, 0.30))
    # 先固定截面，再优化位置，避免覆盖率打分总选中面积较大的中心星形。
    section = rng.choices(("top", "middle", "bottom"), weights=(5, 3, 2), k=1)[0]
    for _ in range(32):
        x = round(rng.uniform(0.12, 0.88) * image.width - diameter / 2)
        if section == "top":
            y = round(-diameter * rng.uniform(0.035, 0.24) + image.height / 2)
        elif section == "bottom":
            y = round(-diameter * rng.uniform(0.70, 0.98) + image.height / 2)
        else:
            y = round(-diameter * rng.uniform(0.30, 0.65) + image.height / 2)
        layer = Image.new("L", image.size, 0)
        layer.paste(mask, (x, y))
        alpha = np.asarray(layer).astype(np.float32) / 255
        active = alpha > 0.15
        overlap = float((active & ink).sum() / ink.sum())
        coverage = float(active.mean())
        if overlap < 0.01 or coverage < 0.01 or coverage > 0.45:
            continue
        score = abs(coverage - target)
        if best is None or score < best[0]:
            best = (score, alpha, coverage, overlap, x, y, section)
    if best is None:
        # 极细长行在固定截面的随机位置可能没有足够墨迹；将星形中心对准正文中位位置。
        ys, xs = np.where(ink)
        x, y = round(float(np.median(xs)) - diameter / 2), round(float(np.median(ys)) - diameter / 2)
        layer = Image.new("L", image.size, 0)
        layer.paste(mask, (x, y))
        alpha = np.asarray(layer).astype(np.float32) / 255
        active = alpha > 0.15
        overlap = float((active & ink).sum() / ink.sum())
        coverage = float(active.mean())
        if overlap < 0.01 or not 0.01 <= coverage <= 0.45:
            raise RuntimeError(f"Could not place a visible overlapping stamp: {image.size}; {style.body.reference}")
        best = (abs(coverage - target), alpha, coverage, overlap, x, y, "middle")
    _, alpha, coverage, overlap, x, y, section = best
    noise = np.random.default_rng(rng.getrandbits(64))
    alpha *= noise.uniform(0.90, 1.0, alpha.shape).astype(np.float32)
    opacity = float(np.clip(style.opacity * rng.uniform(0.9, 1.15), 0.40, 0.98))
    alpha = alpha[..., None] * opacity
    color = np.array(style.color, dtype=np.float32)
    # 以乘色为主模拟印泥透出黑字，少量普通混合模拟扫描后的遮盖。
    stamped = 0.85 * pixels * color / 255 + 0.15 * color
    output = Image.fromarray(np.clip(pixels * (1 - alpha) + stamped * alpha, 0, 255).astype(np.uint8))
    if red_mask(np.asarray(output)).sum() < 8:
        raise RuntimeError(f"Stamp is not visibly red after compositing: {style.body.reference}")
    return output, {"diameter": diameter, "position": [x, y], "section": section,
                    "opacity": opacity, "mask_fraction": coverage, "body_overlap_fraction": overlap,
                    "red_fraction": float(red_mask(np.asarray(output)).mean()), "stamp_text": text}


def generate(args: argparse.Namespace) -> dict:
    if args.images <= 0 or not args.stamp_text.strip():
        raise ValueError("--images must be positive and --stamp-text must be nonempty")
    ensure_empty_outputs(args.output_dir, args.annotation, args.report_dir)
    partial = args.annotation.with_name(args.annotation.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Partial annotation already exists: {partial}")
    paths = reference_paths(args.references, args.data_dir)
    styles = measure_styles(paths)
    keys = set(load_keys(args.keys))
    fonts = load_fonts(args.fonts or default_font_paths(), sorted(keys | set(args.stamp_text)))
    stamp_fonts = [face for face in fonts if set(args.stamp_text) <= face.chars]
    if not stamp_fonts:
        raise ValueError("No font covers --stamp-text")
    corpus = load_corpus(args.corpus, keys, fonts)
    units = [record[0] for record in corpus]
    rng = random.Random(args.seed)
    # 均匀使用指定参考，图数足够时保证每张参考至少被采样一次。
    selected_styles = [styles[index % len(styles)] for index in range(args.images)]
    rng.shuffle(selected_styles)
    plan = []
    for style in selected_styles:
        desired = style.aspect / style.body.ink_fraction
        low, high = bisect_left(units, desired * 0.80), bisect_right(units, desired * 1.20)
        if low == high:
            closest = min(range(len(units)), key=lambda i: abs(units[i] - desired))
            low, high = closest, closest + 1
        record = corpus[rng.randrange(low, high)]
        choices = record[3]
        weights = [(5 if "Serif" in fonts[i].name or "UMing" in fonts[i].name else 2)
                   * (0.35 if "Bold" in fonts[i].path.stem else 1) for i in choices]
        plan.append((style, record, rng.choices(choices, weights=weights, k=1)[0]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.annotation.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    preview = Image.new("RGB", (1600, 40 * 86), "#dedede")
    preview_draw = ImageDraw.Draw(preview)
    lengths: Counter[int] = Counter()
    references: Counter[str] = Counter()
    sections: Counter[str] = Counter()
    preview_count = 0
    red_fractions = []
    print(f"Generating {args.images} red-stamp images from {len(styles)} reference styles", flush=True)
    with partial.open("x", encoding="utf-8") as labels, \
            (args.report_dir / "samples.jsonl").open("x", encoding="utf-8") as metadata:
        for number, (style, record, face_index) in enumerate(plan, 1):
            _, text, source_line, _ = record
            face = fonts[face_index]
            clean = render_image(text, face, style.body, rng)
            stamp_face = rng.choice(stamp_fonts)
            image, stamp = overlay_stamp(clean, style, stamp_face, args.stamp_text, rng)
            filename = f"{args.date}_{number:05d}.png"
            path = args.output_dir / filename
            with path.open("xb") as handle:
                image.save(handle, format="PNG")
            relative = Path(os.path.relpath(path, args.annotation.parent)).as_posix()
            labels.write(format_annotation_line(relative, text))
            metadata.write(json.dumps({
                "file": relative, "text": text, "corpus_line": source_line,
                "font": str(face.path), "font_index": face.index,
                "stamp_font": str(stamp_face.path), "stamp_font_index": stamp_face.index,
                "style": asdict(style), "stamp": stamp,
            }, ensure_ascii=False) + "\n")
            lengths[len(text)] += 1
            references[style.body.reference] += 1
            sections[stamp["section"]] += 1
            red_fractions.append(stamp["red_fraction"])
            if (number - 1) % max(1, args.images // 40) == 0 and preview_count < 40:
                thumbnail = image.copy()
                thumbnail.thumbnail((1580, 60))
                y = preview_count * 86
                preview_draw.text((5, y + 2), f"{filename}  chars={len(text)}  stamp={stamp['section']}", fill="black")
                preview.paste(thumbnail, (5, y + 22))
                preview_count += 1
            if number % 100 == 0 or number == args.images:
                print(f"Completed {number}/{args.images}", flush=True)
    preview.crop((0, 0, 1600, preview_count * 86)).save(args.report_dir / "preview.png")
    report = {
        "images": args.images, "date": args.date, "seed": args.seed,
        "annotation": str(args.annotation), "output_dir": str(args.output_dir),
        "annotation_format": "relative/path.png<TAB>body text; stamp text excluded",
        "corpus": str(args.corpus), "corpus_sha256": sha256(args.corpus),
        "keys_sha256": sha256(args.keys), "reference_manifest_sha256": sha256(args.references),
        "references": dict(references), "reference_sha256": {str(path): sha256(path) for path in paths},
        "reference_styles": [asdict(style) for style in styles],
        "font_sha256": {str(face.path): sha256(face.path) for face in fonts},
        "length_histogram": dict(sorted(lengths.items())), "sections": dict(sections),
        "red_fraction_mean": sum(red_fractions) / len(red_fractions),
    }
    with (args.report_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.link(partial, args.annotation)
    partial.unlink()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--references", type=Path, default=Path(__file__).with_name("red_stamp_references.txt"),
                        help="参考 PNG 清单，每行路径相对于 --data-dir")
    parser.add_argument("--corpus", type=Path, help="正文文本来源，默认 train_annotation.txt")
    parser.add_argument("--keys", type=Path, default=ROOT / "recognition/keys.txt")
    parser.add_argument("--fonts", type=Path, nargs="+")
    parser.add_argument("--stamp-text", default="合成样本识别测试专用章", help="红章干扰文字，不加入正文标注")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    args.date = datetime.now().strftime("%Y%m%d")
    prefix = f"synthesis_red_{args.date}"
    args.corpus = args.corpus or args.data_dir / "train_annotation.txt"
    args.output_dir = args.output_dir or args.data_dir / f"{prefix}_images"
    args.annotation = args.annotation or args.data_dir / f"{prefix}_annotation.txt"
    args.report_dir = args.report_dir or REPORT_ROOT / prefix
    return args


def main() -> int:
    try:
        report = generate(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {report['images']} images; annotations: {report['annotation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
