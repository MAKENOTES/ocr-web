#!/usr/bin/env python3
"""为 keys-append 的每个字生成固定数量 PNG，支持直接运行或 python -m。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.annotation import format_annotation_line
from recognition.utils.synthesis.style import Style, sample_styles

DATA = ROOT / "recognition/data"
UPSTREAM_COMMIT = "a08835263289c9488ba8c24abacc321318c3f1e0"


@dataclass(frozen=True)
class FontFace:
    path: Path
    index: int
    name: str
    chars: frozenset[str]


def read_chars(path: Path) -> list[str]:
    chars = path.read_text(encoding="utf-8-sig").splitlines()
    if not chars:
        raise ValueError(f"Empty character file: {path}")
    for number, char in enumerate(chars, 1):
        if len(char) != 1 or not char.isprintable() or char.isspace():
            raise ValueError(f"Expected one visible Unicode character: {path}:{number}")
    if len(chars) != len(set(chars)):
        raise ValueError(f"Duplicate characters in: {path}")
    return chars


def default_font_paths() -> list[Path]:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    return [Path(p) for p in candidates if Path(p).is_file()]


def load_fonts(paths: list[Path], targets: list[str]) -> list[FontFace]:
    faces = []
    for path in sorted(set(p.resolve() for p in paths)):
        if not path.is_file():
            raise FileNotFoundError(f"Font does not exist: {path}")
        indices = [0]
        if path.suffix.lower() == ".ttc":
            collection = TTCollection(path, lazy=True)
            indices = list(range(len(collection.fonts)))
            collection.close()
        for index in indices:
            with TTFont(path, fontNumber=index, lazy=True) as font:
                name = font["name"].getDebugName(1) or path.stem
                # Noto 集合同时包含日/韩/港台字形，默认只使用简体中文的比例字体。
                if "CJK" in name and (" SC" not in name or "Mono" in name):
                    continue
                if "UMing" in name and " CN" not in name:
                    continue
                cmap = font.getBestCmap() or {}
                chars = frozenset(chr(code) for code, glyph in cmap.items()
                                  if glyph != ".notdef")
            if set(targets) & chars:
                faces.append(FontFace(path, index, name, chars))
    supported = set().union(*(face.chars for face in faces))
    missing = set(targets) - supported
    if missing:
        raise ValueError("Fonts do not cover target characters: " + "".join(sorted(missing)))
    return faces


def companion_pool(keys: list[str], targets: list[str], face: FontFace) -> list[str]:
    excluded = set(targets)
    return [char for char in keys if char not in excluded and char in face.chars]


def make_text(target: str, companions: list[str], length: int, rng: random.Random) -> str:
    """配字排除所有目标字，使逐字图片数和出现次数均严格等于配额。"""
    if length < 1:
        raise ValueError("Text length must be positive")
    if target in companions:
        raise ValueError(f"Target character included in companions: {target}")
    if length > 1 and not companions:
        raise ValueError(f"No companion characters for target: {target}")
    chars = rng.choices(companions, k=length - 1) if length > 1 else []
    chars.insert(rng.randrange(length), target)
    return "".join(chars)


@lru_cache(maxsize=96)
def get_font(path: Path, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=index)


def render_image(text: str, face: FontFace, style: Style, rng: random.Random) -> Image.Image:
    # 使用上游实际渲染引擎调用的同一个字形绘制 API；避免自动换字/无限重试。
    from text_renderer.utils.draw_utils import draw_text_on_bg
    from text_renderer.utils.font_text import FontText

    if not set(text) <= face.chars:
        raise ValueError(f"Font {face.path} does not support: {text!r}")
    height = max(16, min(64, style.height + rng.randint(-2, 2)))
    ink_height = max(10, round(height * style.ink_fraction))
    font_size = max(12, round(ink_height * 1.08))
    mask = draw_text_on_bg(
        FontText(get_font(face.path, face.index, font_size), text, str(face.path)),
        (*style.foreground, 255),
        char_spacing=-1,
    )
    if mask.getchannel("A").getbbox() is None:
        raise ValueError(f"Empty rendered text: {face.path}: {text!r}")
    # 按完整文字 mask 等比缩放；不裁剪字形，也不在图片内放无标注文字。
    width = max(1, round(mask.width * ink_height / mask.height))
    mask = mask.resize((width, ink_height), Image.Resampling.LANCZOS)
    left = max(1, round(height * style.side_fraction) + rng.randint(0, 2))
    right = max(1, round(height * style.side_fraction) + rng.randint(0, 3))
    top = rng.randint(1, max(1, height - ink_height - 1))
    image = Image.new("RGB", (width + left + right, height), style.background)
    image.paste(mask, (left, top), mask)
    if rng.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.45)))
    if rng.random() < 0.12:
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        pixels = np.asarray(image).astype(np.float32)
        pixels += noise_rng.normal(0, rng.uniform(0.3, 1.2), pixels.shape)
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return image


def ensure_empty_outputs(output_dir: Path, annotation: Path, report_dir: Path) -> None:
    for directory in (output_dir, report_dir):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"Output directory must be empty: {directory}")
    if annotation.exists():
        raise FileExistsError(f"Annotation already exists: {annotation}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(args: argparse.Namespace) -> dict:
    if args.per_char <= 0 or args.style_samples <= 0:
        raise ValueError("--per-char and --style-samples must be positive")
    ensure_empty_outputs(args.output_dir, args.annotation, args.report_dir)
    targets, keys = read_chars(args.targets), read_chars(args.keys)
    target_set = set(targets)
    missing = set(targets) - set(keys)
    if missing:
        raise ValueError(f"Targets absent from {args.keys}: {''.join(sorted(missing))}")
    try:
        version = importlib.metadata.version("text_renderer")
        from text_renderer.utils.draw_utils import draw_text_on_bg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install text_renderer using synthesis/requirements.txt with --no-deps") from exc
    fonts = load_fonts(args.fonts or default_font_paths(), targets)
    pools = [companion_pool(keys, targets, face) for face in fonts]
    if any(not pool for pool in pools):
        raise ValueError("Each font must support at least one non-target companion character")
    styles = sample_styles(args.references, args.style_samples, args.seed)
    rng = random.Random(args.seed)
    font_options = [[i for i, face in enumerate(fonts) if char in face.chars]
                    for char in targets]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.annotation.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # 标注仅在全部成功后发布；中断保留已有产物用于排查，重跑拒绝覆盖。
    partial = args.annotation.with_name(args.annotation.name + ".partial")
    counts: Counter[str] = Counter()
    font_counts: Counter[str] = Counter()
    preview = Image.new("RGB", (1200, 1056), "#dedede")
    preview_draw = ImageDraw.Draw(preview)
    preview_count = 0
    preview_interval = max(1, len(targets) * args.per_char // 60)
    records = args.report_dir / "samples.jsonl"
    print(f"Generating {len(targets)} × {args.per_char} = {len(targets) * args.per_char} images; "
          f"{len(fonts)} font faces, {len(styles)} reference styles", flush=True)
    with partial.open("x", encoding="utf-8") as labels, records.open("x", encoding="utf-8") as metadata:
        for target_index, target in enumerate(targets):
            for sample_index in range(args.per_char):
                choices = font_options[target_index]
                # 常规字重为主，粗体为辅，与真实截图/文档样本相近。
                weights = [1 if "Bold" in fonts[i].path.stem else 3 for i in choices]
                face_index = rng.choices(choices, weights=weights, k=1)[0]
                face = fonts[face_index]
                style = rng.choice(styles)
                length = style.length
                text = make_text(target, pools[face_index], length, rng)
                image = render_image(text, face, style, rng)
                filename = f"u{ord(target):06x}_{sample_index:03d}.png"
                path = args.output_dir / filename
                with path.open("xb") as image_file:
                    image.save(image_file, format="PNG")
                relative = Path(os.path.relpath(path, args.annotation.parent)).as_posix()
                labels.write(format_annotation_line(relative, text))
                counts.update(char for char in text if char in target_set)
                font_counts[f"{face.path.name}#{face.index}"] += 1
                metadata.write(json.dumps({"file": relative, "text": text, "target": target,
                                           "font": str(face.path), "font_index": face.index,
                                           "style": style.to_dict()}, ensure_ascii=False) + "\n")
                number = target_index * args.per_char + sample_index
                if number % preview_interval == 0 and preview_count < 60:
                    thumbnail = image.copy()
                    thumbnail.thumbnail((288, 48))
                    x, y = (preview_count % 4) * 300, (preview_count // 4) * 70
                    preview_draw.text((x + 3, y + 2), filename, fill="black")
                    preview.paste(thumbnail, (x + 3, y + 19))
                    preview_count += 1
            if (target_index + 1) % 10 == 0 or target_index + 1 == len(targets):
                print(f"Completed {target_index + 1}/{len(targets)} characters "
                      f"({(target_index + 1) * args.per_char} images)", flush=True)
    if counts != Counter({char: args.per_char for char in targets}):
        raise RuntimeError("Generated target character counts do not match quotas")
    preview.save(args.report_dir / "preview.png")
    report = {
        "images": sum(counts.values()), "target_characters": len(targets),
        "per_char": args.per_char, "counts": dict(counts), "seed": args.seed,
        "renderer": "oh-my-ocr/text_renderer.utils.draw_utils.draw_text_on_bg",
        "renderer_version": version, "tested_upstream_commit": UPSTREAM_COMMIT,
        "targets": str(args.targets), "targets_sha256": sha256(args.targets),
        "keys": str(args.keys), "keys_sha256": sha256(args.keys),
        "output_dir": str(args.output_dir), "annotation": str(args.annotation),
        "annotation_format": "relative/path.png<TAB>text",
        "reference_directories": [str(path) for path in args.references],
        "reference_styles": len(styles), "fonts": dict(font_counts),
        "font_sha256": {str(face.path): sha256(face.path) for face in fonts},
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 同目录硬链接原子发布完整标注；目标已存在时失败，不会覆盖。
    os.link(partial, args.annotation)
    partial.unlink()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "recognition/keys-append.txt")
    parser.add_argument("--keys", type=Path, default=ROOT / "recognition/keys.txt")
    parser.add_argument("--per-char", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--references", type=Path, nargs="+",
                        default=[DATA / "train_images", DATA / "val_images"])
    parser.add_argument("--style-samples", type=int, default=1600)
    parser.add_argument("--fonts", type=Path, nargs="+", help="TTF/OTF/TTC 字体文件；默认本机中文字体")
    parser.add_argument("--output-dir", type=Path, default=DATA / "synthesis_images")
    parser.add_argument("--annotation", type=Path, default=DATA / "synthesis_annotation.txt")
    parser.add_argument("--report-dir", type=Path, default=DATA / "synthesis_report")
    return parser.parse_args(argv)


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
