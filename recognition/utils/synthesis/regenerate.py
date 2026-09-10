#!/usr/bin/env python3
"""按已校对的文本清单重新合成 PNG，保留条数和文字覆盖；默认仅校验。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.annotation import format_annotation_line, split_annotation_fields
from recognition.utils.synthesis.generate import (
    DATA, default_font_paths, ensure_empty_outputs, load_fonts, read_chars,
    render_image, sha256,
)
from recognition.utils.synthesis.style import sample_styles


def load_texts(path: Path, keys: set[str]) -> list[str]:
    """只读取正文；字符规范需预先校对，禁止边渲染边猜测或丢弃字符。"""
    texts = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _, text = split_annotation_fields(raw, path, number)
        if not text or not text.strip():
            raise ValueError(f"Empty source label: {path}:{number}")
        missing = set(text) - keys
        if missing:
            raise ValueError(f"Source characters absent from keys: {path}:{number}: {sorted(missing)!r}")
        texts.append(text)
    if not texts:
        raise ValueError(f"No source labels: {path}")
    return texts


def generate(args) -> dict:
    """复用 generate.py 的字体与渲染器，重新抽样风格，不读取旧合成图像素。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        raise ValueError(f"Invalid filename prefix: {args.prefix!r}")
    if args.style_samples <= 0:
        raise ValueError("--style-samples must be positive")
    ensure_empty_outputs(args.output_dir, args.annotation, args.report_dir)
    partial = args.annotation.with_name(args.annotation.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Partial annotation already exists: {partial}")

    keys = set(read_chars(args.keys))
    texts = load_texts(args.source_annotation, keys)
    fonts = load_fonts(args.fonts or default_font_paths(), sorted(set("".join(texts))))
    choices = []
    for number, text in enumerate(texts, 1):
        options = [i for i, font in enumerate(fonts) if set(text) <= font.chars]
        if not options:
            raise ValueError(f"No font covers source label: {args.source_annotation}:{number}: {text!r}")
        choices.append(options)
    styles = sample_styles(args.references, args.style_samples, args.seed)
    report = {
        "images": len(texts), "seed": args.seed, "date": args.date,
        "source_annotation": str(args.source_annotation.resolve()),
        "source_sha256": sha256(args.source_annotation), "keys_sha256": sha256(args.keys),
        "annotation": str(args.annotation.resolve()), "output_dir": str(args.output_dir.resolve()),
        "reference_directories": [str(p.resolve()) for p in args.references],
        "reference_styles": len(styles),
        "font_sha256": {str(f.path): sha256(f.path) for f in fonts},
        "annotation_format": "relative/path.png<TAB>text", "dry_run": not args.apply,
    }
    if not args.apply:
        return report

    for directory in (args.output_dir, args.annotation.parent, args.report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    preview = Image.new("RGB", (1500, 1200), "#dedede")
    draw = ImageDraw.Draw(preview)
    interval = max(1, len(texts) // 48)
    slot = 0
    with partial.open("x", encoding="utf-8") as labels, \
            (args.report_dir / "samples.jsonl").open("x", encoding="utf-8") as metadata:
        for index, (text, options) in enumerate(zip(texts, choices), 1):
            weights = [1 if "Bold" in fonts[i].path.stem else 3 for i in options]
            face = fonts[rng.choices(options, weights=weights, k=1)[0]]
            style = rng.choice(styles)
            image = render_image(text, face, style, rng)
            filename = f"{args.prefix}_{args.date}_{index:05d}.png"
            destination = args.output_dir / filename
            with destination.open("xb") as handle:
                image.save(handle, format="PNG")
            relative = Path(os.path.relpath(destination, args.annotation.parent)).as_posix()
            labels.write(format_annotation_line(relative, text))
            metadata.write(json.dumps({
                "file": relative, "text": text, "source_line": index,
                "font": str(face.path), "font_index": face.index, "style": style.to_dict(),
            }, ensure_ascii=False) + "\n")
            if (index - 1) % interval == 0 and slot < 48:
                thumb = image.copy()
                thumb.thumbnail((735, 30))
                x, y = (slot % 2) * 750 + 5, (slot // 2) * 50
                draw.text((x, y), f"{index:05d}", fill="black")
                preview.paste(thumb, (x, y + 17))
                slot += 1
            if index % 1000 == 0:
                print(f"{args.prefix}: {index}/{len(texts)}", flush=True)
    preview.save(args.report_dir / "preview.png")
    (args.report_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.link(partial, args.annotation)
    partial.unlink()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-annotation", type=Path, required=True, help="已校对的 Tab 文本清单")
    parser.add_argument("--keys", type=Path, default=ROOT / "recognition/keys.txt", help="本次字典")
    parser.add_argument("--prefix", required=True, help="图片文件名前缀，例如 synthesis1")
    parser.add_argument("--references", type=Path, nargs="+", default=[DATA / "train_images"], help="风格参考目录")
    parser.add_argument("--style-samples", type=int, default=1600, help="参考图抽样数量，张")
    parser.add_argument("--fonts", type=Path, nargs="+", help="字体文件，默认本机中文字体")
    parser.add_argument("--seed", type=int, default=20260910, help="随机种子")
    parser.add_argument("--output-dir", type=Path, required=True, help="新图片目录，必须为空")
    parser.add_argument("--annotation", type=Path, required=True, help="新标注路径，不能已存在")
    parser.add_argument("--report-dir", type=Path, required=True, help="新报告目录，必须为空")
    parser.add_argument("--apply", action="store_true", help="真正生成；默认仅校验输入和字体")
    args = parser.parse_args(argv)
    args.date = datetime.now().strftime("%Y%m%d")
    return args


def main():
    try:
        report = generate(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{'Validated' if report['dry_run'] else 'Saved'} {report['images']} images; {report['annotation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
