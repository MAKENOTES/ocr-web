#!/usr/bin/env python3
"""按字符统计补齐低频字，默认合成 50000 张参考 train/val 风格的 PNG。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录，由当前文件位置推导，不依赖运行目录。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.annotation import format_annotation_line
from recognition.utils.synthesis.generate import (
    DATA, FontFace, default_font_paths, ensure_empty_outputs, load_fonts,
    read_chars, render_image, sha256,
)
from recognition.utils.synthesis.style import Style, sample_styles
from recognition.utils.synthesis.naming import REPORT_ROOT


@dataclass(frozen=True)
class Sample:
    text: str
    targets: str
    font_index: int
    style: Style


def read_statistics(path: Path) -> dict[str, int]:
    """严格按 CSV 第一列字符、最后一列出现次数读取，支持 BOM 和引号。"""
    counts = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(f"Expected a CSV header with at least two columns: {path}")
        for row in reader:
            location = f"{path}:{reader.line_num}"
            if len(row) != len(header):
                raise ValueError(f"Unexpected CSV column count: {location}")
            char = row[0]
            if len(char) != 1 or not char.isprintable() or char.isspace():
                raise ValueError(f"Expected one visible Unicode character: {location}: {char!r}")
            if char in counts:
                raise ValueError(f"Duplicate character: {location}: {char!r}")
            try:
                count = int(row[-1])
            except ValueError as exc:
                raise ValueError(f"Invalid character count: {location}: {row[-1]!r}") from exc
            if count < 0:
                raise ValueError(f"Negative character count: {location}")
            counts[char] = count
    if not counts:
        raise ValueError(f"No character rows found: {path}")
    return counts


def build_quotas(counts: dict[str, int], target_count: int, images: int) -> dict[str, int]:
    if not 100 <= target_count <= 120:
        raise ValueError("--target-count must be between 100 and 120")
    if images <= 0:
        raise ValueError("--images must be positive")
    quotas = {char: target_count - count for char, count in counts.items() if count < 100}
    if not quotas:
        raise ValueError("No target characters with counts below 100 in the CSV")
    if sum(quotas.values()) > images * 24:
        minimum = (sum(quotas.values()) + 23) // 24
        raise ValueError(f"Insufficient --images for at most 24 characters per image; need >= {minimum}")
    return quotas


def plan_samples(quotas: dict[str, int], companions: list[str], fonts: list[FontFace],
                 styles: list[Style], images: int, seed: int) -> list[Sample]:
    """先分配全部目标次数，再填充非目标配字；渲染前就验证配额和字体。"""
    if set(quotas) & set(companions):
        raise ValueError("Companion characters must exclude every target character")
    rng = random.Random(seed)
    tokens = [char for char, count in quotas.items() for _ in range(count)]
    rng.shuffle(tokens)
    base, extra = divmod(len(tokens), images)
    # 每张分到相近数量的目标字，避免后半批只剩常用配字；余数随机分配。
    sizes = [base + (index < extra) for index in range(images)]
    rng.shuffle(sizes)
    pools = [[char for char in companions if char in face.chars] for face in fonts]
    samples = []
    offset = 0
    for size in sizes:
        targets = tokens[offset:offset + size]
        offset += size
        style = rng.choice(styles)
        length = max(size, style.length)
        choices = [index for index, face in enumerate(fonts)
                   if set(targets) <= face.chars and (length == size or pools[index])]
        if not choices:
            raise ValueError(
                f"No single font can render targets {''.join(targets)!r} with companions; "
                "provide --fonts with full target coverage and non-target characters"
            )
        weights = [1 if "Bold" in fonts[index].path.stem else 3 for index in choices]
        font_index = rng.choices(choices, weights=weights, k=1)[0]
        chars = targets + rng.choices(pools[font_index], k=length - size)
        rng.shuffle(chars)
        samples.append(Sample("".join(chars), "".join(targets), font_index, style))
    actual = Counter(char for sample in samples for char in sample.text if char in quotas)
    if actual != Counter(quotas):
        raise RuntimeError("Planned target counts do not match quotas")
    return samples


def generate(args: argparse.Namespace) -> dict:
    if args.style_samples <= 0:
        raise ValueError("--style-samples must be positive")
    partial = args.annotation.with_name(args.annotation.name + ".partial")
    if not args.dry_run:
        ensure_empty_outputs(args.output_dir, args.annotation, args.report_dir)
        if partial.exists():
            raise FileExistsError(f"Partial annotation already exists: {partial}")
    counts = read_statistics(args.stats)
    quotas = build_quotas(counts, args.target_count, args.images)
    keys = read_chars(args.keys)
    missing = set(quotas) - set(keys)
    if missing:
        raise ValueError(f"Targets absent from {args.keys}: {''.join(sorted(missing))}")
    fonts = load_fonts(args.fonts or default_font_paths(), list(quotas))
    styles = sample_styles(args.references, args.style_samples, args.seed)
    # 只用 CSV 中已达到 100 次的字配文，避免无意增加任何低频目标字。
    companions = [char for char in keys if counts.get(char, 0) >= 100]
    samples = plan_samples(quotas, companions, fonts, styles, args.images, args.seed)
    report = {
        "images": len(samples), "target_characters": len(quotas),
        "target_count": args.target_count, "added_occurrences": sum(quotas.values()),
        "counts": {char: {"before": counts[char], "added": added,
                          "after": counts[char] + added} for char, added in quotas.items()},
        "seed": args.seed, "date": args.date,
        "stats": str(args.stats), "stats_sha256": sha256(args.stats),
        "keys": str(args.keys), "keys_sha256": sha256(args.keys),
        "output_dir": str(args.output_dir), "annotation": str(args.annotation),
        "annotation_format": "relative/path.png<TAB>text",
        "reference_directories": [str(path) for path in args.references],
        "reference_styles": len(styles),
        "lengthened_samples": sum(len(sample.text) > sample.style.length for sample in samples),
        "font_sha256": {str(face.path): sha256(face.path) for face in fonts},
        "dry_run": args.dry_run,
    }
    print(f"Planned {len(samples)} images; {len(quotas)} target characters, "
          f"{sum(quotas.values())} added occurrences; final count {args.target_count}", flush=True)
    if args.dry_run:
        return report
    try:
        report["renderer_version"] = importlib.metadata.version("text_renderer")
        from text_renderer.utils.draw_utils import draw_text_on_bg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install text_renderer using synthesis/requirements.txt with --no-deps") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.annotation.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    actual: Counter[str] = Counter()
    font_counts: Counter[str] = Counter()
    preview = Image.new("RGB", (1200, 1056), "#dedede")
    preview_draw = ImageDraw.Draw(preview)
    preview_count = 0
    preview_interval = max(1, len(samples) // 60)
    with partial.open("x", encoding="utf-8") as labels, \
            (args.report_dir / "samples.jsonl").open("x", encoding="utf-8") as metadata:
        for number, sample in enumerate(samples, 1):
            face = fonts[sample.font_index]
            image = render_image(sample.text, face, sample.style, rng)
            filename = f"synthesis_balanced_{args.date}_{number:05d}.png"
            path = args.output_dir / filename
            with path.open("xb") as handle:
                image.save(handle, format="PNG")
            relative = Path(os.path.relpath(path, args.annotation.parent)).as_posix()
            labels.write(format_annotation_line(relative, sample.text))
            actual.update(char for char in sample.text if char in quotas)
            font_counts[f"{face.path.name}#{face.index}"] += 1
            metadata.write(json.dumps({
                "file": relative, "text": sample.text, "targets": sample.targets,
                "font": str(face.path), "font_index": face.index,
                "style": sample.style.to_dict(),
            }, ensure_ascii=False) + "\n")
            if (number - 1) % preview_interval == 0 and preview_count < 60:
                thumbnail = image.copy()
                thumbnail.thumbnail((288, 48))
                x, y = (preview_count % 4) * 300, (preview_count // 4) * 70
                preview_draw.text((x + 3, y + 2), filename, fill="black")
                preview.paste(thumbnail, (x + 3, y + 19))
                preview_count += 1
            if number % 1000 == 0 or number == len(samples):
                print(f"Completed {number}/{len(samples)} images", flush=True)
    if actual != Counter(quotas):
        raise RuntimeError("Generated target character counts do not match quotas")
    report["fonts"] = dict(font_counts)
    preview.save(args.report_dir / "preview.png")
    with (args.report_dir / "character_counts.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["字符", "原次数", "新增次数", "最终次数"])
        writer.writerows((char, counts[char], actual[char], counts[char] + actual[char])
                         for char in quotas)
    with (args.report_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    # 全部验证成功后才发布标注；拒绝覆盖，中断时保留现场。
    os.link(partial, args.annotation)
    partial.unlink()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path,
                        default=ROOT / "outputs/recognition_character_stats/character_stats.csv")
    parser.add_argument("--keys", type=Path, default=ROOT / "recognition/keys.txt")
    parser.add_argument("--images", type=int, default=50000, help="新增图片总数，默认 50000")
    parser.add_argument("--target-count", type=int, default=110, help="原次数加新增次数，范围 100～120")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--references", type=Path, nargs="+",
                        default=[DATA / "train_images", DATA / "val_images"])
    parser.add_argument("--style-samples", type=int, default=1600)
    parser.add_argument("--fonts", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="校验统计、字体、风格和全部文字配额，不写文件")
    args = parser.parse_args(argv)
    # 启动时读取一次系统本地日期，跨午夜生成也保持同一批命名一致。
    args.date = datetime.now().strftime("%Y%m%d")
    prefix = f"synthesis_balanced_{args.date}"
    args.output_dir = args.output_dir or DATA / f"{prefix}_images"
    args.annotation = args.annotation or DATA / f"{prefix}_annotation.txt"
    args.report_dir = args.report_dir or REPORT_ROOT / prefix
    return args


def main() -> int:
    try:
        report = generate(parse_args())
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if report["dry_run"]:
        print("Dry run passed; no files written")
    else:
        print(f"Saved {report['images']} images; annotations: {report['annotation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
