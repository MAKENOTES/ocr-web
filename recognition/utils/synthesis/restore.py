#!/usr/bin/env python3
"""按标注原路径补回已删除的合成 PNG，复用逐图记录的字体、风格及专项效果。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录，由当前文件位置推导，不依赖运行目录。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.annotation import split_annotation_fields
from recognition.utils.synthesis.generate import load_fonts, read_chars, render_image, sha256
from recognition.utils.synthesis.generate_degraded import degrade
from recognition.utils.synthesis.generate_edge import apply_edge
from recognition.utils.synthesis.generate_lines import add_lines
from recognition.utils.synthesis.generate_red_stamp import StampStyle, overlay_stamp
from recognition.utils.synthesis.style import Style


def read_plan(annotation: Path, records_path: Path, keys: set[str]) -> list[tuple]:
    """严格匹配记录和标注；所有路径先校验，已有图片和越界路径一律拒绝。"""
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    lines = annotation.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) != len(records):
        raise ValueError(f"Annotation/record count mismatch: {annotation}, {records_path}")
    plan, seen = [], set()
    for number, (line, record) in enumerate(zip(lines, records), 1):
        relative, text = split_annotation_fields(line, annotation, number)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".png":
            raise ValueError(f"Unsafe PNG path: {annotation}:{number}: {relative}")
        destination = annotation.parent / path
        if not destination.resolve().is_relative_to(annotation.parent.resolve()):
            raise ValueError(f"Image path escapes annotation directory: {destination}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Image already exists: {destination}")
        if destination.resolve() in seen:
            raise ValueError(f"Duplicate image path: {annotation}:{number}: {destination}")
        seen.add(destination.resolve())
        if text != record["text"] or path.name != Path(record["file"]).name:
            raise ValueError(f"Annotation/record mismatch: {annotation}:{number}: {relative}")
        if not text or text != text.strip() or set(text) - keys:
            raise ValueError(f"Empty, untrimmed or unknown label: {annotation}:{number}: {text!r}")
        plan.append((destination, text, record))
    return plan


def body_style(data: dict) -> Style:
    return Style(**dict(data, background=tuple(data["background"]), foreground=tuple(data["foreground"])))


def render_record(text, record, fonts, rng):
    """按记录字段显式调用各专项脚本；重新采样细节，正文始终来自标注。"""
    face = fonts[(record["font"], record["font_index"])]
    style_data = record["style"]
    style = body_style(style_data.get("body", style_data))
    image = render_image(text, face, style, rng)
    if "stamp" in record:
        stamp_style = StampStyle(**dict(style_data, body=style, color=tuple(style_data["color"])))
        stamp_face = fonts[(record["stamp_font"], record["stamp_font_index"])]
        image, effect = overlay_stamp(image, stamp_style, stamp_face, record["stamp"]["stamp_text"],
                                      rng, scene=record["stamp"]["scene"])
        return image, {"stamp": effect}
    if "lines" in record:
        image, effect = add_lines(image, style, record["lines"]["pattern"], rng)
        return image, {"lines": effect}
    effect = record.get("effect", {})
    pattern = effect.get("pattern", "plain")
    if pattern in ("low_resolution", "joined_strokes", "broken_strokes", "scan_band"):
        image, effect = degrade(image, style, pattern, rng)
    elif pattern in ("tight_right", "clip_right", "tight_left", "clip_left", "clip_vertical", "fragment"):
        fragment = None
        if pattern == "fragment":
            fragment = render_image(effect["fragment_character"], face, style, rng)
        image, effect = apply_edge(image, text, style, pattern, rng, fragment)
    elif pattern not in ("plain", "confusable"):
        raise ValueError(f"Unknown recorded synthesis effect: {pattern}")
    return image, {"effect": {**record.get("effect", {}), **effect} or {"pattern": "plain"}}


def generate(args):
    if args.report_dir.exists():
        raise FileExistsError(f"Report directory already exists: {args.report_dir}")
    source_hash, keys_hash = sha256(args.annotation), sha256(args.keys)
    plan = read_plan(args.annotation, args.records, set(read_chars(args.keys)))
    paths, characters = set(), set()
    for _, text, record in plan:
        paths.add(Path(record["font"])); characters.update(text)
        if "stamp" in record:
            paths.add(Path(record["stamp_font"])); characters.update(record["stamp"]["stamp_text"])
        characters.update(record.get("effect", {}).get("fragment_character", ""))
    faces = load_fonts(list(paths), sorted(characters))
    fonts = {(str(f.path), f.index): f for f in faces}
    for _, text, record in plan:
        face = fonts[(record["font"], record["font_index"])]
        required = set(text + record.get("effect", {}).get("fragment_character", ""))
        if not required <= face.chars:
            raise ValueError(f"Recorded font cannot render label: {face.path}: {text!r}")
        if "stamp" in record:
            stamp_face = fonts[(record["stamp_font"], record["stamp_font_index"])]
            if not set(record["stamp"]["stamp_text"]) <= stamp_face.chars:
                raise ValueError(f"Recorded font cannot render stamp: {stamp_face.path}")
    summary = {"images": len(plan), "annotation": str(args.annotation.resolve()),
               "annotation_sha256": source_hash, "keys_sha256": keys_hash,
               "source_records": str(args.records.resolve()), "records_sha256": sha256(args.records),
               "seed": args.seed, "dry_run": not args.apply,
               "font_sha256": {str(p): sha256(p) for p in paths}}
    if not args.apply:
        return summary

    args.report_dir.mkdir(parents=True)
    preview = Image.new("RGB", (1500, 1200), "#dedede")
    draw = ImageDraw.Draw(preview)
    patterns = Counter()
    with (args.report_dir / "samples.jsonl").open("x", encoding="utf-8") as output:
        for number, (destination, text, record) in enumerate(plan, 1):
            # 每条记录独立种子，便于按行复现，不受之前图片随机分支次数影响。
            image, effect = render_record(text, record, fonts, random.Random(args.seed + number))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                image.save(handle, format="PNG")
            current = dict(record, file=destination.relative_to(args.annotation.parent).as_posix(),
                           seed=args.seed + number, sha256=sha256(destination), **effect)
            output.write(json.dumps(current, ensure_ascii=False) + "\n")
            key = current.get("stamp", {}).get("scene") or current.get("lines", {}).get("pattern") or current.get("effect", {}).get("pattern", "plain")
            patterns[key] += 1
            interval = max(1, len(plan) // 48)
            if (number - 1) % interval == 0 and (number - 1) // interval < 48:
                slot = (number - 1) // interval
                thumb = image.copy(); thumb.thumbnail((735, 30))
                x, y = slot % 2 * 750 + 5, slot // 2 * 50
                draw.text((x, y), f"{number:05d} {key}", fill="black")
                preview.paste(thumb, (x, y + 17))
            if number % 1000 == 0:
                print(f"{args.annotation.stem}: {number}/{len(plan)}", flush=True)
    if sha256(args.annotation) != source_hash or sha256(args.keys) != keys_hash:
        raise RuntimeError("Annotation or keys changed during generation; verify generated images")
    summary["patterns"] = dict(patterns)
    preview.save(args.report_dir / "preview.png")
    (args.report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--annotation", type=Path, required=True, help="现有 Tab 标注，原样保留")
    parser.add_argument("--records", type=Path, required=True, help="同批次 samples.jsonl 生成记录")
    parser.add_argument("--keys", type=Path, default=ROOT / "recognition/keys.txt", help="正文字符字典")
    parser.add_argument("--report-dir", type=Path, required=True, help="本次报告目录，必须不存在")
    parser.add_argument("--seed", type=int, default=20260911, help="每行独立随机种子的基数")
    parser.add_argument("--apply", action="store_true", help="实际补回图片，默认仅校验")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = generate(parse_args())
    print(f"{'Validated' if result['dry_run'] else 'Restored'} {result['images']} images")
