#!/usr/bin/env python3
"""移除 train+val 已充分覆盖字符的合成样本；默认预览，不修改 keys。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

try:
    from recognition.utils.annotation import split_annotation_fields
    from recognition.utils.character_stats import DEFAULT_OUTPUT_DIR, RECOGNITION_DIR
    from recognition.utils.remove_characters_by_stats import (
        RemovalPlan, apply_plan, load_statistics, read_lines,
    )
except ModuleNotFoundError:  # 支持直接运行，不依赖 torch 或合成渲染模块。
    from annotation import split_annotation_fields
    from character_stats import DEFAULT_OUTPUT_DIR, RECOGNITION_DIR
    from remove_characters_by_stats import RemovalPlan, apply_plan, load_statistics, read_lines


TARGETS = ("synthesis1", "synthesis2")  # 按常见字规则清理的两份合成数据。
DATASETS = ("train", "val", *TARGETS)  # 用于校验 CSV 和保留图片引用的四份清单。


def build_plan(csv_path: Path, data_dir: Path, threshold: int = 100,
               match: str = "all") -> RemovalPlan:
    """以 train+val 次数筛选常见字，整条移除；检查 CSV、路径及保留引用。"""
    if threshold <= 0:
        raise ValueError("--threshold must be positive")
    if match not in ("all", "any"):
        raise ValueError(f"Unknown --match rule: {match}")
    counts = load_statistics(csv_path)
    common = {char: (f"train+val>={threshold}",) for char, row in counts.items()
              if row["train"] + row["val"] >= threshold}
    common_set = set(common)
    data_dir = data_dir.resolve()
    originals = {csv_path: "".join(read_lines(csv_path))}
    replacements = {}
    removed_rows = []
    removed_images = set()
    retained_images = set()
    for name in DATASETS:
        path = data_dir / f"{name}_annotation.txt"
        if path.is_symlink():
            raise ValueError(f"Annotation must not be a symbolic link: {path}")
        lines = read_lines(path)
        originals[path] = "".join(lines)
        actual: Counter[str] = Counter()
        kept = []
        for number, line in enumerate(lines, 1):
            relative, label = split_annotation_fields(line, path, number)
            label = label.strip()  # 与 character_stats.py 和训练保持一致。
            actual.update(label)
            chars = set(label)
            matched = chars & common_set
            selected = bool(chars) and (chars <= common_set if match == "all" else bool(matched))
            lexical_path = path.parent / relative
            image_path = lexical_path.resolve()
            if name in TARGETS and selected:
                if (Path(relative).is_absolute() or Path(relative).suffix != ".png"
                        or not image_path.is_relative_to(data_dir) or lexical_path != image_path):
                    raise ValueError(f"Expected an ordinary relative PNG inside {data_dir}: {path}:{number}: {relative}")
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image to remove does not exist: {path}:{number}: {image_path}")
                removed_images.add(image_path)
                removed_rows.append({
                    "dataset": name, "annotation": str(path), "line": number,
                    "image": str(image_path), "label": label, "characters": sorted(matched),
                })
            else:
                kept.append(line)
                retained_images.add(image_path)
        for char, row in counts.items():
            if row[name] != actual[char]:
                raise ValueError(
                    f"CSV is stale; rerun character_stats.py: {csv_path}; "
                    f"{path}: {char!r}: CSV={row[name]}, actual={actual[char]}"
                )
        if len(kept) != len(lines):
            replacements[path] = "".join(kept)
    conflicts = removed_images & retained_images
    if conflicts:
        raise ValueError(f"Images to remove are still referenced by retained annotations: {sorted(map(str, conflicts))}")
    return RemovalPlan(common, counts, originals, replacements, removed_rows, sorted(removed_images))


def removal_impact(plan: RemovalPlan, threshold: int) -> dict:
    """统计整条删除会连带损失多少非充分覆盖字符，以及是否使合计跌破阈值。"""
    removed = Counter(char for row in plan.removed_rows for char in row["label"])
    uncommon = {char: count for char, count in removed.items() if char not in plan.characters}
    below_threshold = []
    for char, count in uncommon.items():
        if char not in plan.counts:
            continue
        before = sum(plan.counts[char][name] for name in DATASETS)
        if before >= threshold and before - count < threshold:
            below_threshold.append(char)
    return {
        "uncommon_occurrences": sum(uncommon.values()),
        "uncommon_characters": len(uncommon),
        "newly_below_threshold": len(below_threshold),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_DIR / "character_stats.csv")
    parser.add_argument("--data-dir", type=Path, default=RECOGNITION_DIR / "data")
    parser.add_argument("--threshold", type=int, default=100, help="train+val 合计次数阈值，包含等于，默认 100")
    parser.add_argument("--match", choices=("all", "any"), default="all",
                        help="all: 整条文字均为常见字才删除（默认）；any: 包含任一常见字即删除整条")
    parser.add_argument("--apply", action="store_true", help="备份后移除标注和对应图片；不传时只预览")
    parser.add_argument("--backup-dir", type=Path,
                        default=RECOGNITION_DIR.parent / "outputs" / "recognition_common_synthesis_removal")
    parser.add_argument("--max-details", type=int, default=20, help="预览明细条数，0 显示全部")
    args = parser.parse_args(argv)
    if args.max_details < 0:
        parser.error("--max-details must be >= 0")
    try:
        csv_path = args.csv.resolve()
        plan = build_plan(csv_path, args.data_dir, args.threshold, args.match)
        print(f"{'执行' if args.apply else '预览'}；常见字条件: train+val >= {args.threshold}；"
              f"命中 {len(plan.characters)} 个字符；匹配方式: {args.match}")
        for name in TARGETS:
            path = args.data_dir.resolve() / f"{name}_annotation.txt"
            total = len(plan.originals[path].splitlines())
            removed = sum(row["dataset"] == name for row in plan.removed_rows)
            print(f"{path}: 原有 {total} 条，移除 {removed} 条，保留 {total - removed} 条")
        print(f"共移除 {len(plan.removed_rows)} 条标注、{len(plan.images)} 张图片（按路径去重）。")
        impact = removal_impact(plan, args.threshold)
        print(f"连带移除 train+val 未达阈值或 CSV 未收录的字符: "
              f"{impact['uncommon_characters']} 个字符、{impact['uncommon_occurrences']} 次出现；"
              f"其中 {impact['newly_below_threshold']} 个字符的四份标注合计将从达标变为低于阈值。")
        for row in plan.removed_rows[:args.max_details or None]:
            print(f"  {row['annotation']}:{row['line']} -> {row['image']} {row['label']!r}")
        if args.apply:
            backup = apply_plan(plan, csv_path, args.backup_dir.resolve())
            print("移除完成；请重新运行 character_stats.py 更新统计。" if backup else "没有需要移除的内容。")
        else:
            print("仅预览，未修改文件；加 --apply 执行。train/val 和 keys 保持不变。")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
