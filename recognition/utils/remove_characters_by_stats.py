#!/usr/bin/env python3
"""按字符统计清理 append/bench1/bench2 图片、整条标注及 keys；默认仅预览。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from recognition.utils.annotation import split_annotation_fields
    from recognition.utils.character_stats import DEFAULT_OUTPUT_DIR, RECOGNITION_DIR, load_keys
    from recognition.utils.convert_annotation_delimiter import atomic_write
except ModuleNotFoundError:  # 支持直接运行脚本，不依赖 torch
    from annotation import split_annotation_fields
    from character_stats import DEFAULT_OUTPUT_DIR, RECOGNITION_DIR, load_keys
    from convert_annotation_delimiter import atomic_write


TARGETS = ("synthesis")
DATASETS = ("train", "val", *TARGETS)


@dataclass
class RemovalPlan:
    characters: dict[str, tuple[str, ...]]
    counts: dict[str, dict[str, int]]
    originals: dict[Path, str]
    replacements: dict[Path, str]
    removed_rows: list[dict]
    images: list[Path]


def load_statistics(path: Path) -> dict[str, dict[str, int]]:
    """只使用中文字符列和五份数据的次数列；拒绝损坏或不完整的 CSV。"""
    result = {}
    required = {"字符", *(f"{name}_次数" for name in DATASETS)}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV 缺少必要列: {path}: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            char = row["字符"]
            if char is None or len(char) != 1 or char in result:
                raise ValueError(f"CSV 字符无效或重复: {path}:{reader.line_num}: {char!r}")
            try:
                counts = {name: int(row[f"{name}_次数"]) for name in DATASETS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV 次数必须是非负整数: {path}:{reader.line_num}") from exc
            if any(count < 0 for count in counts.values()):
                raise ValueError(f"CSV 次数必须是非负整数: {path}:{reader.line_num}")
            result[char] = counts
    if not result:
        raise ValueError(f"CSV 为空: {path}")
    return result


def select_characters(counts: dict[str, dict[str, int]], mode: str) -> dict[str, tuple[str, ...]]:
    selected = {}
    for char, row in counts.items():
        if row["train"] or row["val"]:
            continue
        reasons = []
        if mode in ("1", "both") and sum(row[name] for name in TARGETS) < 10:
            reasons.append("1")
        if mode in ("2", "both") and sum(row[name] > 0 for name in TARGETS) == 1:
            reasons.append("2")
        if reasons:
            selected[char] = tuple(reasons)
    return selected


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.readlines()


def build_plan(
    csv_path: Path, keys_path: Path, data_dir: Path, mode: str = "both",
    chars: str | None = None,
) -> RemovalPlan:
    if mode not in ("1", "2", "both"):
        raise ValueError(f"未知移除方式: {mode}")
    counts = load_statistics(csv_path)
    keys = load_keys(keys_path)
    if set(keys) != set(counts):
        raise ValueError(f"CSV 字符集合与字典不一致，请重新运行 character_stats.py: {csv_path}; {keys_path}")
    selected = select_characters(counts, mode)
    if chars is not None:
        requested = set(chars)
        if not requested or requested - set(selected):
            raise ValueError(f"指定字符为空、不在字典中或不满足移除条件: {sorted(requested - set(selected))!r}")
        selected = {char: reasons for char, reasons in selected.items() if char in requested}
    selected_set = set(selected)
    originals = {}
    replacements = {}
    removed_rows = []
    retained_images = set()
    removed_images = set()
    data_dir = data_dir.resolve()

    for name in DATASETS:
        path = data_dir / f"{name}_annotation.txt"
        if path.is_symlink():
            raise ValueError(f"标注文件不能是符号链接: {path}")
        lines = read_lines(path)
        originals[path] = "".join(lines)
        actual: Counter[str] = Counter()
        kept = []
        for line_number, line in enumerate(lines, 1):
            relative, label = split_annotation_fields(line, path, line_number)
            label = label.strip()  # 与 character_stats.py 和训练的文字统计一致
            actual.update(label)
            image_path = (path.parent / relative).resolve()
            matched = selected_set.intersection(label)
            if name in TARGETS and matched:
                # 不猜扩展名，不允许越界路径或通过符号链接删除其他目录的图片。
                lexical_path = path.parent / relative
                if (Path(relative).is_absolute() or Path(relative).suffix != ".png"
                        or not image_path.is_relative_to(data_dir)
                        or lexical_path != image_path):
                    raise ValueError(f"待删图片必须是数据目录内的普通相对 PNG 路径: {path}:{line_number}: {relative}")
                if not image_path.is_file():
                    raise FileNotFoundError(f"待删图片不存在: {path}:{line_number}: {image_path}")
                removed_images.add(image_path)
                removed_rows.append({
                    "dataset": name, "annotation": str(path), "line": line_number,
                    "image": str(image_path), "label": label, "characters": sorted(matched),
                })
            else:
                kept.append(line)
                retained_images.add(image_path)
        for char, row in counts.items():
            if row[name] != actual[char]:
                raise ValueError(
                    f"CSV 已过期，请重新运行 character_stats.py: {csv_path}; "
                    f"{path}: {char!r}: CSV={row[name]}, 当前={actual[char]}"
                )
        if len(kept) != len(lines):
            replacements[path] = "".join(kept)

    conflicts = removed_images & retained_images
    if conflicts:
        raise ValueError(f"待删图片仍被 train/val 或保留标注引用: {sorted(map(str, conflicts))}")
    if keys_path.is_symlink():
        raise ValueError(f"字典不能是符号链接: {keys_path}")
    key_lines = read_lines(keys_path)
    originals[keys_path] = "".join(key_lines)
    if selected:
        replacements[keys_path] = "".join(
            line for line in key_lines if line.rstrip("\r\n") not in selected_set
        )
    return RemovalPlan(selected, counts, originals, replacements, removed_rows, sorted(removed_images))


def apply_plan(plan: RemovalPlan, csv_path: Path, backup_root: Path) -> Path | None:
    """先备份所有将被修改/删除的文件，再替换文字文件、删除图片。"""
    if not plan.replacements:
        return None
    # 防止预览计划构建后，另一个进程修改标注或字典。
    for path, original in plan.originals.items():
        if "".join(read_lines(path)) != original:
            raise ValueError(f"规划后文件发生变化，请重新运行: {path}")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="removal-", dir=backup_root))
    print(f"备份目录: {backup}", flush=True)
    mapping = []
    sources = [*plan.replacements, *plan.images]
    # 备份失败时尚未修改任何源文件；成功后 manifest 可用于逐项恢复。
    for index, source in enumerate(sources):
        target = backup / "files" / f"{index:08d}{source.suffix}"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        mapping.append({"original": str(source.resolve()), "backup": str(target.resolve())})
    shutil.copy2(csv_path, backup / "character_stats.csv")
    (backup / "manifest.json").write_text(json.dumps({
        "characters": plan.characters, "removed_rows": plan.removed_rows, "files": mapping,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        for path, content in plan.replacements.items():
            atomic_write(path, content)
        for path in plan.images:
            path.unlink()
    except OSError as exc:
        raise OSError(f"清理未全部完成，可按 manifest.json 从备份恢复: {backup}: {exc}") from exc
    return backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_DIR / "character_stats.csv")
    parser.add_argument("--keys", type=Path, default=RECOGNITION_DIR / "keys.txt")
    parser.add_argument("--data-dir", type=Path, default=RECOGNITION_DIR / "data")
    parser.add_argument("--mode", choices=("1", "2", "both"), default="both",
                        help="1: 三者合计 <10（含零次）；2: 仅在三者之一出现；both: 两者并集（默认）")
    parser.add_argument("--apply", action="store_true", help="备份后真正移除；不传时只预览")
    parser.add_argument("--chars", help="只处理指定字符，例如 癫；仍须满足所选移除条件")
    parser.add_argument("--backup-dir", type=Path,
                        default=RECOGNITION_DIR.parent / "outputs" / "recognition_character_removal")
    parser.add_argument("--max-details", type=int, default=20, help="显示的图片明细数量；0 为全部")
    args = parser.parse_args(argv)
    if args.max_details < 0:
        parser.error("--max-details 必须 >= 0")
    try:
        plan = build_plan(args.csv.resolve(), args.keys.absolute(), args.data_dir, args.mode, args.chars)
        print(f"{'执行' if args.apply else '预览'}；移除方式: {args.mode}；均要求 train/val 次数为 0")
        print("字符\t方式\tappend\tbench1\tbench2\t三者合计")
        for char, reasons in plan.characters.items():
            values = [plan.counts[char][name] for name in TARGETS]
            print(f"{char!r}\t{'+'.join(reasons)}\t" + "\t".join(map(str, [*values, sum(values)])))
        for name in TARGETS:
            removed = sum(row["dataset"] == name for row in plan.removed_rows)
            print(f"{(args.data_dir / (name + '_annotation.txt')).resolve()}: 移除 {removed} 条标注")
        print(f"{args.keys.resolve()}: 移除 {len(plan.characters)} 个字符")
        print(f"共移除 {len(plan.removed_rows)} 条标注、{len(plan.images)} 张图片（按路径去重）。")
        for row in plan.removed_rows[:args.max_details or None]:
            print(f"  {row['annotation']}:{row['line']} -> {row['image']} {row['label']!r}")
        if args.max_details and len(plan.removed_rows) > args.max_details:
            print("其余图片明细省略，可用 --max-details 0 查看全部。")
        if args.apply:
            backup = apply_plan(plan, args.csv.resolve(), args.backup_dir.resolve())
            print("移除完成；请重新生成字符统计 CSV。" if backup else "没有需要移除的内容。")
        else:
            print("仅预览，未写入或删除任何文件；加 --apply 才会真正移除。")
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
