#!/usr/bin/env python3
"""逐字统计 keys 在各份标注及合计中的出现次数、占全部文字字符数的百分比。"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

try:
    from recognition.utils.annotation import split_annotation_fields
except ModuleNotFoundError:  # 支持直接运行当前脚本，不依赖 torch
    from annotation import split_annotation_fields


RECOGNITION_DIR = Path(__file__).resolve().parents[1]
SYNTHESIS_DATASET = "synthesis"
DEFAULT_ANNOTATIONS = [
    RECOGNITION_DIR / "data" / f"{name}_annotation.txt"
    for name in ("train", "val", SYNTHESIS_DATASET)
]
DEFAULT_OUTPUT_DIR = RECOGNITION_DIR.parent / "outputs" / "recognition_character_stats"


def load_keys(keys_path: Path) -> list[str]:
    """保留 keys 的原顺序及空格；检查会破坏类别映射的重复/多字符行。"""
    keys = []
    seen = set()
    with keys_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            char = line.rstrip("\r\n")
            if not char:
                continue
            if len(char) != 1 or char in seen:
                raise ValueError(
                    f"字典行必须为不重复的单个 Unicode 字符: "
                    f"{keys_path}:{line_number}: {char!r}"
                )
            keys.append(char)
            seen.add(char)
    if not keys:
        raise ValueError(f"字典为空: {keys_path}")
    return keys


def count_annotation(annotation_path: Path) -> dict:
    """仅读取标注文字，不加载/修改图片；与训练一致 strip 并跳过空标签。"""
    counts: Counter[str] = Counter()
    sample_count = 0
    empty_labels = 0
    with annotation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _, label = split_annotation_fields(line, annotation_path, line_number)
            label = label.strip()
            if not label:
                empty_labels += 1
                continue
            sample_count += 1
            counts.update(label)
    return {
        "annotation": str(annotation_path.resolve()),
        "sample_count": sample_count,
        "skipped_empty_labels": empty_labels,
        "counts": counts,
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_statistics(keys_path: Path, annotation_paths: list[Path]) -> tuple[list[dict], dict]:
    """返回宽表与汇总；all 按标注记录累加，跨文件重复样本不去重。"""
    keys = load_keys(keys_path)
    key_set = set(keys)
    datasets = {}
    resolved_paths = set()
    for path in annotation_paths:
        name = path.stem.removesuffix("_annotation")
        if name == "all" or name in datasets:
            raise ValueError(f"数据集名称重复或使用保留名称 all: {path}")
        if path.resolve() in resolved_paths:
            raise ValueError(f"重复输入标注文件: {path}")
        resolved_paths.add(path.resolve())
        datasets[name] = count_annotation(path)
    if not datasets:
        raise ValueError("至少需要一份标注文件")

    aggregate = {
        "annotation": None,
        "sample_count": sum(item["sample_count"] for item in datasets.values()),
        "skipped_empty_labels": sum(item["skipped_empty_labels"] for item in datasets.values()),
        "counts": Counter(),
    }
    for item in datasets.values():
        aggregate["counts"].update(item["counts"])
    datasets["all"] = aggregate

    summaries = {}
    for name, item in datasets.items():
        counts = item["counts"]
        total = sum(counts.values())
        known = sum(counts[char] for char in keys)
        covered = sum(counts[char] > 0 for char in keys)
        summaries[name] = {
            "annotation": item["annotation"],
            "sample_count": item["sample_count"],
            "skipped_empty_labels": item["skipped_empty_labels"],
            "character_count": total,
            "unique_character_count": len(counts),
            "covered_key_count": covered,
            "key_coverage_ratio": ratio(covered, len(keys)),
            "missing_key_count": len(keys) - covered,
            "key_character_count": known,
            "unknown_character_count": total - known,
            "unknown_character_ratio": ratio(total - known, total),
            "unknown_unique_count": len(set(counts) - key_set),
            "key_frequency_buckets": {
                "0": sum(counts[char] == 0 for char in keys),
                "1-10": sum(1 <= counts[char] <= 10 for char in keys),
                "11-50": sum(11 <= counts[char] <= 50 for char in keys),
                "51-100": sum(51 <= counts[char] <= 100 for char in keys),
                "101+": sum(counts[char] > 100 for char in keys),
            },
        }

    rows = []
    for char in keys:
        row = {"character": char}
        for name, item in datasets.items():
            count = item["counts"][char]
            row.update({
                f"{name}_count": count,
                f"{name}_ratio": ratio(count, summaries[name]["character_count"]),
            })
        rows.append(row)

    # 合计分母相同，按次数排序等价于按未舍入的合计占比排序；同频保持 keys 顺序。
    rows.sort(key=lambda row: row["all_count"], reverse=True)

    summary = {
        "keys_path": str(keys_path.resolve()),
        "key_count": len(keys),
        "definitions": {
            "unit": "Unicode code point；不做全半角/标点/Unicode 归一化",
            "labels": "复用训练的 Tab 解析规则，strip 首尾空白；跳过空标签并计数",
            "all": "所有输入标注的记录相加，不去重；默认输入包含 val，不等于训练集总量",
            "count": "该字符的出现次数，同一样本内重复出现多次也累加",
            "ratio": "该字符次数 / 该数据集全部文字字符数（含字典外字符）；CSV 和终端显示百分数，保留五位小数",
            "all_ratio": "所有输入中的该字符总次数 / 所有输入的文字总字符数，不是各文件占比的平均值",
            "rows": "只包含 keys 中的字符，按合计占比倒序，同频保持 keys 顺序，零出现也保留",
            "sample_count": "数据集中的非空标签记录数，仅用于汇总，不是字符占比的分母",
            "validation": "只校验标注格式和字典，不检查图片存在性或跨数据集重复",
        },
        "datasets": summaries,
    }
    return rows, summary


def format_character_row(row: dict, dataset_names: list[str]) -> dict:
    """CSV 和终端使用相同的简洁表头与百分比格式。"""
    formatted = {"字符": row["character"]}
    for name in dataset_names:
        title = "合计" if name == "all" else name
        formatted[f"{title}_占比"] = f"{row[f'{name}_ratio']:.5%}"
    for name in dataset_names:
        title = "合计" if name == "all" else name
        formatted[f"{title}_次数"] = row[f"{name}_count"]
    return formatted


def write_statistics(rows: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = list(summary["datasets"])
    # utf-8-sig 便于表格软件直接识别中文，百分号明确占比的单位。
    with (output_dir / "character_stats.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(format_character_row(rows[0], dataset_names))
        )
        writer.writeheader()
        writer.writerows(format_character_row(row, dataset_names) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, default=RECOGNITION_DIR / "keys.txt")
    parser.add_argument("--annotations", type=Path, nargs="+", default=DEFAULT_ANNOTATIONS,
                        help=f"标注文件路径；默认统计 train、val、{SYNTHESIS_DATASET}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--chars", help="终端仅显示指定字符，例如 王 或 王李；默认显示全部，CSV 始终保存全部 keys 字符"
    )
    args = parser.parse_args(argv)
    try:
        rows, summary = build_statistics(args.keys, args.annotations)
        if args.chars is not None:
            missing = set(args.chars) - {row["character"] for row in rows}
            if missing:
                raise ValueError(f"查询字符不在字典中: {sorted(missing)!r}")
        write_statistics(rows, summary, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"字典字符数: {summary['key_count']}；按合计占比倒序，每个字符一行，包含零出现的字符。")
    print("占比 = 该字符出现次数 / 对应数据集全部文字字符数 × 100%")
    print("数据集\t文字总字符数（占比分母）")
    for name, item in summary["datasets"].items():
        print(f"{'合计' if name == 'all' else name}\t{item['character_count']}")
    dataset_names = list(summary["datasets"])
    print("\n" + "\t".join(format_character_row(rows[0], dataset_names)))
    for row in rows:
        if args.chars is None or row["character"] in args.chars:
            print("\t".join(str(value) for value in format_character_row(row, dataset_names).values()))
    print(f"\n已向 CSV 写入 {len(rows)} 个字符的逐字明细（含零出现字符）。")
    print(f"逐字 CSV: {args.output_dir / 'character_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
