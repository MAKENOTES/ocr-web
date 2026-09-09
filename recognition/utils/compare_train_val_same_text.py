#!/usr/bin/env python3
"""找出 Recognition 训练集与验证集中相同的识别文字。

默认读取 ``recognition/data/train_annotation.txt`` 和
``recognition/data/val_annotation.txt``，并将相同文字及两组对应图片路径写入
``recognition/data/train_val_same_text.txt``。脚本不检查或修改图片。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from recognition.utils.annotation import split_annotation_fields
except ModuleNotFoundError:  # 支持直接运行当前脚本
    from annotation import split_annotation_fields


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "train_val_same_text.txt"


@dataclass(frozen=True)
class AnnotationRecord:
    """一条图片文字标注。"""

    image_path: str
    text: str
    line_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="列出 train 和 val 中识别文字完全相同的图片。"
    )
    parser.add_argument(
        "--train-annotation",
        type=Path,
        default=DATA_DIR / "train_annotation.txt",
        help="训练集标注文件。",
    )
    parser.add_argument(
        "--val-annotation",
        type=Path,
        default=DATA_DIR / "val_annotation.txt",
        help="验证集标注文件。",
    )
    return parser.parse_args()


def normalize_text(text_field: str) -> str:
    """移除字段外空白，保留正文内容。"""
    return text_field.strip()


def load_annotations(annotation_path: Path) -> list[AnnotationRecord]:
    """读取标注文件，遇到格式错误时报告具体文件与行号。"""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    records: list[AnnotationRecord] = []
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            image_path, text_field = split_annotation_fields(
                line, annotation_path, line_number
            )

            text = normalize_text(text_field)
            if not text:
                raise ValueError(f"识别文字为空: {annotation_path}:{line_number}")

            records.append(
                AnnotationRecord(
                    image_path=image_path,
                    text=text,
                    line_number=line_number,
                )
            )

    if not records:
        raise ValueError(f"标注文件为空: {annotation_path}")
    return records


def group_by_text(
    records: list[AnnotationRecord],
) -> dict[str, list[AnnotationRecord]]:
    grouped: defaultdict[str, list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.text].append(record)
    return dict(grouped)


def find_common_texts(
    train_records: list[AnnotationRecord],
    val_records: list[AnnotationRecord],
) -> list[str]:
    train_by_text = group_by_text(train_records)
    val_by_text = group_by_text(val_records)
    return sorted(set(train_by_text) & set(val_by_text))


def format_common_texts(
    common_texts: list[str],
    train_records: list[AnnotationRecord],
    val_records: list[AnnotationRecord],
) -> str:
    """按相同文字分组，列出两组全部图片路径，保留标注顺序与相对路径。"""
    train_by_text = group_by_text(train_records)
    val_by_text = group_by_text(val_records)

    lines: list[str] = []
    for index, text in enumerate(common_texts, start=1):
        lines.append(f"[{index}] 文字: {text!r}\n")
        for name, grouped in (("train", train_by_text), ("val", val_by_text)):
            lines.append(f"  {name}:\n")
            lines.extend(f"    {record.image_path}\n" for record in grouped[text])
        lines.append("\n")
    return "".join(lines)


def main() -> int:
    args = parse_args()
    try:
        train_records = load_annotations(args.train_annotation)
        val_records = load_annotations(args.val_annotation)
        common_texts = find_common_texts(train_records, val_records)
        OUTPUT_PATH.write_text(
            format_common_texts(common_texts, train_records, val_records),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"找到 {len(common_texts)} 种相同文字。")
    print(f"结果已写入: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
