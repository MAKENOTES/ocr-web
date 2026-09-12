#!/usr/bin/env python3
"""从 Recognition 训练标注中提取字符字典。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from recognition.utils.annotation import split_annotation_fields
except ModuleNotFoundError:  # 支持直接运行当前脚本
    from annotation import split_annotation_fields


RECOGNITION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RECOGNITION_DIR / "data" / "append1_annotation.txt"
DEFAULT_OUTPUT = RECOGNITION_DIR / "keys-new.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Recognition 训练标注中提取并排序所有唯一字符。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入标注文件（默认：recognition/data/train_annotation.txt）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出字典文件（默认：recognition/keys-new.txt）。",
    )
    return parser.parse_args()


def parse_annotation_text(
    raw_line: str,
    annotation_path: Path,
    line_number: int,
) -> str:
    """使用与 TextRecognitionDataset 相同的规则读取一行文字。"""
    _, text_field = split_annotation_fields(
        raw_line, annotation_path, line_number
    )

    text = text_field.strip()
    if not text:
        raise ValueError(f"识别文字为空: {annotation_path}:{line_number}")
    return text


def extract_chars(annotation_path: Path) -> set[str]:
    """读取标注并返回其中全部唯一字符。"""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    all_chars: set[str] = set()
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = parse_annotation_text(line, annotation_path, line_number)
            all_chars.update(text)

    if not all_chars:
        raise ValueError(f"标注文件没有可写入字典的字符: {annotation_path}")
    return all_chars


def extract_chars_to_dict(
    input_file: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT,
) -> int:
    """提取、排序并写入字典，返回唯一字符数。"""
    annotation_path = Path(input_file)
    output_path = Path(output_file)
    sorted_chars = sorted(extract_chars(annotation_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{char}\n" for char in sorted_chars),
        encoding="utf-8",
    )
    return len(sorted_chars)


def main() -> int:
    args = parse_args()
    try:
        char_count = extract_chars_to_dict(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"提取完成！共计 {char_count} 个唯一字符。")
    print(f"字典已保存至: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
