#!/usr/bin/env python3
"""找出 Recognition 标注中存在、但字符字典未收录的字符。"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

try:
    from recognition.utils.annotation import split_annotation_fields
except ModuleNotFoundError:  # 支持直接运行当前脚本
    from annotation import split_annotation_fields


def extract_chars_from_annotation(input_file: str | Path) -> set[str]:
    """按 ``图片路径<TAB>文字`` 格式提取标注中的全部唯一字符。"""
    annotation_path = Path(input_file)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    all_chars: set[str] = set()
    record_count = 0
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            _, text_field = split_annotation_fields(
                line, annotation_path, line_number
            )
            text = text_field.strip()
            if not text:
                raise ValueError(
                    f"识别文字为空: {annotation_path}:{line_number}"
                )
            all_chars.update(text)
            record_count += 1

    if record_count == 0:
        raise ValueError(f"标注文件为空: {annotation_path}")
    return all_chars


def load_keys(keys_file: str | Path) -> set[str]:
    """读取每个非空行恰好包含一个字符的字典文件。"""
    keys_path = Path(keys_file)
    if not keys_path.is_file():
        raise FileNotFoundError(f"字符字典不存在: {keys_path}")

    keys: set[str] = set()
    with keys_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            char = line.rstrip("\r\n")
            if not char:
                continue
            if len(char) != 1:
                raise ValueError(
                    f"字符字典每个非空行必须恰好包含一个字符: "
                    f"{keys_path}:{line_number}: {char!r}"
                )
            keys.add(char)

    if not keys:
        raise ValueError(f"字符字典没有有效字符: {keys_path}")
    return keys


def compare_keys(
    annotation_files: Iterable[str | Path],
    keys_file: str | Path = "recognition/keys.txt",
    output_file: str | Path = "keys-compare.txt",
) -> list[str]:
    """把所有标注中未被 ``keys_file`` 收录的字符写入输出文件。"""
    annotation_paths = [Path(path) for path in annotation_files]
    if not annotation_paths:
        raise ValueError("annotation_files 不能为空")

    annotation_chars: set[str] = set()
    for annotation_path in annotation_paths:
        chars = extract_chars_from_annotation(annotation_path)
        print(f"从 {annotation_path} 中提取 {len(chars)} 个唯一字符。")
        annotation_chars.update(chars)

    keys_path = Path(keys_file)
    keys = load_keys(keys_path)
    missing_in_keys = sorted(annotation_chars - keys)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{char}\n" for char in missing_in_keys),
        encoding="utf-8",
    )

    print(f"keys 文件 {keys_path} 中共 {len(keys)} 个字符。")
    print(f"标注中共 {len(annotation_chars)} 个唯一字符。")
    print(f"标注中存在、但 keys 未收录的字符共 {len(missing_in_keys)} 个。")
    print(f"结果已写入: {output_path}")
    return missing_in_keys


def main() -> int:
    try:
        compare_keys(
            annotation_files=[
                "recognition/data/train_annotation.txt",
                "recognition/data/val_annotation.txt",
                "recognition/data/append_annotation.txt",
                "recognition/data/bench1_annotation.txt",
                "recognition/data/bench2_annotation.txt",
            ],
            keys_file="recognition/keys.txt",
            output_file="recognition/keys-compare.txt",
        )
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
