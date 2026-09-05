"""移除 Recognition 标注文字字段最外层的一对双引号。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from recognition.utils.annotation import (
        format_annotation_line,
        split_annotation_fields,
    )
    from recognition.utils.convert_annotation_delimiter import (
        DEFAULT_ANNOTATIONS,
        atomic_write,
    )
except ModuleNotFoundError:  # 支持直接运行当前脚本
    from annotation import format_annotation_line, split_annotation_fields
    from convert_annotation_delimiter import DEFAULT_ANNOTATIONS, atomic_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Recognition 标注统一为 图片路径<TAB>无外围引号文字 格式。"
    )
    parser.add_argument(
        "annotations",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ANNOTATIONS),
        help="待处理文件；不传时处理 recognition/data 下的五个默认标注文件。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验并报告，不写入文件。",
    )
    return parser.parse_args()


def remove_outer_quotes_content(
    content: str,
    annotation_path: Path,
) -> tuple[str, int]:
    """返回去除文字外围双引号后的内容及修改行数。"""
    output_lines: list[str] = []
    changed_count = 0
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        image_path, text_field = split_annotation_fields(
            line, annotation_path, line_number
        )
        text = text_field.strip()
        if not text:
            raise ValueError(f"识别文字为空: {annotation_path}:{line_number}")
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            changed_count += 1
        if not text:
            raise ValueError(
                f"移除外围双引号后识别文字为空: {annotation_path}:{line_number}"
            )
        output_lines.append(format_annotation_line(image_path, text))
    return "".join(output_lines), changed_count


def main() -> int:
    args = parse_args()
    try:
        converted: list[tuple[Path, str, int]] = []
        for annotation_path in args.annotations:
            if not annotation_path.is_file():
                raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
            content = annotation_path.read_text(encoding="utf-8-sig")
            new_content, count = remove_outer_quotes_content(
                content, annotation_path
            )
            converted.append((annotation_path, new_content, count))

        if not args.check:
            for annotation_path, new_content, count in converted:
                if count:
                    atomic_write(annotation_path, new_content)
                print(f"{annotation_path}: 移除外围双引号 {count} 行")
        else:
            for annotation_path, _, count in converted:
                print(f"{annotation_path}: 待移除外围双引号 {count} 行")
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
