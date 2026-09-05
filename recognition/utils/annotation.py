"""Recognition 标注行的共享解析与格式化工具。"""

from __future__ import annotations

from pathlib import Path


ANNOTATION_SEPARATOR = "\t"


def split_annotation_fields(
    raw_line: str,
    annotation_path: str | Path,
    line_number: int,
) -> tuple[str, str]:
    """按 ``图片路径<TAB>文字`` 契约拆分一行标注。"""
    line = raw_line.rstrip("\r\n")
    if not line:
        raise ValueError(f"空白标注行: {annotation_path}:{line_number}")

    path_field, separator, text_field = line.partition(ANNOTATION_SEPARATOR)
    if not separator:
        raise ValueError(
            "标注格式错误，缺少路径与文字之间的 Tab 制表符: "
            f"{annotation_path}:{line_number}"
        )

    image_path = path_field.strip()
    if not image_path:
        raise ValueError(f"图片路径为空: {annotation_path}:{line_number}")
    return image_path, text_field


def format_annotation_line(image_path: str, text_field: str) -> str:
    """生成一行标准 Tab 分隔标注，并保留文字字段内容。"""
    return f"{image_path}{ANNOTATION_SEPARATOR}{text_field}\n"
