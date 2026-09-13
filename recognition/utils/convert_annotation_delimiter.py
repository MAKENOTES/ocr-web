"""把 Recognition 标注从首个逗号分隔转换为 Tab 分隔。"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path

try:
    from recognition.utils.annotation import ANNOTATION_SEPARATOR
except ModuleNotFoundError:  # 支持直接运行当前脚本
    from annotation import ANNOTATION_SEPARATOR


RECOGNITION_DIR = Path(__file__).resolve().parents[1]  # recognition 包目录，由当前文件位置推导。
# 旧分隔符转换的默认五份清单，不包含当前全部合成数据。
DEFAULT_ANNOTATIONS = (
    RECOGNITION_DIR / "data" / "all_annotation.txt",
    RECOGNITION_DIR / "data" / "append_annotation.txt",
    RECOGNITION_DIR / "data" / "train_annotation.txt",
    RECOGNITION_DIR / "data" / "val_annotation.txt",
    RECOGNITION_DIR / "data" / "val_unknown.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Recognition 标注统一为 图片路径<TAB>文字 格式。"
    )
    parser.add_argument(
        "annotations",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ANNOTATIONS),
        help="待转换文件；不传时转换 recognition/data 下的五个默认标注文件。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验并报告，不写入文件。",
    )
    return parser.parse_args()


def convert_content(content: str, annotation_path: Path) -> tuple[str, int]:
    """返回转换后的内容及由逗号改为 Tab 的行数。"""
    converted_lines: list[str] = []
    converted_count = 0
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        raw_line = line[: -len(newline)] if newline else line
        if not raw_line:
            raise ValueError(f"空白标注行: {annotation_path}:{line_number}")

        tab_index = raw_line.find(ANNOTATION_SEPARATOR)
        comma_index = raw_line.find(",")
        if tab_index >= 0 and (comma_index < 0 or tab_index < comma_index):
            path_field, text_field = raw_line.split(ANNOTATION_SEPARATOR, 1)
        elif comma_index >= 0:
            path_field, text_field = raw_line.split(",", 1)
            text_field = text_field.lstrip(" ")
            converted_count += 1
        else:
            raise ValueError(
                "标注格式错误，缺少路径与文字分隔符（逗号或 Tab）: "
                f"{annotation_path}:{line_number}"
            )

        image_path = path_field.strip()
        if not image_path:
            raise ValueError(f"图片路径为空: {annotation_path}:{line_number}")
        if not text_field.strip():
            raise ValueError(f"识别文字为空: {annotation_path}:{line_number}")
        converted_lines.append(
            f"{image_path}{ANNOTATION_SEPARATOR}{text_field}{newline}"
        )
    return "".join(converted_lines), converted_count


def atomic_write(path: Path, content: str) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, original_mode)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        converted: list[tuple[Path, str, int]] = []
        for annotation_path in args.annotations:
            if not annotation_path.is_file():
                raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
            content = annotation_path.read_text(encoding="utf-8-sig")
            new_content, count = convert_content(content, annotation_path)
            converted.append((annotation_path, new_content, count))

        if not args.check:
            for annotation_path, new_content, count in converted:
                if count:
                    atomic_write(annotation_path, new_content)
                print(f"{annotation_path}: 转换 {count} 行")
        else:
            for annotation_path, _, count in converted:
                print(f"{annotation_path}: 待转换 {count} 行")
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
