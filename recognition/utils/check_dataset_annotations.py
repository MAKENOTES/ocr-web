#!/usr/bin/env python3
"""校对 Recognition 的 all/train/val/val_unknown 四份标注清单。

校对以图片文件名（例如 ``du20260902-002391.png``）为唯一键：

1. ``all`` 必须恰好等于 ``train``、``val`` 与 ``val_unknown`` 的并集，
   且三个子集必须两两不重叠；
2. 同一文件名在 all 与所属 split 中的识别文字必须一致。

脚本只读取标注文件，不修改标注或图片。校对通过返回 0，发现差异返回 1。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class AnnotationRecord:
    """以图片文件名为键的一条标注。"""

    annotation_path: Path
    line_number: int
    image_path: str
    filename: str
    label: str


@dataclass(frozen=True)
class LabelMismatch:
    """all 与某个 split 中同名图片的文字差异。"""

    filename: str
    split_name: str
    all_record: AnnotationRecord
    split_record: AnnotationRecord


@dataclass(frozen=True)
class AuditResult:
    """四份标注清单的集合与文字校对结果。"""

    train_val_overlap: tuple[str, ...]
    train_val_unknown_overlap: tuple[str, ...]
    val_val_unknown_overlap: tuple[str, ...]
    missing_from_splits: tuple[str, ...]
    extra_in_train: tuple[str, ...]
    extra_in_val: tuple[str, ...]
    extra_in_val_unknown: tuple[str, ...]
    label_mismatches: tuple[LabelMismatch, ...]

    @property
    def is_valid(self) -> bool:
        return not any(
            (
                self.train_val_overlap,
                self.train_val_unknown_overlap,
                self.val_val_unknown_overlap,
                self.missing_from_splits,
                self.extra_in_train,
                self.extra_in_val,
                self.extra_in_val_unknown,
                self.label_mismatches,
            )
        )


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(
        description=(
            "按图片文件名校对 all/train/val/val_unknown 标注的集合与识别文字。"
        )
    )
    parser.add_argument(
        "--all-annotation",
        type=Path,
        default=data_dir / "all_annotation.txt",
        help="全量标注文件。",
    )
    parser.add_argument(
        "--train-annotation",
        type=Path,
        default=data_dir / "train_annotation.txt",
        help="训练集标注文件。",
    )
    parser.add_argument(
        "--val-annotation",
        type=Path,
        default=data_dir / "val_annotation.txt",
        help="验证集标注文件。",
    )
    parser.add_argument(
        "--val-unknown-annotation",
        type=Path,
        default=data_dir / "val_unknown.txt",
        help="含字典外字符的隔离验证集标注文件。",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=20,
        help="每类问题最多显示多少条详情；0 表示全部显示（默认：20）。",
    )
    return parser.parse_args()


def normalize_label(label_field: str) -> str:
    """去掉格式空白和一对外围双引号，保留实际文字内容。"""
    label = label_field.strip()
    if len(label) >= 2 and label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    return label.strip()


def load_annotation_records(annotation_path: Path) -> dict[str, AnnotationRecord]:
    """读取标注，并按不含目录的文件名建立唯一映射。"""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    records: dict[str, AnnotationRecord] = {}
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\r\n")
            if not raw_line:
                raise ValueError(f"空白标注行: {annotation_path}:{line_number}")

            path_field, separator, label_field = raw_line.partition(",")
            if not separator:
                raise ValueError(
                    f"标注格式错误，缺少路径与文字之间的逗号: "
                    f"{annotation_path}:{line_number}"
                )

            image_path = path_field.strip()
            if not image_path:
                raise ValueError(f"图片路径为空: {annotation_path}:{line_number}")
            if "\\" in image_path:
                raise ValueError(
                    f"标注路径必须使用正斜杠 '/': {annotation_path}:{line_number}: "
                    f"{image_path!r}"
                )

            posix_path = PurePosixPath(image_path)
            if posix_path.is_absolute() or ".." in posix_path.parts:
                raise ValueError(
                    f"标注路径必须是无 '..' 的相对路径: "
                    f"{annotation_path}:{line_number}: {image_path!r}"
                )
            if posix_path.suffix != ".png":
                raise ValueError(
                    f"Recognition 标注图片必须精确使用 .png 扩展名: "
                    f"{annotation_path}:{line_number}: {image_path!r}"
                )

            label = normalize_label(label_field)
            if not label:
                raise ValueError(f"识别文字为空: {annotation_path}:{line_number}")

            filename = posix_path.name
            if filename in records:
                first = records[filename]
                raise ValueError(
                    f"同一标注文件中图片文件名重复: {filename!r}，位于 "
                    f"{annotation_path}:{first.line_number} 和 "
                    f"{annotation_path}:{line_number}；校对要求文件名唯一"
                )

            records[filename] = AnnotationRecord(
                annotation_path=annotation_path,
                line_number=line_number,
                image_path=image_path,
                filename=filename,
                label=label,
            )

    if not records:
        raise ValueError(f"标注文件为空: {annotation_path}")

    return records


def audit_annotations(
    all_records: dict[str, AnnotationRecord],
    train_records: dict[str, AnnotationRecord],
    val_records: dict[str, AnnotationRecord],
    val_unknown_records: dict[str, AnnotationRecord],
) -> AuditResult:
    all_names = set(all_records)
    train_names = set(train_records)
    val_names = set(val_records)
    val_unknown_names = set(val_unknown_records)
    split_names = train_names | val_names | val_unknown_names

    label_mismatches: list[LabelMismatch] = []
    for split_name, split_records in (
        ("train", train_records),
        ("val", val_records),
        ("val_unknown", val_unknown_records),
    ):
        for filename in sorted(all_names & set(split_records)):
            all_record = all_records[filename]
            split_record = split_records[filename]
            if all_record.label != split_record.label:
                label_mismatches.append(
                    LabelMismatch(
                        filename=filename,
                        split_name=split_name,
                        all_record=all_record,
                        split_record=split_record,
                    )
                )

    return AuditResult(
        train_val_overlap=tuple(sorted(train_names & val_names)),
        train_val_unknown_overlap=tuple(sorted(train_names & val_unknown_names)),
        val_val_unknown_overlap=tuple(sorted(val_names & val_unknown_names)),
        missing_from_splits=tuple(sorted(all_names - split_names)),
        extra_in_train=tuple(sorted(train_names - all_names)),
        extra_in_val=tuple(sorted(val_names - all_names)),
        extra_in_val_unknown=tuple(sorted(val_unknown_names - all_names)),
        label_mismatches=tuple(label_mismatches),
    )


def _limited(items: tuple[T, ...], max_details: int) -> tuple[T, ...]:
    if max_details == 0:
        return items
    return items[:max_details]


def _print_name_problem(
    title: str,
    filenames: tuple[str, ...],
    max_details: int,
) -> None:
    if not filenames:
        return
    print(f"[错误] {title}: {len(filenames)}")
    for filename in _limited(filenames, max_details):
        print(f"  - {filename}")
    if max_details and len(filenames) > max_details:
        print(f"  ... 另有 {len(filenames) - max_details} 条未显示")


def print_report(
    all_records: dict[str, AnnotationRecord],
    train_records: dict[str, AnnotationRecord],
    val_records: dict[str, AnnotationRecord],
    val_unknown_records: dict[str, AnnotationRecord],
    result: AuditResult,
    max_details: int,
) -> None:
    print(f"all 标注数: {len(all_records)}")
    print(f"train 标注数: {len(train_records)}")
    print(f"val 标注数: {len(val_records)}")
    print(f"val_unknown 标注数: {len(val_unknown_records)}")

    filename_sets_valid = not any(
        (
            result.train_val_overlap,
            result.train_val_unknown_overlap,
            result.val_val_unknown_overlap,
            result.missing_from_splits,
            result.extra_in_train,
            result.extra_in_val,
            result.extra_in_val_unknown,
        )
    )
    _print_name_problem(
        "train 与 val 中出现相同文件名",
        result.train_val_overlap,
        max_details,
    )
    _print_name_problem(
        "train 与 val_unknown 中出现相同文件名",
        result.train_val_unknown_overlap,
        max_details,
    )
    _print_name_problem(
        "val 与 val_unknown 中出现相同文件名",
        result.val_val_unknown_overlap,
        max_details,
    )
    _print_name_problem(
        "all 中的文件名未分配到 train、val 或 val_unknown",
        result.missing_from_splits,
        max_details,
    )
    _print_name_problem(
        "train 中的文件名不存在于 all",
        result.extra_in_train,
        max_details,
    )
    _print_name_problem(
        "val 中的文件名不存在于 all",
        result.extra_in_val,
        max_details,
    )
    _print_name_problem(
        "val_unknown 中的文件名不存在于 all",
        result.extra_in_val_unknown,
        max_details,
    )
    if filename_sets_valid:
        print(
            "文件名集合一致: all = train + val + val_unknown，"
            "且三个集合两两无重叠。"
        )

    if result.label_mismatches:
        print(f"[错误] 同名图片的识别文字不一致: {len(result.label_mismatches)}")
        for mismatch in _limited(result.label_mismatches, max_details):
            all_record = mismatch.all_record
            split_record = mismatch.split_record
            print(
                f"  - {mismatch.filename}: "
                f"all 第 {all_record.line_number} 行={all_record.label!r}；"
                f"{mismatch.split_name} 第 {split_record.line_number} 行="
                f"{split_record.label!r}"
            )
        if max_details and len(result.label_mismatches) > max_details:
            print(
                f"  ... 另有 {len(result.label_mismatches) - max_details} 条未显示"
            )
    else:
        print(
            "识别文字一致: all 与 train/val/val_unknown 中的同名图片文字完全相同。"
        )

    if result.is_valid:
        print("校对通过。")
    else:
        print("校对失败，请按上述文件名修正标注。")


def main() -> int:
    args = parse_args()
    if args.max_details < 0:
        print("错误: --max-details 不能小于 0", file=sys.stderr)
        return 1

    try:
        all_records = load_annotation_records(args.all_annotation)
        train_records = load_annotation_records(args.train_annotation)
        val_records = load_annotation_records(args.val_annotation)
        val_unknown_records = load_annotation_records(args.val_unknown_annotation)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    result = audit_annotations(
        all_records,
        train_records,
        val_records,
        val_unknown_records,
    )
    print_report(
        all_records,
        train_records,
        val_records,
        val_unknown_records,
        result,
        args.max_details,
    )
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
