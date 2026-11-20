#!/usr/bin/env python3
"""核对 train.py 使用的训练标注是否覆盖 keys.txt 的全部字符。

训练标注列表直接读取 ``recognition.training_data.TRAIN_ANNOTATION``，与 train.py
交给 TextRecognitionDataset 的输入保持一致。脚本按训练规则解析标注：只按首个 Tab
拆分，文字 strip 后为空则跳过；只读标注文本，不检查图片。

检查两个方向：
1. keys.txt 中有、训练文字中没有的字符：模型缺少该字符的正样本；
2. 训练文字中有、keys.txt 中没有的字符：训练时会被编码为 ``[UNK]``。

两个方向都没有问题时返回 0，否则列出具体字符并返回 1；读取或格式错误返回 2。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录，直接运行脚本时用于导入 recognition 包。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.training_data import TRAIN_ANNOTATION
from recognition.utils.annotation import split_annotation_fields


RECOGNITION_DIR = Path(__file__).resolve().parents[1]  # recognition 包目录，由当前文件位置推导。
DEFAULT_KEYS = RECOGNITION_DIR / "keys.txt"  # 默认字符字典；内容和顺序属于模型 ABI。


@dataclass(frozen=True)
class KeyEntry:
    """字典中的一个字符及其在字典中的位置。"""

    char: str
    index: int  # 过滤空行后的 1 基序号；LabelConverter 的类别编号为 index + 1（前面还有 blank/UNK）。


@dataclass(frozen=True)
class AnnotationScan:
    """训练标注的文字统计结果。"""

    annotation_paths: tuple[Path, ...]
    record_count: int
    skipped_empty_labels: int
    char_counts: dict[str, int]  # 字符到训练文字中出现次数的映射。


@dataclass(frozen=True)
class CoverageReport:
    """训练集对字典的双向覆盖结果。"""

    scan: AnnotationScan
    keys: tuple[KeyEntry, ...]
    missing_keys: tuple[KeyEntry, ...]  # 字典中有、训练文字中没有的字符，保持字典顺序。
    unknown_chars: tuple[tuple[str, int], ...]  # 训练文字中有、字典中没有的字符及次数，按次数降序。

    @property
    def is_valid(self) -> bool:
        return not self.missing_keys and not self.unknown_chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="核对 train.py 使用的训练标注是否覆盖 keys.txt 的全部字符。"
    )
    parser.add_argument(
        "--keys",
        type=Path,
        default=DEFAULT_KEYS,
        help="字符字典（默认：recognition/keys.txt）。",
    )
    return parser.parse_args()


def load_keys(keys_path: Path) -> tuple[KeyEntry, ...]:
    """按 LabelConverter 的换行处理规则读取字典，并校验每行一个字符且不重复。"""
    if not keys_path.is_file():
        raise FileNotFoundError(f"字典文件不存在: {keys_path}")

    entries: list[KeyEntry] = []
    first_lines: dict[str, int] = {}
    with keys_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            char = line.rstrip("\n").rstrip("\r")
            if not char:
                continue
            if len(char) != 1:
                raise ValueError(
                    f"字典每个非空行必须只有一个字符: "
                    f"{keys_path}:{line_number}: {char!r}"
                )
            if char in first_lines:
                raise ValueError(
                    f"字典字符重复: {char!r}，位于 {keys_path}:"
                    f"{first_lines[char]} 和 {keys_path}:{line_number}"
                )
            entries.append(KeyEntry(char=char, index=len(entries) + 1))
            first_lines[char] = line_number

    if not entries:
        raise ValueError(f"字典文件没有有效字符: {keys_path}")
    return tuple(entries)


def resolve_annotation_paths(
    annotation_paths: tuple[str, ...] | list[str],
    root: Path = ROOT,
) -> tuple[Path, ...]:
    """把 training_data 中的相对路径解析到仓库根目录，绝对路径保持不变。"""
    resolved_paths: list[Path] = []
    for raw_path in annotation_paths:
        path = Path(raw_path)
        resolved_paths.append(path if path.is_absolute() else root / path)
    return tuple(resolved_paths)


def scan_annotations(annotation_paths: tuple[Path, ...] | list[Path]) -> AnnotationScan:
    """按训练规则读取标注文字，统计每个字符的出现次数。"""
    char_counts: Counter[str] = Counter()
    record_count = 0
    skipped_empty_labels = 0
    resolved_paths: list[Path] = []

    for annotation_path in annotation_paths:
        path = Path(annotation_path)
        if not path.is_file():
            raise FileNotFoundError(f"标注文件不存在: {path}")
        resolved_paths.append(path)

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                _, label_field = split_annotation_fields(line, path, line_number)
                label = label_field.strip()
                if not label:
                    skipped_empty_labels += 1
                    continue

                record_count += 1
                char_counts.update(label)

    if record_count == 0:
        raise ValueError("训练标注没有非空文字记录")

    return AnnotationScan(
        annotation_paths=tuple(resolved_paths),
        record_count=record_count,
        skipped_empty_labels=skipped_empty_labels,
        char_counts=dict(char_counts),
    )


def build_report(keys: tuple[KeyEntry, ...], scan: AnnotationScan) -> CoverageReport:
    """比较训练文字与字典，生成两个方向的覆盖结果。"""
    key_chars = {entry.char for entry in keys}
    missing_keys = tuple(
        entry for entry in keys if entry.char not in scan.char_counts
    )
    unknown_chars = tuple(
        sorted(
            (
                (char, count)
                for char, count in scan.char_counts.items()
                if char not in key_chars
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return CoverageReport(
        scan=scan,
        keys=keys,
        missing_keys=missing_keys,
        unknown_chars=unknown_chars,
    )


def _format_char(char: str) -> str:
    """同时显示字符和码点，便于定位空格、控制符等不可见字符。"""
    return f"{char!r} (U+{ord(char):04X})"


def format_report(report: CoverageReport) -> str:
    scan = report.scan
    total_chars = sum(scan.char_counts.values())
    covered_count = len(report.keys) - len(report.missing_keys)
    lines = [
        "训练标注（train.py 使用的训练集）:",
        *(f"  - {path}" for path in scan.annotation_paths),
        f"训练记录数: {scan.record_count}",
        f"跳过的空标签: {scan.skipped_empty_labels}",
        f"训练文字总字符数: {total_chars}",
        f"字典字符数: {len(report.keys)}",
        f"已覆盖字典字符: {covered_count}",
        f"未覆盖字典字符: {len(report.missing_keys)}",
    ]

    if report.missing_keys:
        lines.extend(
            ("", f"[错误] keys.txt 中未被训练集覆盖的字符: {len(report.missing_keys)}")
        )
        for entry in report.missing_keys:
            lines.append(f"  字典序号 {entry.index}: {_format_char(entry.char)}")

    if report.unknown_chars:
        unknown_total = sum(count for _, count in report.unknown_chars)
        lines.extend(
            (
                "",
                "[警告] 训练文字中 keys.txt 未收录的字符: "
                f"{len(report.unknown_chars)} 种，共 {unknown_total} 次；"
                "训练时会编码为 [UNK]",
            )
        )
        for char, count in report.unknown_chars:
            lines.append(f"  {_format_char(char)}: {count} 次")

    lines.append("")
    if report.is_valid:
        lines.append(
            f"核对通过：训练集覆盖 keys.txt 的全部 {len(report.keys)} 个字符，"
            "且没有字典外字符。"
        )
    else:
        problems = []
        if report.missing_keys:
            problems.append(f"{len(report.missing_keys)} 个字典字符未在训练文字中出现")
        if report.unknown_chars:
            problems.append(
                f"{len(report.unknown_chars)} 种字典外字符出现在训练文字中"
            )
        lines.append(f"核对失败：{'；'.join(problems)}。")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        keys = load_keys(args.keys)
        annotation_paths = resolve_annotation_paths(TRAIN_ANNOTATION)
        scan = scan_annotations(annotation_paths)
        report = build_report(keys, scan)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(format_report(report))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
