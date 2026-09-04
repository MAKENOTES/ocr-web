#!/usr/bin/env python3
"""从 Recognition 错误案例报告中提取识别错误的真实字符。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


RECOGNITION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RECOGNITION_DIR / "error_cases.txt"
DEFAULT_OUTPUT = RECOGNITION_DIR / "keys-error.txt"
REPORT_HEADER = "CER\t文件名\t真实\t预测"


@dataclass(frozen=True)
class ErrorCase:
    """一条 Recognition 评估错误记录。"""

    filename: str
    ground_truth: str
    prediction: str
    line_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对齐 error_cases.txt 中的真实文字和预测文字，"
            "提取识别错误的真实字符。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="错误案例报告（默认：recognition/error_cases.txt）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出字符文件（默认：recognition/keys-error.txt）。",
    )
    return parser.parse_args()


def load_error_cases(report_path: Path) -> list[ErrorCase]:
    """读取 evaluate_val.py 生成的错误案例报告。"""
    if not report_path.is_file():
        raise FileNotFoundError(f"错误案例报告不存在: {report_path}")

    cases: list[ErrorCase] = []
    header_found = False
    with report_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\r\n")
            if not header_found:
                if raw_line == REPORT_HEADER:
                    header_found = True
                continue

            if not raw_line:
                continue

            fields = raw_line.split("\t", 3)
            if len(fields) != 4:
                raise ValueError(
                    "错误案例格式错误，应为四个制表符分隔字段: "
                    f"{report_path}:{line_number}"
                )

            cer_field, filename, ground_truth, prediction = fields
            try:
                float(cer_field)
            except ValueError as exc:
                raise ValueError(
                    f"CER 不是有效数字: {report_path}:{line_number}: "
                    f"{cer_field!r}"
                ) from exc
            if not filename:
                raise ValueError(
                    f"错误案例文件名为空: {report_path}:{line_number}"
                )
            if not ground_truth:
                raise ValueError(
                    f"错误案例真实文字为空: {report_path}:{line_number}"
                )

            cases.append(
                ErrorCase(
                    filename=filename,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    line_number=line_number,
                )
            )

    if not header_found:
        raise ValueError(f"错误案例报告缺少表头 {REPORT_HEADER!r}: {report_path}")
    if not cases:
        raise ValueError(f"错误案例报告中没有错误记录: {report_path}")
    return cases


def extract_error_characters(ground_truth: str, prediction: str) -> set[str]:
    """通过字符级编辑距离对齐，返回发生错误的真实字符。"""
    row_count = len(ground_truth) + 1
    column_count = len(prediction) + 1
    distances = [[0] * column_count for _ in range(row_count)]

    for row in range(row_count):
        distances[row][0] = row
    for column in range(column_count):
        distances[0][column] = column

    for row, true_char in enumerate(ground_truth, start=1):
        for column, predicted_char in enumerate(prediction, start=1):
            substitution_cost = 0 if true_char == predicted_char else 1
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + substitution_cost,
            )

    error_characters: set[str] = set()
    row = len(ground_truth)
    column = len(prediction)
    while row > 0 or column > 0:
        if (
            row > 0
            and column > 0
            and ground_truth[row - 1] == prediction[column - 1]
            and distances[row][column] == distances[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
            continue

        if (
            row > 0
            and column > 0
            and distances[row][column] == distances[row - 1][column - 1] + 1
        ):
            error_characters.add(ground_truth[row - 1])
            row -= 1
            column -= 1
            continue

        if row > 0 and distances[row][column] == distances[row - 1][column] + 1:
            error_characters.add(ground_truth[row - 1])
            row -= 1
            continue

        if column > 0 and distances[row][column] == distances[row][column - 1] + 1:
            inserted_char = prediction[column - 1]
            # 把重复输出归因到相邻的真实字符，例如 ``：`` 被识别为 ``：：``。
            if row > 0 and ground_truth[row - 1] == inserted_char:
                error_characters.add(ground_truth[row - 1])
            elif row < len(ground_truth) and ground_truth[row] == inserted_char:
                error_characters.add(ground_truth[row])
            column -= 1
            continue

        raise RuntimeError("字符对齐回溯失败")

    return error_characters


def extract_error_keys(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> tuple[int, int]:
    """提取并写入错误字符，返回错误案例数和唯一字符数。"""
    report_path = Path(input_path)
    destination_path = Path(output_path)
    cases = load_error_cases(report_path)

    error_characters: set[str] = set()
    for case in cases:
        error_characters.update(
            extract_error_characters(case.ground_truth, case.prediction)
        )

    if not error_characters:
        raise ValueError(f"错误案例中没有可提取的真实错误字符: {report_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        "".join(f"{char}\n" for char in sorted(error_characters)),
        encoding="utf-8",
    )
    return len(cases), len(error_characters)


def main() -> int:
    args = parse_args()
    try:
        case_count, character_count = extract_error_keys(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"已处理错误案例: {case_count} 条")
    print(f"已提取唯一错误字符: {character_count} 个")
    print(f"结果已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
