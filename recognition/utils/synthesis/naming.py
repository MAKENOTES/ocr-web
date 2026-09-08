"""三个合成入口的批次、文件名和报告目录约定。"""

from __future__ import annotations

from pathlib import Path
import re


REPORT_ROOT = Path(__file__).resolve().parent / "synthesis_report"


def numbered_filename(batch: str, date: str, number: int) -> str:
    if not re.fullmatch(r"synthesis[1-9]\d*", batch) or number <= 0:
        raise ValueError(f"Invalid synthesis batch or sequence: {batch!r}, {number}")
    return f"{batch}_{date}_{number:05d}.png"


def numbered_outputs(data_dir: Path, date: str, output_dir: Path | None = None,
                     annotation: Path | None = None) -> tuple[str, Path, Path]:
    """默认取第一个未占用编号；显式路径包含编号时，图片名沿用该编号。"""
    numbers = set()
    for path, pattern in ((output_dir, r"synthesis([1-9]\d*)_images"),
                          (annotation, r"synthesis([1-9]\d*)_annotation\.txt")):
        match = re.fullmatch(pattern, path.name) if path is not None else None
        if match:
            numbers.add(int(match[1]))
    if len(numbers) > 1:
        raise ValueError("Output directory and annotation have different synthesis batch numbers")
    number = next(iter(numbers)) if numbers else 1
    while not numbers:
        batch = f"synthesis{number}"
        candidates = [data_dir / f"{batch}_images", data_dir / f"{batch}_annotation.txt",
                      data_dir / f"{batch}_annotation.txt.partial", REPORT_ROOT / f"{batch}_{date}"]
        if not any(path.exists() or path.is_symlink() for path in candidates):
            break
        number += 1
    batch = f"synthesis{number}"
    return batch, output_dir or data_dir / f"{batch}_images", annotation or data_dir / f"{batch}_annotation.txt"
