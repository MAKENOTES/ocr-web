#!/usr/bin/env python3
"""核对验证集标注中的字符是否全部存在于 Recognition 字典。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __package__:
    from recognition.utils.dictionary import parse_annotation_text
else:
    from utils.dictionary import parse_annotation_text


RECOGNITION_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATION = RECOGNITION_DIR / "data" / "val_annotation.txt"
DEFAULT_KEYS = RECOGNITION_DIR / "keys.txt"
OUTPUT_PATH = RECOGNITION_DIR / "data" / "val_unknown_keys.txt"
UNKNOWN_IMAGES_DIR = RECOGNITION_DIR / "data" / "val_unknow"
UNKNOWN_ANNOTATION = RECOGNITION_DIR / "data" / "val_unknow.txt"


@dataclass(frozen=True)
class UnknownCharacterRecord:
    """一张包含字典外字符的验证图片。"""

    image_path: str
    text: str
    line_number: int
    unknown_chars: tuple[str, ...]


@dataclass(frozen=True)
class TransferJob:
    """一张待转移图片及其目标标注行。"""

    source_image: Path
    destination_image: Path
    line_number: int
    destination_annotation_line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="列出验证标注中包含 keys.txt 未收录字符的图片。"
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
        help="待核对的标注文件（默认：recognition/data/val_annotation.txt）。",
    )
    parser.add_argument(
        "--keys",
        type=Path,
        default=DEFAULT_KEYS,
        help="字符字典（默认：recognition/keys.txt）。",
    )
    parser.add_argument(
        "--move-unknown",
        action="store_true",
        help=(
            "把含字典外字符的图片和标注移出验证集，固定转移到 "
            "recognition/data/val_unknow/ 和 val_unknow.txt。"
        ),
    )
    return parser.parse_args()


def load_keys(keys_path: Path) -> set[str]:
    """按 LabelConverter 的换行处理规则读取并校验字符字典。"""
    if not keys_path.is_file():
        raise FileNotFoundError(f"字典文件不存在: {keys_path}")

    keys: set[str] = set()
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
            if char in keys:
                raise ValueError(
                    f"字典字符重复: {char!r}，位于 {keys_path}:"
                    f"{first_lines[char]} 和 {keys_path}:{line_number}"
                )
            keys.add(char)
            first_lines[char] = line_number

    if not keys:
        raise ValueError(f"字典文件没有有效字符: {keys_path}")
    return keys


def find_unknown_character_records(
    annotation_path: Path,
    keys: set[str],
) -> tuple[int, list[UnknownCharacterRecord]]:
    """返回标注总数及所有包含字典外字符的记录。"""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    record_count = 0
    unknown_records: list[UnknownCharacterRecord] = []
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = parse_annotation_text(line, annotation_path, line_number)
            image_path = line.partition(",")[0].strip()
            unknown_chars = tuple(
                dict.fromkeys(char for char in text if char not in keys)
            )
            record_count += 1
            if unknown_chars:
                unknown_records.append(
                    UnknownCharacterRecord(
                        image_path=image_path,
                        text=text,
                        line_number=line_number,
                        unknown_chars=unknown_chars,
                    )
                )

    if record_count == 0:
        raise ValueError(f"标注文件为空: {annotation_path}")
    return record_count, unknown_records


def format_report(
    annotation_path: Path,
    keys_path: Path,
    key_count: int,
    record_count: int,
    unknown_records: list[UnknownCharacterRecord],
) -> str:
    all_unknown_chars = sorted(
        {char for record in unknown_records for char in record.unknown_chars}
    )
    lines = [
        f"标注文件: {annotation_path}",
        f"字典文件: {keys_path}",
        f"字典字符数: {key_count}",
        f"验证图片数: {record_count}",
        f"含字典外字符的图片数: {len(unknown_records)}",
        f"字典外字符种类数: {len(all_unknown_chars)}",
    ]
    if all_unknown_chars:
        lines.append(f"全部字典外字符: {''.join(all_unknown_chars)!r}")

    for index, record in enumerate(unknown_records, start=1):
        lines.extend(
            (
                "",
                f"[{index}] 图片: {record.image_path} (第 {record.line_number} 行)",
                f"  文字: {record.text!r}",
                f"  字典未收录: {''.join(record.unknown_chars)!r}",
            )
        )

    if not unknown_records:
        lines.extend(("", "核对通过：验证集文字全部包含在字典中。"))
    return "\n".join(lines) + "\n"


def _validate_empty_transfer_targets(
    images_dir: Path,
    annotation_path: Path,
) -> None:
    if images_dir.exists():
        if not images_dir.is_dir():
            raise ValueError(f"转移目标不是目录: {images_dir}")
        if any(images_dir.iterdir()):
            raise ValueError(f"转移目标目录不是空目录: {images_dir}")

    if annotation_path.exists():
        if not annotation_path.is_file():
            raise ValueError(f"转移目标标注不是文件: {annotation_path}")
        if annotation_path.read_bytes():
            raise ValueError(f"转移目标标注文件不是空文件: {annotation_path}")


def build_transfer_plan(
    annotation_path: Path,
    unknown_records: list[UnknownCharacterRecord],
    destination_images_dir: Path,
) -> tuple[list[TransferJob], str, str]:
    """在修改文件前解析并校验全部图片，生成转移计划。"""
    annotation_dir = annotation_path.parent.resolve()
    source_images_dir = (annotation_dir / "val_images").resolve()
    destination_root = destination_images_dir.resolve()
    try:
        destination_root.relative_to(annotation_dir)
    except ValueError as exc:
        raise ValueError(
            f"图片转移目标必须位于标注目录内: {destination_images_dir}"
        ) from exc
    if destination_root == source_images_dir:
        raise ValueError("图片转移目标不能与 val_images 相同")

    lines = annotation_path.read_text(encoding="utf-8-sig").splitlines(
        keepends=True
    )
    unknown_by_line = {record.line_number: record for record in unknown_records}
    if len(unknown_by_line) != len(unknown_records):
        raise ValueError("待转移记录中存在重复标注行号")

    transfer_jobs: list[TransferJob] = []
    retained_lines: list[str] = []
    destination_lines: list[str] = []
    seen_destinations: set[Path] = set()

    for line_number, line in enumerate(lines, start=1):
        record = unknown_by_line.get(line_number)
        if record is None:
            retained_lines.append(line)
            continue

        raw_line = line.rstrip("\r\n")
        path_field, separator, text_field = raw_line.partition(",")
        image_path = path_field.strip()
        if not separator or image_path != record.image_path:
            raise ValueError(
                f"转移前标注内容发生变化: {annotation_path}:{line_number}"
            )

        posix_path = PurePosixPath(image_path)
        if (
            posix_path.is_absolute()
            or ".." in posix_path.parts
            or "\\" in image_path
        ):
            raise ValueError(
                f"待转移图片必须使用无 '..' 的相对路径: "
                f"{annotation_path}:{line_number}: {image_path!r}"
            )

        source_image = (
            annotation_dir / Path(*posix_path.parts)
        ).resolve()
        try:
            relative_image_path = source_image.relative_to(source_images_dir)
        except ValueError as exc:
            raise ValueError(
                f"待转移图片不在 val_images 中: "
                f"{annotation_path}:{line_number}: {image_path!r}"
            ) from exc
        if not source_image.is_file():
            raise FileNotFoundError(
                f"待转移图片不存在: {annotation_path}:{line_number}: "
                f"{source_image}"
            )

        destination_image = destination_root / relative_image_path
        if destination_image in seen_destinations:
            raise ValueError(f"多条标注指向同一转移目标: {destination_image}")
        if destination_image.exists():
            raise FileExistsError(f"图片转移目标已存在: {destination_image}")
        seen_destinations.add(destination_image)

        destination_annotation_path = destination_image.relative_to(
            annotation_dir
        ).as_posix()
        destination_annotation_line = (
            f"{destination_annotation_path},{text_field}\n"
        )
        transfer_jobs.append(
            TransferJob(
                source_image=source_image,
                destination_image=destination_image,
                line_number=line_number,
                destination_annotation_line=destination_annotation_line,
            )
        )
        destination_lines.append(destination_annotation_line)

    if len(transfer_jobs) != len(unknown_records):
        missing_lines = sorted(set(unknown_by_line) - set(range(1, len(lines) + 1)))
        raise ValueError(f"待转移标注行不存在: {missing_lines}")

    return transfer_jobs, "".join(retained_lines), "".join(destination_lines)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def move_unknown_records(
    annotation_path: Path,
    unknown_records: list[UnknownCharacterRecord],
    destination_images_dir: Path,
    destination_annotation_path: Path,
) -> int:
    """事务式移动问题图片和标注；失败时尽量恢复原状。"""
    if not unknown_records:
        return 0

    _validate_empty_transfer_targets(
        destination_images_dir,
        destination_annotation_path,
    )
    transfer_jobs, retained_content, destination_content = build_transfer_plan(
        annotation_path,
        unknown_records,
        destination_images_dir,
    )

    original_annotation = annotation_path.read_bytes()
    original_destination_annotation = (
        destination_annotation_path.read_bytes()
        if destination_annotation_path.exists()
        else None
    )
    destination_dir_existed = destination_images_dir.exists()
    moved_jobs: list[TransferJob] = []

    try:
        destination_images_dir.mkdir(parents=True, exist_ok=True)
        for job in transfer_jobs:
            job.destination_image.parent.mkdir(parents=True, exist_ok=True)
            job.source_image.replace(job.destination_image)
            moved_jobs.append(job)

        _atomic_write_bytes(
            destination_annotation_path,
            destination_content.encode("utf-8"),
        )
        _atomic_write_bytes(annotation_path, retained_content.encode("utf-8"))
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            _atomic_write_bytes(annotation_path, original_annotation)
        except OSError as rollback_exc:
            rollback_errors.append(f"恢复验证标注失败: {rollback_exc}")

        try:
            if original_destination_annotation is None:
                destination_annotation_path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(
                    destination_annotation_path,
                    original_destination_annotation,
                )
        except OSError as rollback_exc:
            rollback_errors.append(f"恢复目标标注失败: {rollback_exc}")

        for job in reversed(moved_jobs):
            try:
                job.source_image.parent.mkdir(parents=True, exist_ok=True)
                job.destination_image.replace(job.source_image)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"恢复图片失败 {job.destination_image}: {rollback_exc}"
                )

        if not destination_dir_existed:
            for directory in sorted(
                {job.destination_image.parent for job in moved_jobs},
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass

        if rollback_errors:
            raise RuntimeError(
                f"转移失败: {exc}；回滚不完整: {'；'.join(rollback_errors)}"
            ) from exc
        raise RuntimeError(f"转移失败，已恢复原数据: {exc}") from exc

    return len(transfer_jobs)


def main() -> int:
    args = parse_args()
    try:
        keys = load_keys(args.keys)
        record_count, unknown_records = find_unknown_character_records(
            args.annotation,
            keys,
        )
        report = format_report(
            annotation_path=args.annotation,
            keys_path=args.keys,
            key_count=len(keys),
            record_count=record_count,
            unknown_records=unknown_records,
        )
        OUTPUT_PATH.write_text(report, encoding="utf-8")
        moved_count = 0
        if args.move_unknown:
            moved_count = move_unknown_records(
                annotation_path=args.annotation,
                unknown_records=unknown_records,
                destination_images_dir=UNKNOWN_IMAGES_DIR,
                destination_annotation_path=UNKNOWN_ANNOTATION,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"含字典外字符的图片数: {len(unknown_records)}")
    print(f"结果已写入: {OUTPUT_PATH}")
    if args.move_unknown:
        print(f"已转移图片和标注: {moved_count} 条")
        if moved_count:
            print(f"图片目录: {UNKNOWN_IMAGES_DIR}")
            print(f"标注文件: {UNKNOWN_ANNOTATION}")
        return 0
    return 1 if unknown_records else 0


if __name__ == "__main__":
    raise SystemExit(main())
