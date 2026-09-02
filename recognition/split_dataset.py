#!/usr/bin/env python3
"""将 Recognition 全量训练数据按标注顺序均匀划分为训练集和验证集。

脚本默认只预演，不修改任何文件。确认输出无误后，传入 ``--apply`` 才会：

1. 从 ``train_annotation.txt`` 中均匀抽取最接近指定比例的样本；
2. 将对应图片从 ``train_images`` 移动（或复制）到 ``val_images``；
3. 重写 ``train_annotation.txt``，并生成 ``val_annotation.txt``。

``all_annotation.txt`` 不会被修改，可继续作为全量数据清单。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class AnnotationEntry:
    """一条已完成路径校验的 Recognition 标注。"""

    line_number: int
    image_path: Path
    relative_image_path: Path
    label_field: str


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(
        description=(
            "按标注顺序均匀抽取验证集。默认仅预演；添加 --apply 后才修改数据。"
        )
    )
    parser.add_argument(
        "--train-images",
        type=Path,
        default=data_dir / "train_images",
        help="当前包含全部图片的目录。",
    )
    parser.add_argument(
        "--train-annotation",
        type=Path,
        default=data_dir / "train_annotation.txt",
        help="当前包含全部标注的文件；应用划分后只保留训练集标注。",
    )
    parser.add_argument(
        "--val-images",
        type=Path,
        default=data_dir / "val_images",
        help="验证集图片输出目录。",
    )
    parser.add_argument(
        "--val-annotation",
        type=Path,
        default=data_dir / "val_annotation.txt",
        help="验证集标注输出文件。",
    )
    parser.add_argument(
        "--val-ratio",
        default="0.20",
        help="验证集比例，使用 0 到 1 之间的小数（默认：0.20）。",
    )
    parser.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="应用划分时移动或复制验证图片（默认：move）。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行划分；不传此参数时只校验并显示计划。",
    )
    return parser.parse_args()


def parse_ratio(raw_ratio: str) -> Decimal:
    try:
        ratio = Decimal(raw_ratio)
    except InvalidOperation as exc:
        raise ValueError(f"--val-ratio 不是有效小数: {raw_ratio!r}") from exc

    if not Decimal("0") < ratio < Decimal("1"):
        raise ValueError(f"--val-ratio 必须大于 0 且小于 1，当前为 {raw_ratio!r}")
    return ratio


def validation_sample_count(total_count: int, ratio: Decimal) -> int:
    """返回最接近目标比例的整数样本数，0.5 时向上取整。"""
    if total_count <= 0:
        raise ValueError("训练标注为空，无法划分数据集")

    count = int(
        (Decimal(total_count) * ratio).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    if count <= 0 or count >= total_count:
        raise ValueError(
            f"{total_count} 条数据按 {ratio} 划分后无法同时生成非空训练集和验证集"
        )
    return count


def systematic_validation_indices(total_count: int, val_count: int) -> list[int]:
    """按原始顺序均匀返回恰好 ``val_count`` 个零基索引。

    当数据总数不能被目标数量整除时，相邻抽样间隔会在两个相邻整数之间
    自动调整。例如 20% 通常每隔 5 条抽取 1 条，必要时少量间隔为 4，
    从而保证最终抽取数量精确等于计算出的整数目标。
    """
    if not 0 < val_count < total_count:
        raise ValueError(
            f"val_count 必须在 1 到 total_count - 1 之间，当前为 {val_count}"
        )

    indices = [
        index
        for index in range(total_count)
        if ((index + 1) * val_count) // total_count
        > (index * val_count) // total_count
    ]
    if len(indices) != val_count:
        raise RuntimeError(
            f"系统抽样数量异常: 期望 {val_count}，实际 {len(indices)}"
        )
    return indices


def _normalized_label(label_field: str) -> str:
    label = label_field.strip()
    if len(label) >= 2 and label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    return label.strip()


def load_annotations(
    annotation_path: Path,
    train_images: Path,
) -> list[AnnotationEntry]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"训练标注文件不存在: {annotation_path}")
    if not train_images.is_dir():
        raise FileNotFoundError(f"训练图片目录不存在: {train_images}")

    annotation_dir = annotation_path.parent.resolve()
    train_root = train_images.resolve()
    entries: list[AnnotationEntry] = []
    seen_paths: dict[Path, int] = {}

    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\r\n")
            if not raw_line:
                raise ValueError(f"空白标注行: {annotation_path}:{line_number}")

            path_field, separator, label_field = raw_line.partition(",")
            if not separator:
                raise ValueError(
                    f"标注格式错误，缺少路径与文本之间的逗号: "
                    f"{annotation_path}:{line_number}"
                )

            path_text = path_field.strip()
            if not path_text:
                raise ValueError(f"图片路径为空: {annotation_path}:{line_number}")
            if not _normalized_label(label_field):
                raise ValueError(f"识别文本为空: {annotation_path}:{line_number}")
            if "\\" in path_text:
                raise ValueError(
                    f"标注路径必须使用正斜杠 '/': {annotation_path}:{line_number}: "
                    f"{path_text!r}"
                )

            posix_path = PurePosixPath(path_text)
            if posix_path.is_absolute() or ".." in posix_path.parts:
                raise ValueError(
                    f"标注路径必须是无 '..' 的相对路径: "
                    f"{annotation_path}:{line_number}: {path_text!r}"
                )
            if posix_path.suffix != ".png":
                raise ValueError(
                    f"Recognition 标注图片必须精确使用 .png 扩展名: "
                    f"{annotation_path}:{line_number}: {path_text!r}"
                )

            image_path = (annotation_dir / Path(*posix_path.parts)).resolve()
            try:
                relative_image_path = image_path.relative_to(train_root)
            except ValueError as exc:
                raise ValueError(
                    f"标注图片不在训练图片目录中: {annotation_path}:{line_number}: "
                    f"{image_path}"
                ) from exc

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"标注对应图片不存在: {annotation_path}:{line_number}: {image_path}"
                )
            if image_path in seen_paths:
                first_line = seen_paths[image_path]
                raise ValueError(
                    f"图片被重复标注: {image_path}，位于第 {first_line} 行和第 "
                    f"{line_number} 行"
                )

            seen_paths[image_path] = line_number
            entries.append(
                AnnotationEntry(
                    line_number=line_number,
                    image_path=image_path,
                    relative_image_path=relative_image_path,
                    label_field=label_field,
                )
            )

    if not entries:
        raise ValueError(f"训练标注为空: {annotation_path}")

    annotated_paths = set(seen_paths)
    disk_paths = {
        path.resolve()
        for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    }
    unannotated_paths = sorted(disk_paths - annotated_paths)
    if unannotated_paths:
        examples = ", ".join(str(path) for path in unannotated_paths[:5])
        raise ValueError(
            f"训练图片目录中有 {len(unannotated_paths)} 个 PNG 未出现在标注文件中；"
            f"示例: {examples}"
        )

    return entries


def validate_empty_outputs(val_images: Path, val_annotation: Path) -> None:
    if val_annotation.exists():
        if not val_annotation.is_file():
            raise ValueError(f"验证标注路径不是文件: {val_annotation}")
        if val_annotation.read_text(encoding="utf-8-sig").strip():
            raise ValueError(
                f"验证标注文件不是空文件，为防止重复划分已停止: {val_annotation}"
            )

    if val_images.exists():
        if not val_images.is_dir():
            raise ValueError(f"验证图片路径不是目录: {val_images}")
        existing_file = next(
            (path for path in val_images.rglob("*") if path.is_file()), None
        )
        if existing_file is not None:
            raise ValueError(
                f"验证图片目录不是空目录，为防止覆盖已停止: {existing_file}"
            )


def validate_paths(
    train_images: Path,
    val_images: Path,
    train_annotation: Path,
    val_annotation: Path,
) -> None:
    train_root = train_images.resolve()
    val_root = val_images.resolve()
    if (
        train_root == val_root
        or train_root in val_root.parents
        or val_root in train_root.parents
    ):
        raise ValueError(
            f"训练图片目录与验证图片目录不能相同或互相嵌套: "
            f"{train_root}, {val_root}"
        )
    if train_annotation.resolve() == val_annotation.resolve():
        raise ValueError("训练标注文件与验证标注文件不能是同一个文件")


def annotation_line(
    entry: AnnotationEntry,
    image_path: Path,
    annotation_path: Path,
) -> str:
    relative_path = os.path.relpath(image_path, start=annotation_path.parent)
    return f"{Path(relative_path).as_posix()},{entry.label_field}\n"


def build_annotation_contents(
    entries: list[AnnotationEntry],
    val_indices: set[int],
    train_images: Path,
    val_images: Path,
    train_annotation: Path,
    val_annotation: Path,
) -> tuple[str, str, list[AnnotationEntry]]:
    train_lines: list[str] = []
    val_lines: list[str] = []
    val_entries: list[AnnotationEntry] = []

    for index, entry in enumerate(entries):
        if index in val_indices:
            val_entries.append(entry)
            destination = val_images / entry.relative_image_path
            val_lines.append(annotation_line(entry, destination, val_annotation))
        else:
            destination = train_images / entry.relative_image_path
            train_lines.append(annotation_line(entry, destination, train_annotation))

    return "".join(train_lines), "".join(val_lines), val_entries


def _rollback_transfers(
    completed: list[tuple[Path, Path]],
    mode: str,
) -> list[str]:
    errors: list[str] = []
    for source, destination in reversed(completed):
        try:
            if mode == "move":
                if destination.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
            elif destination.exists():
                destination.unlink()
        except OSError as exc:
            errors.append(f"无法回滚 {destination}: {exc}")
    return errors


def transfer_validation_images(
    entries: list[AnnotationEntry],
    val_images: Path,
    mode: str,
) -> list[tuple[Path, Path]]:
    destinations = [val_images / entry.relative_image_path for entry in entries]
    collisions = [path for path in destinations if path.exists()]
    if collisions:
        examples = ", ".join(str(path) for path in collisions[:5])
        raise FileExistsError(f"验证图片目标已存在，拒绝覆盖: {examples}")

    completed: list[tuple[Path, Path]] = []
    try:
        for entry, destination in zip(entries, destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if mode == "move":
                shutil.move(str(entry.image_path), str(destination))
            else:
                shutil.copy2(entry.image_path, destination)
            completed.append((entry.image_path, destination))
    except BaseException as exc:
        rollback_errors = _rollback_transfers(completed, mode)
        if rollback_errors:
            raise RuntimeError(
                f"图片处理失败且回滚不完整: {exc}; " + "; ".join(rollback_errors)
            ) from exc
        raise
    return completed


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def apply_split(
    val_entries: list[AnnotationEntry],
    train_content: str,
    val_content: str,
    train_annotation: Path,
    val_annotation: Path,
    val_images: Path,
    mode: str,
) -> None:
    original_train = train_annotation.read_bytes()
    val_existed = val_annotation.exists()
    original_val = val_annotation.read_bytes() if val_existed else b""

    completed = transfer_validation_images(val_entries, val_images, mode)
    try:
        atomic_write_bytes(train_annotation, train_content.encode("utf-8"))
        atomic_write_bytes(val_annotation, val_content.encode("utf-8"))
    except BaseException as exc:
        restore_errors: list[str] = []
        try:
            atomic_write_bytes(train_annotation, original_train)
        except OSError as restore_exc:
            restore_errors.append(f"无法恢复 {train_annotation}: {restore_exc}")
        try:
            if val_existed:
                atomic_write_bytes(val_annotation, original_val)
            else:
                val_annotation.unlink(missing_ok=True)
        except OSError as restore_exc:
            restore_errors.append(f"无法恢复 {val_annotation}: {restore_exc}")
        restore_errors.extend(_rollback_transfers(completed, mode))
        if restore_errors:
            raise RuntimeError(
                f"写入标注失败且回滚不完整: {exc}; " + "; ".join(restore_errors)
            ) from exc
        raise


def print_plan(
    total_count: int,
    val_count: int,
    ratio: Decimal,
    val_indices: list[int],
    mode: str,
    apply: bool,
) -> None:
    train_count = total_count - val_count
    actual_ratio = Decimal(val_count) / Decimal(total_count)
    selected_positions = [index + 1 for index in val_indices]
    gaps = [
        right - left
        for left, right in zip(selected_positions, selected_positions[1:])
    ]

    print(f"数据总数: {total_count}")
    print(f"训练集: {train_count}")
    print(
        f"验证集: {val_count}（目标 {ratio * 100}%，"
        f"实际 {actual_ratio * 100:.6f}%）"
    )
    print(f"平均抽样间隔: {Decimal(total_count) / Decimal(val_count):.6f} 条")
    if gaps:
        print(f"相邻验证样本的实际间隔: {min(gaps)}～{max(gaps)} 条")
    print(f"图片处理方式: {'移动' if mode == 'move' else '复制'}")
    print(f"运行模式: {'执行划分' if apply else '仅预演，不修改文件'}")


def main() -> int:
    args = parse_args()
    try:
        ratio = parse_ratio(args.val_ratio)
        validate_paths(
            args.train_images,
            args.val_images,
            args.train_annotation,
            args.val_annotation,
        )
        validate_empty_outputs(args.val_images, args.val_annotation)
        entries = load_annotations(args.train_annotation, args.train_images)
        val_count = validation_sample_count(len(entries), ratio)
        val_index_list = systematic_validation_indices(len(entries), val_count)
        train_content, val_content, val_entries = build_annotation_contents(
            entries=entries,
            val_indices=set(val_index_list),
            train_images=args.train_images.resolve(),
            val_images=args.val_images.resolve(),
            train_annotation=args.train_annotation.resolve(),
            val_annotation=args.val_annotation.resolve(),
        )

        print_plan(
            total_count=len(entries),
            val_count=val_count,
            ratio=ratio,
            val_indices=val_index_list,
            mode=args.mode,
            apply=args.apply,
        )
        if not args.apply:
            print("预演完成。确认无误后添加 --apply 执行。")
            return 0

        apply_split(
            val_entries=val_entries,
            train_content=train_content,
            val_content=val_content,
            train_annotation=args.train_annotation.resolve(),
            val_annotation=args.val_annotation.resolve(),
            val_images=args.val_images.resolve(),
            mode=args.mode,
        )
        print("划分完成。")
        return 0
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
