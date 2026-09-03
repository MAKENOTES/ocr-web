#!/usr/bin/env python3
"""从 Recognition 全量数据确定性地生成 80/20 训练集和验证集。

脚本默认只预演，不修改任何文件。确认输出无误后，传入 ``--apply`` 才会：

1. 读取 ``all_annotation.txt`` 和 ``all_images`` 中的全量数据；
2. 按标注顺序均匀选择最接近 20% 的样本作为验证集，其余作为训练集；
3. 将图片复制到 ``train_images`` / ``val_images``；
4. 生成 ``train_annotation.txt`` / ``val_annotation.txt``。

全量图片和 ``all_annotation.txt`` 始终保持不变。为避免覆盖手工调整的数据，训练集或
验证集输出只要不是空的，脚本就会停止。
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


CopyJob = tuple[AnnotationEntry, Path]


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(
        description=(
            "从 all 全量数据按标注顺序均匀生成训练集和验证集。"
            "默认仅预演；添加 --apply 后才复制图片并写入标注。"
        )
    )
    parser.add_argument(
        "--all-images",
        type=Path,
        default=data_dir / "all_images",
        help="包含全部图片的源目录。",
    )
    parser.add_argument(
        "--all-annotation",
        type=Path,
        default=data_dir / "all_annotation.txt",
        help="包含全部标注的源文件。",
    )
    parser.add_argument(
        "--train-images",
        type=Path,
        default=data_dir / "train_images",
        help="训练集图片输出目录。",
    )
    parser.add_argument(
        "--train-annotation",
        type=Path,
        default=data_dir / "train_annotation.txt",
        help="训练集标注输出文件。",
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
        raise ValueError("全量标注为空，无法划分数据集")

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
    all_images: Path,
) -> list[AnnotationEntry]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"全量标注文件不存在: {annotation_path}")
    if not all_images.is_dir():
        raise FileNotFoundError(f"全量图片目录不存在: {all_images}")

    annotation_dir = annotation_path.parent.resolve()
    all_root = all_images.resolve()
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
                relative_image_path = image_path.relative_to(all_root)
            except ValueError as exc:
                raise ValueError(
                    f"标注图片不在全量图片目录中: {annotation_path}:{line_number}: "
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
        raise ValueError(f"全量标注为空: {annotation_path}")

    annotated_paths = set(seen_paths)
    disk_paths = {
        path.resolve()
        for path in all_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    }
    unannotated_paths = sorted(disk_paths - annotated_paths)
    if unannotated_paths:
        examples = ", ".join(str(path) for path in unannotated_paths[:5])
        raise ValueError(
            f"全量图片目录中有 {len(unannotated_paths)} 个 PNG 未出现在标注文件中；"
            f"示例: {examples}"
        )

    return entries


def _validate_empty_output(
    image_dir: Path,
    annotation_path: Path,
    split_name: str,
) -> None:
    if annotation_path.exists():
        if not annotation_path.is_file():
            raise ValueError(f"{split_name}标注路径不是文件: {annotation_path}")
        if annotation_path.read_text(encoding="utf-8-sig").strip():
            raise ValueError(
                f"{split_name}标注文件不是空文件，为防止覆盖手工数据已停止: "
                f"{annotation_path}"
            )

    if image_dir.exists():
        if not image_dir.is_dir():
            raise ValueError(f"{split_name}图片路径不是目录: {image_dir}")
        existing_file = next(
            (path for path in image_dir.rglob("*") if path.is_file()), None
        )
        if existing_file is not None:
            raise ValueError(
                f"{split_name}图片目录不是空目录，为防止覆盖手工数据已停止: "
                f"{existing_file}"
            )


def validate_empty_outputs(
    train_images: Path,
    train_annotation: Path,
    val_images: Path,
    val_annotation: Path,
) -> None:
    _validate_empty_output(train_images, train_annotation, "训练集")
    _validate_empty_output(val_images, val_annotation, "验证集")


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_paths(
    all_images: Path,
    train_images: Path,
    val_images: Path,
    all_annotation: Path,
    train_annotation: Path,
    val_annotation: Path,
) -> None:
    image_roots = {
        "全量": all_images.resolve(),
        "训练集": train_images.resolve(),
        "验证集": val_images.resolve(),
    }
    image_items = list(image_roots.items())
    for index, (left_name, left_path) in enumerate(image_items):
        for right_name, right_path in image_items[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise ValueError(
                    f"{left_name}图片目录与{right_name}图片目录不能相同或互相嵌套: "
                    f"{left_path}, {right_path}"
                )

    annotation_paths = {
        "全量": all_annotation.resolve(),
        "训练集": train_annotation.resolve(),
        "验证集": val_annotation.resolve(),
    }
    if len(set(annotation_paths.values())) != len(annotation_paths):
        details = ", ".join(
            f"{name}={path}" for name, path in annotation_paths.items()
        )
        raise ValueError(f"全量、训练集和验证集标注文件必须互不相同: {details}")


def annotation_line(
    entry: AnnotationEntry,
    image_path: Path,
    annotation_path: Path,
) -> str:
    relative_path = os.path.relpath(image_path, start=annotation_path.parent)
    return f"{Path(relative_path).as_posix()},{entry.label_field}\n"


def build_split(
    entries: list[AnnotationEntry],
    val_indices: set[int],
    train_images: Path,
    val_images: Path,
    train_annotation: Path,
    val_annotation: Path,
) -> tuple[str, str, list[CopyJob]]:
    train_lines: list[str] = []
    val_lines: list[str] = []
    copy_jobs: list[CopyJob] = []

    for index, entry in enumerate(entries):
        if index in val_indices:
            destination = val_images / entry.relative_image_path
            val_lines.append(annotation_line(entry, destination, val_annotation))
        else:
            destination = train_images / entry.relative_image_path
            train_lines.append(annotation_line(entry, destination, train_annotation))
        copy_jobs.append((entry, destination))

    return "".join(train_lines), "".join(val_lines), copy_jobs


def _rollback_copies(destinations: list[Path]) -> list[str]:
    errors: list[str] = []
    for destination in reversed(destinations):
        try:
            destination.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"无法删除已复制文件 {destination}: {exc}")
    return errors


def copy_split_images(copy_jobs: list[CopyJob]) -> list[Path]:
    destinations = [destination for _, destination in copy_jobs]
    collisions = [path for path in destinations if path.exists()]
    if collisions:
        examples = ", ".join(str(path) for path in collisions[:5])
        raise FileExistsError(f"训练集或验证集图片目标已存在，拒绝覆盖: {examples}")

    completed: list[Path] = []
    try:
        for entry, destination in copy_jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.image_path, destination)
            completed.append(destination)
    except BaseException as exc:
        rollback_errors = _rollback_copies(completed)
        if rollback_errors:
            raise RuntimeError(
                f"图片复制失败且回滚不完整: {exc}; " + "; ".join(rollback_errors)
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


def _restore_annotation(path: Path, original_content: bytes | None) -> None:
    if original_content is None:
        path.unlink(missing_ok=True)
    else:
        atomic_write_bytes(path, original_content)


def apply_split(
    copy_jobs: list[CopyJob],
    train_content: str,
    val_content: str,
    train_annotation: Path,
    val_annotation: Path,
) -> None:
    originals = {
        train_annotation: (
            train_annotation.read_bytes() if train_annotation.exists() else None
        ),
        val_annotation: val_annotation.read_bytes() if val_annotation.exists() else None,
    }

    completed = copy_split_images(copy_jobs)
    try:
        atomic_write_bytes(train_annotation, train_content.encode("utf-8"))
        atomic_write_bytes(val_annotation, val_content.encode("utf-8"))
    except BaseException as exc:
        restore_errors: list[str] = []
        for annotation_path, original_content in originals.items():
            try:
                _restore_annotation(annotation_path, original_content)
            except OSError as restore_exc:
                restore_errors.append(f"无法恢复 {annotation_path}: {restore_exc}")
        restore_errors.extend(_rollback_copies(completed))
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
    apply: bool,
) -> None:
    train_count = total_count - val_count
    actual_ratio = Decimal(val_count) / Decimal(total_count)
    selected_positions = [index + 1 for index in val_indices]
    gaps = [
        right - left
        for left, right in zip(selected_positions, selected_positions[1:])
    ]

    print(f"全量数据: {total_count}")
    print(f"训练集: {train_count}")
    print(
        f"验证集: {val_count}（目标 {ratio * 100}%，"
        f"实际 {actual_ratio * 100:.6f}%）"
    )
    print(f"平均抽样间隔: {Decimal(total_count) / Decimal(val_count):.6f} 条")
    if gaps:
        print(f"相邻验证样本的实际间隔: {min(gaps)}～{max(gaps)} 条")
    print("图片处理方式: 从 all_images 复制到 train_images / val_images")
    print(f"运行模式: {'执行划分' if apply else '仅预演，不修改文件'}")


def main() -> int:
    args = parse_args()
    try:
        ratio = parse_ratio(args.val_ratio)
        validate_paths(
            args.all_images,
            args.train_images,
            args.val_images,
            args.all_annotation,
            args.train_annotation,
            args.val_annotation,
        )
        validate_empty_outputs(
            args.train_images,
            args.train_annotation,
            args.val_images,
            args.val_annotation,
        )
        entries = load_annotations(args.all_annotation, args.all_images)
        val_count = validation_sample_count(len(entries), ratio)
        val_index_list = systematic_validation_indices(len(entries), val_count)
        train_content, val_content, copy_jobs = build_split(
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
            apply=args.apply,
        )
        if not args.apply:
            print("预演完成。确认无误后添加 --apply 执行。")
            return 0

        apply_split(
            copy_jobs=copy_jobs,
            train_content=train_content,
            val_content=val_content,
            train_annotation=args.train_annotation.resolve(),
            val_annotation=args.val_annotation.resolve(),
        )
        print("划分完成；all 全量数据未修改。")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
