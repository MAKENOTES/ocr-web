#!/usr/bin/env python3
"""按持久化 manifest 固定小图训练/验证归属，仅为新增样本分组。
默认执行移动；先用 --dry-run 核对目标目录，数据传输成功后再写清单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")  # 划分时支持的图片扩展名，大小写不敏感。
MANIFEST_FIELDS = ("sample_id", "image_name", "label_name", "split")  # 持久化划分清单字段及写出顺序。


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_name: str
    label_name: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class Assignment:
    sample_id: str
    image_name: str
    label_name: str
    split: str


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(
        description=(
            "Persistently split paired detection images and labels. Existing CSV "
            "assignments never change; only new samples are assigned."
        )
    )
    parser.add_argument("--source-images", type=Path, default=data_dir / "train_images")
    parser.add_argument("--source-labels", type=Path, default=data_dir / "train_labels")
    parser.add_argument("--train-images", type=Path, default=data_dir / "train_images")
    parser.add_argument("--train-labels", type=Path, default=data_dir / "train_labels")
    parser.add_argument("--val-images", type=Path, default=data_dir / "val_images")
    parser.add_argument("--val-labels", type=Path, default=data_dir / "val_labels")
    parser.add_argument("--manifest", type=Path, default=data_dir / "split_manifest.csv")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument(
        "--seed",
        default="42",
        help="Stable seed used only to order newly discovered samples (default: 42).",
    )
    parser.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="Move source pairs into their split, or copy them (default: move).",
    )
    parser.add_argument(
        "--image-extensions",
        default=",".join(DEFAULT_IMAGE_EXTENSIONS),
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned split without changing files or CSV.",
    )
    return parser.parse_args()


def normalize_extensions(raw: str) -> set[str]:
    extensions = set()
    for value in raw.split(","):
        value = value.strip().lower()
        if value:
            extensions.add(value if value.startswith(".") else f".{value}")
    if not extensions:
        raise ValueError("--image-extensions cannot be empty")
    return extensions


def discover_samples(image_dir: Path, label_dir: Path, extensions: set[str]) -> list[Sample]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    seen_stems: dict[str, Path] = {}
    samples: list[Sample] = []
    missing_labels: list[str] = []
    ambiguous_labels: list[str] = []

    for image_path in image_paths:
        sample_id = image_path.stem
        if sample_id in seen_stems:
            first = seen_stems[sample_id]
            raise ValueError(
                f"Two images have the same stem and would share one label: {first.name}, "
                f"{image_path.name}"
            )
        seen_stems[sample_id] = image_path

        candidates = [label_dir / f"gt_{sample_id}.txt", label_dir / f"{sample_id}.txt"]
        matches = [path for path in candidates if path.is_file()]
        if not matches:
            missing_labels.append(image_path.name)
            continue
        if len(matches) > 1:
            ambiguous_labels.append(image_path.name)
            continue

        label_path = matches[0]
        samples.append(
            Sample(
                sample_id=sample_id,
                image_name=image_path.name,
                label_name=label_path.name,
                image_path=image_path,
                label_path=label_path,
            )
        )

    if missing_labels or ambiguous_labels:
        messages = []
        if missing_labels:
            messages.append(
                f"{len(missing_labels)} image(s) have no matching label; examples: "
                + ", ".join(missing_labels[:5])
            )
        if ambiguous_labels:
            messages.append(
                f"{len(ambiguous_labels)} image(s) match both gt_<stem>.txt and <stem>.txt; "
                "examples: " + ", ".join(ambiguous_labels[:5])
            )
        raise ValueError("\n".join(messages))

    if not samples:
        raise ValueError(f"No paired samples found in {image_dir} and {label_dir}")
    return samples


def load_manifest(path: Path) -> dict[str, Assignment]:
    if not path.exists():
        return {}

    assignments: dict[str, Assignment] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(MANIFEST_FIELDS) - set(reader.fieldnames):
            raise ValueError(
                f"Manifest {path} must contain columns: {', '.join(MANIFEST_FIELDS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            assignment = Assignment(**{field: row[field].strip() for field in MANIFEST_FIELDS})
            if not assignment.sample_id:
                raise ValueError(f"Empty sample_id in {path}:{line_number}")
            if assignment.split not in {"train", "val"}:
                raise ValueError(f"Invalid split in {path}:{line_number}: {assignment.split!r}")
            if assignment.sample_id in assignments:
                raise ValueError(f"Duplicate sample_id in {path}:{line_number}: {assignment.sample_id}")
            assignments[assignment.sample_id] = assignment
    return assignments


def stable_order(sample: Sample, seed: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{sample.sample_id}".encode("utf-8")).hexdigest()
    return digest, sample.sample_id


def assign_new_samples(
    samples: list[Sample],
    existing: dict[str, Assignment],
    val_ratio: float,
    seed: str,
) -> tuple[dict[str, Assignment], int]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"--val-ratio must be between 0 and 1, got {val_ratio}")

    new_samples = []
    for sample in samples:
        old = existing.get(sample.sample_id)
        if old is None:
            new_samples.append(sample)
            continue
        if old.image_name != sample.image_name or old.label_name != sample.label_name:
            raise ValueError(
                f"Sample {sample.sample_id!r} conflicts with its manifest row: "
                f"CSV has ({old.image_name}, {old.label_name}), source has "
                f"({sample.image_name}, {sample.label_name})"
            )

    assignments = dict(existing)
    target_val_count = round((len(existing) + len(new_samples)) * val_ratio)
    existing_val_count = sum(item.split == "val" for item in existing.values())
    new_val_count = min(len(new_samples), max(0, target_val_count - existing_val_count))
    new_val_ids = {
        sample.sample_id
        for sample in sorted(new_samples, key=lambda item: stable_order(item, seed))[:new_val_count]
    }

    for sample in new_samples:
        assignments[sample.sample_id] = Assignment(
            sample_id=sample.sample_id,
            image_name=sample.image_name,
            label_name=sample.label_name,
            split="val" if sample.sample_id in new_val_ids else "train",
        )
    return assignments, len(new_samples)


def same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if not left_chunk and not right_chunk:
                return left_hash.digest() == right_hash.digest()
            left_hash.update(left_chunk)
            right_hash.update(right_chunk)


def destination_paths(
    sample: Sample,
    assignment: Assignment,
    train_images: Path,
    train_labels: Path,
    val_images: Path,
    val_labels: Path,
) -> tuple[Path, Path]:
    if assignment.split == "val":
        return val_images / sample.image_name, val_labels / sample.label_name
    return train_images / sample.image_name, train_labels / sample.label_name


def validate_destinations(
    samples: Iterable[Sample],
    assignments: dict[str, Assignment],
    args: argparse.Namespace,
) -> None:
    errors = []
    for sample in samples:
        image_dst, label_dst = destination_paths(
            sample,
            assignments[sample.sample_id],
            args.train_images,
            args.train_labels,
            args.val_images,
            args.val_labels,
        )
        for source, destination in ((sample.image_path, image_dst), (sample.label_path, label_dst)):
            if same_path(source, destination) or not destination.exists():
                continue
            if not files_equal(source, destination):
                errors.append(f"Destination exists with different content: {destination}")
    if errors:
        raise FileExistsError("\n".join(errors[:10]))


def transfer_file(source: Path, destination: Path, mode: str) -> str:
    if same_path(source, destination):
        return "unchanged"
    if destination.exists():
        # validate_destinations already proved that the files are identical.
        if mode == "move":
            source.unlink()
            return "deduplicated"
        return "unchanged"
    if mode == "move":
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)
    return mode + "d"


def write_manifest(path: Path, assignments: dict[str, Assignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for sample_id in sorted(assignments):
                writer.writerow(vars(assignments[sample_id]))
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    args = parse_args()
    try:
        extensions = normalize_extensions(args.image_extensions)
        samples = discover_samples(args.source_images, args.source_labels, extensions)
        old_assignments = load_manifest(args.manifest)
        assignments, new_count = assign_new_samples(
            samples, old_assignments, args.val_ratio, args.seed
        )
        validate_destinations(samples, assignments, args)

        train_count = sum(item.split == "train" for item in assignments.values())
        val_count = sum(item.split == "val" for item in assignments.values())
        print(f"Paired source samples: {len(samples)}")
        print(f"Existing manifest rows: {len(old_assignments)}")
        print(f"New samples: {new_count}")
        print(f"Persistent split: train={train_count}, val={val_count}, total={len(assignments)}")

        if args.dry_run:
            print("Dry run: no files or manifest were changed.")
            return 0

        for directory in (args.train_images, args.train_labels, args.val_images, args.val_labels):
            directory.mkdir(parents=True, exist_ok=True)

        # Persist assignments before moving files. If a transfer is interrupted,
        # rerunning the command resumes with exactly the same assignments.
        write_manifest(args.manifest, assignments)

        operation_counts: dict[str, int] = {}
        for sample in samples:
            image_dst, label_dst = destination_paths(
                sample,
                assignments[sample.sample_id],
                args.train_images,
                args.train_labels,
                args.val_images,
                args.val_labels,
            )
            for source, destination in ((sample.image_path, image_dst), (sample.label_path, label_dst)):
                result = transfer_file(source, destination, args.mode)
                operation_counts[result] = operation_counts.get(result, 0) + 1

        operations = ", ".join(f"{key}={value}" for key, value in sorted(operation_counts.items()))
        print(f"File operations: {operations}")
        print(f"Manifest written: {args.manifest}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
