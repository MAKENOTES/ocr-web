import csv
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "detection_small" / "split_dataset.py"


def make_pairs(image_dir: Path, label_dir: Path, start: int, stop: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(start, stop):
        (image_dir / f"sample_{index}.png").write_bytes(f"image-{index}".encode())
        (label_dir / f"gt_sample_{index}.txt").write_text(
            f"0 0 10 10 text-{index}\n", encoding="utf-8"
        )


def run_split(
    source_images: Path,
    source_labels: Path,
    train_images: Path,
    train_labels: Path,
    val_images: Path,
    val_labels: Path,
    manifest: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-images",
            str(source_images),
            "--source-labels",
            str(source_labels),
            "--train-images",
            str(train_images),
            "--train-labels",
            str(train_labels),
            "--val-images",
            str(val_images),
            "--val-labels",
            str(val_labels),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def read_assignments(manifest: Path) -> dict[str, str]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row["split"] for row in csv.DictReader(handle)}


def assert_pairs(image_dir: Path, label_dir: Path) -> None:
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    label_stems = {path.stem.removeprefix("gt_") for path in label_dir.glob("*.txt")}
    assert image_stems == label_stems


def test_incremental_split_preserves_old_assignments(tmp_path: Path) -> None:
    train_images = tmp_path / "train_images"
    train_labels = tmp_path / "train_labels"
    val_images = tmp_path / "val_images"
    val_labels = tmp_path / "val_labels"
    manifest = tmp_path / "split_manifest.csv"

    make_pairs(train_images, train_labels, 0, 10)
    run_split(
        train_images,
        train_labels,
        train_images,
        train_labels,
        val_images,
        val_labels,
        manifest,
    )
    first_assignments = read_assignments(manifest)

    assert len(first_assignments) == 10
    assert list(first_assignments.values()).count("val") == 2
    assert len(list(train_images.glob("*.png"))) == 8
    assert len(list(val_images.glob("*.png"))) == 2
    assert_pairs(train_images, train_labels)
    assert_pairs(val_images, val_labels)

    all_images = tmp_path / "all_images"
    all_labels = tmp_path / "all_labels"
    make_pairs(all_images, all_labels, 10, 15)
    run_split(
        all_images,
        all_labels,
        train_images,
        train_labels,
        val_images,
        val_labels,
        manifest,
    )
    second_assignments = read_assignments(manifest)

    assert {key: second_assignments[key] for key in first_assignments} == first_assignments
    assert len(second_assignments) == 15
    assert list(second_assignments.values()).count("val") == 3
    assert len(list(train_images.glob("*.png"))) == 12
    assert len(list(val_images.glob("*.png"))) == 3
    assert_pairs(train_images, train_labels)
    assert_pairs(val_images, val_labels)
