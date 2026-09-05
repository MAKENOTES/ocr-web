import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from recognition.split_dataset import (
    systematic_validation_indices,
    validation_sample_count,
)


SCRIPT = Path(__file__).resolve().parents[1] / "recognition" / "split_dataset.py"


def make_dataset(data_dir: Path, count: int) -> str:
    all_images = data_dir / "all_images"
    all_images.mkdir(parents=True)
    lines = []
    for index in range(count):
        image_name = f"sample_{index:03d}.png"
        (all_images / image_name).write_bytes(f"image-{index}".encode())
        lines.append(f"all_images/{image_name}\t标签,{index}\n")

    annotation_content = "".join(lines)
    (data_dir / "all_annotation.txt").write_text(
        annotation_content, encoding="utf-8"
    )
    (data_dir / "train_annotation.txt").write_text("", encoding="utf-8")
    (data_dir / "val_annotation.txt").write_text("", encoding="utf-8")
    return annotation_content


def run_split(data_dir: Path, *, apply: bool = False) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--all-images",
        str(data_dir / "all_images"),
        "--all-annotation",
        str(data_dir / "all_annotation.txt"),
        "--train-images",
        str(data_dir / "train_images"),
        "--train-annotation",
        str(data_dir / "train_annotation.txt"),
        "--val-images",
        str(data_dir / "val_images"),
        "--val-annotation",
        str(data_dir / "val_annotation.txt"),
    ]
    if apply:
        command.append("--apply")
    return subprocess.run(command, capture_output=True, text=True, check=False)


def annotation_paths(path: Path) -> list[str]:
    return [
        line.partition("\t")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_systematic_split_selects_nearest_twenty_percent_exactly() -> None:
    total_count = 47_808
    val_count = validation_sample_count(total_count, Decimal("0.20"))
    indices = systematic_validation_indices(total_count, val_count)

    assert val_count == 9_562
    assert len(indices) == 9_562
    assert set(right - left for left, right in zip(indices, indices[1:])) == {4, 5}


def test_default_run_only_previews_without_modifying_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    original_annotation = make_dataset(data_dir, 10)

    result = run_split(data_dir)

    assert result.returncode == 0, result.stderr
    assert "全量数据: 10" in result.stdout
    assert "训练集: 8" in result.stdout
    assert "验证集: 2" in result.stdout
    assert "仅预演，不修改文件" in result.stdout
    assert (
        data_dir / "all_annotation.txt"
    ).read_text(encoding="utf-8") == original_annotation
    assert (data_dir / "train_annotation.txt").read_text(encoding="utf-8") == ""
    assert (data_dir / "val_annotation.txt").read_text(encoding="utf-8") == ""
    assert len(list((data_dir / "all_images").glob("*.png"))) == 10
    assert not (data_dir / "train_images").exists()
    assert not (data_dir / "val_images").exists()


def test_apply_copies_every_fifth_image_and_writes_both_splits(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    original_annotation = make_dataset(data_dir, 10)

    result = run_split(data_dir, apply=True)

    assert result.returncode == 0, result.stderr
    assert "划分完成" in result.stdout
    assert annotation_paths(data_dir / "val_annotation.txt") == [
        "val_images/sample_004.png",
        "val_images/sample_009.png",
    ]
    assert annotation_paths(data_dir / "train_annotation.txt") == [
        f"train_images/sample_{index:03d}.png"
        for index in (0, 1, 2, 3, 5, 6, 7, 8)
    ]
    assert "val_images/sample_004.png\t标签,4" in (
        data_dir / "val_annotation.txt"
    ).read_text(encoding="utf-8")
    assert len(list((data_dir / "all_images").glob("*.png"))) == 10
    assert len(list((data_dir / "train_images").glob("*.png"))) == 8
    assert len(list((data_dir / "val_images").glob("*.png"))) == 2
    assert (data_dir / "all_images" / "sample_004.png").is_file()
    assert (data_dir / "val_images" / "sample_004.png").read_bytes() == b"image-4"
    assert (
        data_dir / "all_annotation.txt"
    ).read_text(encoding="utf-8") == original_annotation


def test_missing_all_image_stops_before_any_change(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    original_annotation = make_dataset(data_dir, 10)
    (data_dir / "all_images" / "sample_003.png").unlink()

    result = run_split(data_dir, apply=True)

    assert result.returncode == 1
    assert "标注对应图片不存在" in result.stderr
    assert (
        data_dir / "all_annotation.txt"
    ).read_text(encoding="utf-8") == original_annotation
    assert (data_dir / "train_annotation.txt").read_text(encoding="utf-8") == ""
    assert (data_dir / "val_annotation.txt").read_text(encoding="utf-8") == ""
    assert not (data_dir / "train_images").exists()
    assert not (data_dir / "val_images").exists()


def test_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    make_dataset(data_dir, 10)
    existing = "train_images/manual.png\t手工标注\n"
    (data_dir / "train_annotation.txt").write_text(existing, encoding="utf-8")

    result = run_split(data_dir, apply=True)

    assert result.returncode == 1
    assert "为防止覆盖手工数据已停止" in result.stderr
    assert (data_dir / "train_annotation.txt").read_text(encoding="utf-8") == existing
    assert not (data_dir / "train_images").exists()
    assert not (data_dir / "val_images").exists()
