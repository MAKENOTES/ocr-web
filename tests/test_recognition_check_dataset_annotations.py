import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "recognition"
    / "check_dataset_annotations.py"
)


def write_annotation(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def run_check(data_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--all-annotation",
            str(data_dir / "all_annotation.txt"),
            "--train-annotation",
            str(data_dir / "train_annotation.txt"),
            "--val-annotation",
            str(data_dir / "val_annotation.txt"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_passes_when_filename_union_and_labels_match(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_annotation(
        data_dir / "all_annotation.txt",
        [
            'all_images/du20260902-002391.png, "文字,一"',
            'all_images/sample_b.png, "文字二"',
            'all_images/sample_c.png, "文字三"',
        ],
    )
    write_annotation(
        data_dir / "train_annotation.txt",
        [
            'train_images/sample_c.png, "文字三"',
            'train_images/du20260902-002391.png, "文字,一"',
        ],
    )
    write_annotation(
        data_dir / "val_annotation.txt",
        ['val_images/sample_b.png, "文字二"'],
    )

    result = run_check(data_dir)

    assert result.returncode == 0, result.stderr
    assert "all 标注数: 3" in result.stdout
    assert "文件名集合一致" in result.stdout
    assert "识别文字一致" in result.stdout
    assert "校对通过" in result.stdout


def test_check_reports_missing_extra_and_train_val_overlap(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_annotation(
        data_dir / "all_annotation.txt",
        [
            'all_images/sample_a.png, "甲"',
            'all_images/sample_b.png, "乙"',
        ],
    )
    write_annotation(
        data_dir / "train_annotation.txt",
        [
            'train_images/sample_a.png, "甲"',
            'train_images/extra.png, "额外"',
        ],
    )
    write_annotation(
        data_dir / "val_annotation.txt",
        ['val_images/sample_a.png, "甲"'],
    )

    result = run_check(data_dir)

    assert result.returncode == 1
    assert "train 与 val 中出现相同文件名: 1" in result.stdout
    assert "sample_a.png" in result.stdout
    assert "all 中的文件名未分配到 train 或 val: 1" in result.stdout
    assert "sample_b.png" in result.stdout
    assert "train 中的文件名不存在于 all: 1" in result.stdout
    assert "extra.png" in result.stdout
    assert "校对失败" in result.stdout


def test_check_reports_different_text_for_same_filename(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_annotation(
        data_dir / "all_annotation.txt",
        [
            'all_images/du20260902-002391.png, "原文字"',
            'all_images/sample_b.png, "相同文字"',
        ],
    )
    write_annotation(
        data_dir / "train_annotation.txt",
        ['train_images/du20260902-002391.png, "修改后的文字"'],
    )
    write_annotation(
        data_dir / "val_annotation.txt",
        ['val_images/sample_b.png, "相同文字"'],
    )

    result = run_check(data_dir)

    assert result.returncode == 1
    assert "同名图片的识别文字不一致: 1" in result.stdout
    assert "du20260902-002391.png" in result.stdout
    assert "all 第 1 行='原文字'" in result.stdout
    assert "train 第 1 行='修改后的文字'" in result.stdout


def test_check_rejects_duplicate_basename_even_when_directories_differ(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    write_annotation(
        data_dir / "all_annotation.txt",
        [
            'all_images/group_a/sample.png, "甲"',
            'all_images/group_b/sample.png, "乙"',
        ],
    )
    write_annotation(
        data_dir / "train_annotation.txt",
        ['train_images/sample.png, "甲"'],
    )
    write_annotation(
        data_dir / "val_annotation.txt",
        ['val_images/other.png, "乙"'],
    )

    result = run_check(data_dir)

    assert result.returncode == 1
    assert "图片文件名重复" in result.stderr
    assert "sample.png" in result.stderr
    assert "校对要求文件名唯一" in result.stderr
