from pathlib import Path

import pytest

from recognition.utils import compare_train_val_same_text
from recognition.utils.compare_train_val_same_text import (
    find_common_texts,
    format_common_texts,
    load_annotations,
)


@pytest.mark.parametrize("outside_repo", [False, True])
def test_default_paths_are_relative_to_recognition_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_repo: bool
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path if outside_repo else repo_root)
    monkeypatch.setattr("sys.argv", ["compare_train_val_same_text.py"])

    args = compare_train_val_same_text.parse_args()

    expected_data_dir = repo_root / "recognition" / "data"
    assert args.train_annotation == expected_data_dir / "train_annotation.txt"
    assert args.val_annotation == expected_data_dir / "val_annotation.txt"
    assert compare_train_val_same_text.OUTPUT_PATH == (
        expected_data_dir / "train_val_same_text.txt"
    )


def write_annotation(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def test_outputs_common_text_with_all_train_and_val_paths(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    write_annotation(
        train_path,
        [
            "train_images/a.png\t相同,文字",
            "train_images/b.png\t仅训练集",
            "train_images/c.png\t相同,文字",
        ],
    )
    write_annotation(
        val_path,
        [
            "val_images/d.png\t相同,文字",
            "val_images/e.png\t仅验证集",
            "val_images/nested/a.png\t相同,文字",
        ],
    )

    train_records = load_annotations(train_path)
    val_records = load_annotations(val_path)
    common_texts = find_common_texts(train_records, val_records)
    output = format_common_texts(common_texts, train_records, val_records)

    assert output == (
        "[1] 文字: '相同,文字'\n"
        "  train:\n"
        "    train_images/a.png\n"
        "    train_images/c.png\n"
        "  val:\n"
        "    val_images/d.png\n"
        "    val_images/nested/a.png\n\n"
    )


def test_reports_when_there_are_no_matches(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    write_annotation(train_path, ["train_images/a.png\t甲"])
    write_annotation(val_path, ["val_images/b.png\t乙"])

    train_records = load_annotations(train_path)
    val_records = load_annotations(val_path)
    common_texts = find_common_texts(train_records, val_records)

    assert format_common_texts(common_texts, train_records, val_records) == ""


def test_main_writes_train_and_val_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    output_path = tmp_path / "report.txt"
    write_annotation(train_path, ["train/a.png\t文字"])
    write_annotation(val_path, ["val/b.png\t文字"])
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_train_val_same_text.py",
            "--train-annotation",
            str(train_path),
            "--val-annotation",
            str(val_path),
        ],
    )
    monkeypatch.setattr(compare_train_val_same_text, "OUTPUT_PATH", output_path)

    assert compare_train_val_same_text.main() == 0
    assert output_path.read_text(encoding="utf-8") == (
        "[1] 文字: '文字'\n"
        "  train:\n    train/a.png\n"
        "  val:\n    val/b.png\n\n"
    )


def test_reports_malformed_annotation_line(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    write_annotation(train_path, ["train_images/a.png 没有分隔符"])
    write_annotation(val_path, ["val_images/b.png\t文字"])

    try:
        load_annotations(train_path)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("格式错误的标注行应该抛出 ValueError")

    assert f"{train_path}:1" in error
    assert "缺少路径与文字之间的 Tab 制表符" in error
