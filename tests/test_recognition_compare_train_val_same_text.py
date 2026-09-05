from pathlib import Path

from recognition.utils.compare_train_val_same_text import (
    find_common_texts,
    format_common_texts,
    load_annotations,
)


def write_annotation(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def test_outputs_only_each_common_text(tmp_path: Path) -> None:
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
        ],
    )

    common_texts = find_common_texts(
        load_annotations(train_path), load_annotations(val_path)
    )
    output = format_common_texts(common_texts)

    assert output == "[1] 文字: '相同,文字'\n"
    assert "train_images" not in output
    assert "val_images" not in output


def test_reports_when_there_are_no_matches(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    write_annotation(train_path, ["train_images/a.png\t甲"])
    write_annotation(val_path, ["val_images/b.png\t乙"])

    common_texts = find_common_texts(
        load_annotations(train_path), load_annotations(val_path)
    )

    assert format_common_texts(common_texts) == ""


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
