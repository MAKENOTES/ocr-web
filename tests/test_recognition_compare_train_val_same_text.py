from pathlib import Path

from recognition.compare_train_val_same_text import (
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
            'train_images/a.png, "相同,文字"',
            'train_images/b.png, "仅训练集"',
            'train_images/c.png, "相同,文字"',
        ],
    )
    write_annotation(
        val_path,
        [
            'val_images/d.png, "相同,文字"',
            'val_images/e.png, "仅验证集"',
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
    write_annotation(train_path, ['train_images/a.png, "甲"'])
    write_annotation(val_path, ['val_images/b.png, "乙"'])

    common_texts = find_common_texts(
        load_annotations(train_path), load_annotations(val_path)
    )

    assert format_common_texts(common_texts) == ""


def test_reports_malformed_annotation_line(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    write_annotation(train_path, ["train_images/a.png 没有分隔逗号"])
    write_annotation(val_path, ['val_images/b.png, "文字"'])

    try:
        load_annotations(train_path)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("格式错误的标注行应该抛出 ValueError")

    assert f"{train_path}:1" in error
    assert "缺少路径与文字之间的逗号" in error
