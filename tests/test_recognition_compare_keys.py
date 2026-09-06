from pathlib import Path

import pytest

from recognition.utils.compare_keys import compare_keys


def test_writes_characters_missing_from_keys_across_all_annotations(
    tmp_path: Path,
) -> None:
    first_annotation = tmp_path / "train.txt"
    second_annotation = tmp_path / "append.txt"
    third_annotation = tmp_path / "bench.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = tmp_path / "missing.txt"
    first_annotation.write_text("train/a.png\t甲乙,\n", encoding="utf-8")
    second_annotation.write_text(
        'append/b.png\t乙"丙\n', encoding="utf-8"
    )
    third_annotation.write_text("bench/c.png\t甲丁\n", encoding="utf-8")
    keys_path.write_text("甲\n乙\n", encoding="utf-8")

    missing = compare_keys(
        annotation_files=[
            first_annotation,
            second_annotation,
            third_annotation,
        ],
        keys_file=keys_path,
        output_file=output_path,
    )

    expected = sorted({",", '"', "丙", "丁"})
    assert missing == expected
    assert output_path.read_text(encoding="utf-8").splitlines() == expected


def test_reports_annotation_path_and_line_for_invalid_format(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "train.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = tmp_path / "missing.txt"
    annotation_path.write_text("train/a.png,旧逗号格式\n", encoding="utf-8")
    keys_path.write_text("甲\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"train\.txt:1"):
        compare_keys([annotation_path], keys_path, output_path)

    assert not output_path.exists()


def test_rejects_multicharacter_keys_line(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = tmp_path / "missing.txt"
    annotation_path.write_text("train/a.png\t甲\n", encoding="utf-8")
    keys_path.write_text("甲乙\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"keys\.txt:1"):
        compare_keys([annotation_path], keys_path, output_path)

    assert not output_path.exists()
