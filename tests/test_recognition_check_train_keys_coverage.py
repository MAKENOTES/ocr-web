from pathlib import Path

import pytest

from recognition.tools.check_train_keys_coverage import (
    CoverageReport,
    build_report,
    format_report,
    load_keys,
    resolve_annotation_paths,
    scan_annotations,
)


def write_annotation(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def prepare_report(
    tmp_path: Path, keys: list[str], labels: list[str]
) -> CoverageReport:
    keys_path = tmp_path / "keys.txt"
    keys_path.write_text("".join(f"{char}\n" for char in keys), encoding="utf-8")
    annotation_path = write_annotation(
        tmp_path / "data" / "train_annotation.txt", labels
    )
    return build_report(load_keys(keys_path), scan_annotations([annotation_path]))


def test_full_coverage_passes(tmp_path: Path) -> None:
    report = prepare_report(
        tmp_path,
        keys=["甲", "乙"],
        labels=["images/a.png\t甲乙", "images/b.png\t甲"],
    )

    assert report.is_valid
    assert report.scan.record_count == 2
    assert report.scan.skipped_empty_labels == 0
    assert report.scan.char_counts == {"甲": 2, "乙": 1}
    assert "核对通过" in format_report(report)


def test_missing_keys_keep_dictionary_order_and_index(tmp_path: Path) -> None:
    report = prepare_report(
        tmp_path,
        keys=["甲", "乙", "丙"],
        labels=["images/a.png\t甲"],
    )

    assert not report.is_valid
    assert [entry.char for entry in report.missing_keys] == ["乙", "丙"]
    assert [entry.index for entry in report.missing_keys] == [2, 3]

    text = format_report(report)
    assert "未被训练集覆盖的字符: 2" in text
    assert "字典序号 2: '乙' (U+4E59)" in text
    assert "字典序号 3: '丙' (U+4E19)" in text
    assert "核对失败" in text


def test_unknown_chars_reported_and_empty_labels_skipped(tmp_path: Path) -> None:
    report = prepare_report(
        tmp_path,
        keys=["甲"],
        labels=["images/a.png\t甲丁丁", "images/b.png\t   "],
    )

    assert not report.is_valid
    assert report.scan.record_count == 1
    assert report.scan.skipped_empty_labels == 1
    assert report.unknown_chars == (("丁", 2),)

    text = format_report(report)
    assert "未收录的字符: 1 种，共 2 次" in text
    assert "'丁' (U+4E01): 2 次" in text


def test_unknown_chars_sorted_by_count_descending(tmp_path: Path) -> None:
    report = prepare_report(
        tmp_path,
        keys=["甲"],
        labels=["images/a.png\t甲丁戊戊戊"],
    )

    assert [char for char, _ in report.unknown_chars] == ["戊", "丁"]


def test_malformed_annotation_reports_path_and_line(tmp_path: Path) -> None:
    annotation_path = write_annotation(
        tmp_path / "data" / "train_annotation.txt",
        ["images/a.png\t甲", "缺少制表符"],
    )

    with pytest.raises(ValueError) as excinfo:
        scan_annotations([annotation_path])

    assert "标注格式错误" in str(excinfo.value)
    assert f"{annotation_path}:2" in str(excinfo.value)


def test_load_keys_rejects_duplicate_char(tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.txt"
    keys_path.write_text("甲\n乙\n甲\n", encoding="utf-8")

    with pytest.raises(ValueError, match="字典字符重复"):
        load_keys(keys_path)


def test_resolve_annotation_paths_keeps_absolute(tmp_path: Path) -> None:
    absolute = tmp_path / "absolute.txt"

    resolved = resolve_annotation_paths(
        ["recognition/data/train_annotation.txt", str(absolute)],
        root=tmp_path,
    )

    assert resolved == (tmp_path / "recognition/data/train_annotation.txt", absolute)
