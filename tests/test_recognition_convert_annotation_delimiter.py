from pathlib import Path

import pytest

from recognition.utils.convert_annotation_delimiter import atomic_write, convert_content


def test_converts_only_first_comma_and_preserves_text_commas() -> None:
    path = Path("annotations.txt")
    content = 'images/a.png, "甲,乙，丙"\nimages/b.png,"文字"\n'

    converted, count = convert_content(content, path)

    assert converted == 'images/a.png\t"甲,乙，丙"\nimages/b.png\t"文字"\n'
    assert count == 2


def test_is_idempotent_for_tab_separated_annotations() -> None:
    path = Path("annotations.txt")
    content = 'images/a.png\t"甲,乙"\n'

    converted, count = convert_content(content, path)

    assert converted == content
    assert count == 0


def test_legacy_separator_before_text_tab_is_converted() -> None:
    path = Path("annotations.txt")

    converted, count = convert_content('images/a.png, "甲\t乙"\n', path)

    assert converted == 'images/a.png\t"甲\t乙"\n'
    assert count == 1


def test_reports_malformed_line_with_path_and_line_number() -> None:
    path = Path("annotations.txt")

    with pytest.raises(ValueError, match=r"annotations\.txt:2"):
        convert_content('images/a.png\t"甲"\ninvalid line\n', path)


def test_atomic_write_preserves_existing_file_permissions(tmp_path: Path) -> None:
    path = tmp_path / "annotations.txt"
    path.write_text('images/a.png, "甲"\n', encoding="utf-8")
    path.chmod(0o640)

    atomic_write(path, 'images/a.png\t"甲"\n')

    assert path.stat().st_mode & 0o777 == 0o640
