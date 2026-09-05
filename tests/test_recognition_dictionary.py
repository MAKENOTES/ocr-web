import subprocess
import sys
from pathlib import Path

from recognition.utils.dictionary import extract_chars_to_dict


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "recognition"
    / "utils"
    / "dictionary.py"
)


def test_extracts_complete_text_with_commas_and_internal_quotes(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "train_annotation.txt"
    output_path = tmp_path / "keys.txt"
    annotation_path.write_text(
        'train_images/a.png\t以一组"正常"细胞作为参考，分析\n'
        "train_images/b.png\t字母,a\n",
        encoding="utf-8",
    )

    char_count = extract_chars_to_dict(annotation_path, output_path)

    keys = output_path.read_text(encoding="utf-8").splitlines()
    expected = sorted(set('以一组"正常"细胞作为参考，分析字母,a'))
    assert keys == expected
    assert char_count == len(expected)
    assert '"' in keys


def test_reports_annotation_path_and_line_for_malformed_input(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "train_annotation.txt"
    output_path = tmp_path / "keys.txt"
    annotation_path.write_text(
        "train_images/a.png\t正常\n第二行没有分隔符\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(annotation_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert f"{annotation_path}:2" in result.stderr
    assert "缺少路径与文字之间的 Tab 制表符" in result.stderr
    assert not output_path.exists()


def test_accepts_utf8_bom(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train_annotation.txt"
    output_path = tmp_path / "keys.txt"
    annotation_path.write_text(
        "train_images/a.png\t甲乙\n",
        encoding="utf-8-sig",
    )

    extract_chars_to_dict(annotation_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "乙\n甲\n"
