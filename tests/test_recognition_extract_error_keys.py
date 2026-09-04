import subprocess
import sys
from pathlib import Path

from recognition.utils.extract_error_keys import extract_error_keys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "recognition"
    / "utils"
    / "extract_error_keys.py"
)


def test_extracts_unique_ground_truth_error_characters(tmp_path: Path) -> None:
    input_path = tmp_path / "error_cases.txt"
    output_path = tmp_path / "keys-error.txt"
    input_path.write_text(
        "Accuracy: 90.000%\n"
        "Avg CER: 10.000%\n\n"
        "CER\t文件名\t真实\t预测\n"
        "100.00\ta.png\t旗\t滨\n"
        "100.00\tb.png\t3\t\n"
        "100.00\tc.png\t：\t：：\n"
        "50.00\td.png\t甲乙\t甲丙\n"
        "50.00\te.png\t项目\t项 目\n",
        encoding="utf-8",
    )

    case_count, character_count = extract_error_keys(input_path, output_path)

    assert case_count == 5
    assert character_count == 4
    assert output_path.read_text(encoding="utf-8") == "3\n乙\n旗\n：\n"


def test_reports_line_number_for_malformed_case(tmp_path: Path) -> None:
    input_path = tmp_path / "error_cases.txt"
    output_path = tmp_path / "keys-error.txt"
    input_path.write_text(
        "CER\t文件名\t真实\t预测\n"
        "100.00\tmissing-fields.png\t真实\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert f"{input_path}:2" in result.stderr
    assert "四个制表符分隔字段" in result.stderr
    assert not output_path.exists()


def test_rejects_report_without_header(tmp_path: Path) -> None:
    input_path = tmp_path / "error_cases.txt"
    output_path = tmp_path / "keys-error.txt"
    input_path.write_text("Accuracy: 100.000%\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "缺少表头" in result.stderr
    assert not output_path.exists()
