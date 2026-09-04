import subprocess
import sys
from pathlib import Path

from recognition.utils.compare_key_files import write_append_keys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "recognition"
    / "utils"
    / "compare_key_files.py"
)


def test_writes_only_new_characters_in_new_keys_order(tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.txt"
    new_keys_path = tmp_path / "keys-new.txt"
    output_path = tmp_path / "keys-append.txt"
    keys_path.write_text("a\n \n中\n", encoding="utf-8")
    new_keys_path.write_text("中\n新\n \n新\n🙂\n", encoding="utf-8-sig")

    append_count = write_append_keys(keys_path, new_keys_path, output_path)

    assert append_count == 2
    assert output_path.read_text(encoding="utf-8") == "新\n🙂\n"


def test_cli_reports_missing_input_without_creating_output(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    new_keys_path = tmp_path / "keys-new.txt"
    output_path = tmp_path / "keys-append.txt"
    new_keys_path.write_text("新\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--keys",
            str(missing_path),
            "--keys-new",
            str(new_keys_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert str(missing_path) in result.stderr
    assert not output_path.exists()
