import subprocess
import sys
from pathlib import Path

import pytest

import recognition.utils.check_val_annotation_keys as check_module


def test_script_can_be_run_directly() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(check_module.__file__).resolve()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "列出验证标注中包含 keys.txt 未收录字符的图片" in result.stdout


def test_default_paths_are_relative_to_recognition_directory() -> None:
    recognition_dir = Path(check_module.__file__).resolve().parents[1]

    assert check_module.DEFAULT_ANNOTATION == (
        recognition_dir / "data" / "val_annotation.txt"
    )
    assert check_module.DEFAULT_KEYS == recognition_dir / "keys.txt"
    assert check_module.OUTPUT_PATH == (
        recognition_dir / "data" / "val_unknown_keys.txt"
    )


def run_check(
    annotation_path: Path,
    keys_path: Path,
    output_path: Path,
    monkeypatch,
    *,
    move_unknown: bool = False,
    unknown_images_dir: Path | None = None,
    unknown_annotation_path: Path | None = None,
) -> int:
    monkeypatch.setattr(check_module, "OUTPUT_PATH", output_path)
    if unknown_images_dir is not None:
        monkeypatch.setattr(
            check_module,
            "UNKNOWN_IMAGES_DIR",
            unknown_images_dir,
        )
    if unknown_annotation_path is not None:
        monkeypatch.setattr(
            check_module,
            "UNKNOWN_ANNOTATION",
            unknown_annotation_path,
        )
    arguments = [
        str(check_module.__file__),
        "--annotation",
        str(annotation_path),
        "--keys",
        str(keys_path),
    ]
    if move_unknown:
        arguments.append("--move-unknown")
    monkeypatch.setattr(
        sys,
        "argv",
        arguments,
    )
    return check_module.main()


def test_writes_images_text_and_unknown_characters(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    annotation_path = tmp_path / "val_annotation.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = tmp_path / "val_unknown_keys.txt"
    annotation_path.write_text(
        "val_images/a.png\t甲乙,丙\n"
        "val_images/b.png\t甲丁\n"
        "val_images/c.png\t甲乙\n",
        encoding="utf-8",
    )
    keys_path.write_text("甲\n乙\n", encoding="utf-8")

    return_code = run_check(annotation_path, keys_path, output_path, monkeypatch)
    output = output_path.read_text(encoding="utf-8")
    console = capsys.readouterr()

    assert return_code == 1
    assert "含字典外字符的图片数: 2" in output
    assert "字典外字符种类数: 3" in output
    assert "图片: val_images/a.png (第 1 行)" in output
    assert "文字: '甲乙,丙'" in output
    assert "字典未收录: ',丙'" in output
    assert "图片: val_images/b.png (第 2 行)" in output
    assert "字典未收录: '丁'" in output
    assert "val_images/c.png" not in output
    assert f"结果已写入: {output_path}" in console.out


def test_passes_when_all_characters_are_in_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    annotation_path = tmp_path / "val_annotation.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = tmp_path / "val_unknown_keys.txt"
    annotation_path.write_text(
        "val_images/a.png\t甲 乙\n",
        encoding="utf-8",
    )
    keys_path.write_text(" \n乙\n甲\n", encoding="utf-8")

    return_code = run_check(annotation_path, keys_path, output_path, monkeypatch)
    output = output_path.read_text(encoding="utf-8")

    assert return_code == 0
    assert "含字典外字符的图片数: 0" in output
    assert "核对通过" in output


def test_moves_unknown_images_and_annotations_out_of_val(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_dir = tmp_path / "data"
    val_images_dir = data_dir / "val_images"
    annotation_path = data_dir / "val_annotation.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = data_dir / "val_unknown_keys.txt"
    unknown_images_dir = data_dir / "val_unknown"
    unknown_annotation_path = data_dir / "val_unknown.txt"
    val_images_dir.mkdir(parents=True)
    annotation_path.write_text(
        "val_images/a.png\t甲乙,丙\n"
        "val_images/b.png\t甲丁\n"
        "val_images/c.png\t甲乙\n",
        encoding="utf-8",
    )
    keys_path.write_text("甲\n乙\n", encoding="utf-8")
    for filename in ("a.png", "b.png", "c.png"):
        (val_images_dir / filename).write_bytes(filename.encode("ascii"))

    return_code = run_check(
        annotation_path,
        keys_path,
        output_path,
        monkeypatch,
        move_unknown=True,
        unknown_images_dir=unknown_images_dir,
        unknown_annotation_path=unknown_annotation_path,
    )
    console = capsys.readouterr()

    assert return_code == 0, console.err
    assert not (val_images_dir / "a.png").exists()
    assert not (val_images_dir / "b.png").exists()
    assert (val_images_dir / "c.png").is_file()
    assert (unknown_images_dir / "a.png").read_bytes() == b"a.png"
    assert (unknown_images_dir / "b.png").read_bytes() == b"b.png"
    assert annotation_path.read_text(encoding="utf-8") == (
        "val_images/c.png\t甲乙\n"
    )
    assert unknown_annotation_path.read_text(encoding="utf-8") == (
        "val_unknown/a.png\t甲乙,丙\n"
        "val_unknown/b.png\t甲丁\n"
    )
    assert "已转移图片和标注: 2 条" in console.out


def test_move_refuses_nonempty_destination_without_changing_source(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_dir = tmp_path / "data"
    val_images_dir = data_dir / "val_images"
    annotation_path = data_dir / "val_annotation.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = data_dir / "val_unknown_keys.txt"
    unknown_images_dir = data_dir / "val_unknown"
    unknown_annotation_path = data_dir / "val_unknown.txt"
    val_images_dir.mkdir(parents=True)
    unknown_images_dir.mkdir()
    (unknown_images_dir / "existing.png").write_bytes(b"existing")
    source_image = val_images_dir / "a.png"
    source_image.write_bytes(b"source")
    original_annotation = "val_images/a.png\t甲丁\n"
    annotation_path.write_text(original_annotation, encoding="utf-8")
    keys_path.write_text("甲\n", encoding="utf-8")

    return_code = run_check(
        annotation_path,
        keys_path,
        output_path,
        monkeypatch,
        move_unknown=True,
        unknown_images_dir=unknown_images_dir,
        unknown_annotation_path=unknown_annotation_path,
    )
    console = capsys.readouterr()

    assert return_code == 2
    assert "转移目标目录不是空目录" in console.err
    assert source_image.read_bytes() == b"source"
    assert annotation_path.read_text(encoding="utf-8") == original_annotation
    assert not unknown_annotation_path.exists()


def test_move_rolls_back_images_and_annotations_on_write_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    val_images_dir = data_dir / "val_images"
    annotation_path = data_dir / "val_annotation.txt"
    unknown_images_dir = data_dir / "val_unknown"
    unknown_annotation_path = data_dir / "val_unknown.txt"
    val_images_dir.mkdir(parents=True)
    source_image = val_images_dir / "a.png"
    source_image.write_bytes(b"source")
    original_annotation = "val_images/a.png\t甲丁\n"
    annotation_path.write_text(original_annotation, encoding="utf-8")
    _, unknown_records = check_module.find_unknown_character_records(
        annotation_path,
        {"甲"},
    )

    original_atomic_write = check_module._atomic_write_bytes
    call_count = 0

    def fail_second_write(path: Path, content: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("模拟写入失败")
        original_atomic_write(path, content)

    monkeypatch.setattr(check_module, "_atomic_write_bytes", fail_second_write)

    with pytest.raises(RuntimeError, match="已恢复原数据"):
        check_module.move_unknown_records(
            annotation_path,
            unknown_records,
            unknown_images_dir,
            unknown_annotation_path,
        )

    assert source_image.read_bytes() == b"source"
    assert annotation_path.read_text(encoding="utf-8") == original_annotation
    assert not unknown_annotation_path.exists()
    assert not unknown_images_dir.exists()


def test_rejects_multi_character_key_line(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    annotation_path = tmp_path / "val_annotation.txt"
    keys_path = tmp_path / "keys.txt"
    output_path = tmp_path / "val_unknown_keys.txt"
    annotation_path.write_text(
        "val_images/a.png\t甲乙\n",
        encoding="utf-8",
    )
    keys_path.write_text("甲乙\n", encoding="utf-8")

    return_code = run_check(annotation_path, keys_path, output_path, monkeypatch)
    console = capsys.readouterr()

    assert return_code == 2
    assert f"{keys_path}:1" in console.err
    assert "每个非空行必须只有一个字符" in console.err
    assert not output_path.exists()
