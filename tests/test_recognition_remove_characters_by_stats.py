import json
from pathlib import Path

import pytest

from recognition.utils.character_stats import DEFAULT_ANNOTATIONS, build_statistics, write_statistics
from recognition.utils.remove_characters_by_stats import (
    DATASETS,
    build_plan,
    load_statistics,
    main,
    select_characters,
)


def make_dataset(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_bytes("保\r\n验\r\n稀\r\n独\r\n界\r\n零\r\n \r\n".encode())
    labels = {
        "train": ["保 保"], "val": ["验"],
        "synthesis1": ["保稀", "界" * 10, "保 保", "稀" * 8],
        "synthesis2": ["独" * 12, "保 保"],
    }
    paths = []
    for name, texts in labels.items():
        (tmp_path / f"{name}_images").mkdir()
        annotation = tmp_path / f"{name}_annotation.txt"
        lines = []
        for index, label in enumerate(texts):
            relative = f"{name}_images/{index}.png"
            (tmp_path / relative).write_bytes(b"test image")
            lines.append(f"{relative}\t {label} \r\n")
        annotation.write_bytes("".join(lines).encode())
        paths.append(annotation)
    rows, summary = build_statistics(keys, paths)
    write_statistics(rows, summary, tmp_path / "stats")
    csv_path = tmp_path / "stats" / "character_stats.csv"
    return keys, csv_path


def cli_args(tmp_path, keys, csv_path):
    return ["--keys", str(keys), "--csv", str(csv_path), "--data-dir", str(tmp_path),
            "--backup-dir", str(tmp_path / "backups")]


def test_removal_datasets_match_default_statistics():
    assert DATASETS == ("train", "val", "synthesis1", "synthesis2")
    assert tuple(path.stem.removesuffix("_annotation") for path in DEFAULT_ANNOTATIONS) == DATASETS


@pytest.mark.parametrize("mode, expected", [
    ("1", {"稀", "零"}), ("2", {"稀", "独", "界"}), ("both", {"稀", "独", "界", "零"}),
])
def test_modes_use_occurrences_strict_threshold_and_dataset_presence(tmp_path, mode, expected):
    keys, csv_path = make_dataset(tmp_path)
    plan = build_plan(csv_path, keys, tmp_path, mode)
    assert set(plan.characters) == expected
    assert ("界" in plan.characters) == (mode != "1")  # 10 次不符合 <10，但符合单一来源。
    assert "保" not in plan.characters  # train 保护。
    assert "验" not in plan.characters  # val 保护。


@pytest.mark.parametrize("first_count, second_count, expected", [
    (3, 0, {"甲": ("1", "2")}),
    (0, 3, {"甲": ("1", "2")}),
    (4, 5, {"甲": ("1",)}),
    (5, 5, {}),
    (6, 6, {}),
    (0, 0, {"甲": ("1",)}),
])
def test_rules_use_combined_count_and_distinct_synthesis_sources(first_count, second_count, expected):
    # 两份合计不足 10 次才符合方式 1；两份都有出现时不符合方式 2。
    counts = {"甲": {
        "train": 0, "val": 0, "synthesis1": first_count, "synthesis2": second_count,
    }}
    assert select_characters(counts, "both") == expected


def test_requested_character_only_removes_its_samples_and_key(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    assert main([*cli_args(tmp_path, keys, csv_path), "--chars", "独", "--apply"]) == 0
    assert "独" not in keys.read_text()
    assert "稀" in keys.read_text()
    assert "零" in keys.read_text()
    assert not (tmp_path / "synthesis2_images/0.png").exists()
    assert (tmp_path / "synthesis1_images/0.png").is_file()
    assert (tmp_path / "synthesis1_images/3.png").is_file()


@pytest.mark.parametrize("chars", ["保", "验", "外", ""])
def test_requested_character_cannot_bypass_removal_conditions(tmp_path, chars):
    keys, csv_path = make_dataset(tmp_path)
    assert main([*cli_args(tmp_path, keys, csv_path), "--chars", chars, "--apply"]) == 1
    assert not (tmp_path / "backups").exists()


def test_mode_one_rejects_character_at_threshold(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    assert main([*cli_args(tmp_path, keys, csv_path), "--mode", "1",
                 "--chars", "界", "--apply"]) == 1
    assert not (tmp_path / "backups").exists()


def test_default_preview_writes_nothing(tmp_path, capsys):
    keys, csv_path = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(cli_args(tmp_path, keys, csv_path)) == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not (tmp_path / "backups").exists()
    stdout = capsys.readouterr().out
    assert "共移除 4 条标注、4 张图片" in stdout
    assert "字符\t方式\tsynthesis1\tsynthesis2\t目标合计" in stdout
    assert "append" not in stdout
    assert "bench" not in stdout


def test_apply_deletes_whole_rows_and_images_preserves_other_bytes_and_backs_up(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 0
    deleted = {tmp_path / p for p in (
        "synthesis1_images/0.png", "synthesis1_images/1.png",
        "synthesis1_images/3.png", "synthesis2_images/0.png",
    )}
    for path in deleted:
        assert not path.exists()
    assert keys.read_bytes() == "保\r\n验\r\n \r\n".encode()
    for name in ("train", "val"):
        path = tmp_path / f"{name}_annotation.txt"
        assert path.read_bytes() == before[path]
    for name, retained_index in (("synthesis1", 2), ("synthesis2", 1)):
        annotation = tmp_path / f"{name}_annotation.txt"
        assert annotation.read_bytes() == before[annotation].splitlines(keepends=True)[retained_index]
    for path, content in before.items():
        if path.suffix == ".png" and path not in deleted:
            assert path.read_bytes() == content
    manifest_path, = (tmp_path / "backups").glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        assert Path(entry["backup"]).read_bytes() == before[Path(entry["original"])]
    assert len(manifest["files"]) == 7  # 两份标注、keys、四张图片。
    assert csv_path.read_bytes() == before[csv_path]
    # 执行后旧 CSV 必须重算，防止重复使用已经过期的候选字符表。
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 1


def test_stale_csv_rejected_before_mutation(tmp_path, capsys):
    keys, csv_path = make_dataset(tmp_path)
    train = tmp_path / "train_annotation.txt"
    train.write_text("train_images/0.png\t保稀\n", encoding="utf-8")
    before = keys.read_bytes()
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 1
    assert "CSV 已过期" in capsys.readouterr().err
    assert keys.read_bytes() == before
    assert not (tmp_path / "backups").exists()


def test_missing_image_fails_before_any_write(tmp_path, capsys):
    keys, csv_path = make_dataset(tmp_path)
    image = tmp_path / "synthesis1_images/0.png"
    image.unlink()
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 1
    assert str(image) in capsys.readouterr().err
    assert not (tmp_path / "backups").exists()
    assert "稀" in keys.read_text()


def test_image_referenced_by_retained_row_is_protected(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    train = tmp_path / "train_annotation.txt"
    train.write_text("synthesis1_images/0.png\t保 保\n", encoding="utf-8")
    with pytest.raises(ValueError, match="仍被 train/val 或保留标注引用"):
        build_plan(csv_path, keys, tmp_path)


def test_duplicate_removed_image_is_only_deleted_once(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    annotation = tmp_path / "synthesis1_annotation.txt"
    annotation.write_text(annotation.read_text().replace("synthesis1_images/3.png", "synthesis1_images/0.png"))
    plan = build_plan(csv_path, keys, tmp_path)
    assert len(plan.removed_rows) == 4
    assert len(plan.images) == 3
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 0
    assert not (tmp_path / "synthesis1_images/0.png").exists()


def test_backup_failure_leaves_sources_untouched(tmp_path, monkeypatch):
    keys, csv_path = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def fail_copy(*args, **kwargs):
        raise OSError("backup failed")

    monkeypatch.setattr("recognition.utils.remove_characters_by_stats.shutil.copy2", fail_copy)
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 1
    for path, content in before.items():
        assert path.read_bytes() == content


def test_symlink_candidate_is_rejected(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    image = tmp_path / "synthesis1_images/0.png"
    image.unlink()
    image.symlink_to(tmp_path / "train_images/0.png")
    with pytest.raises(ValueError, match="普通相对 PNG 路径"):
        build_plan(csv_path, keys, tmp_path)


@pytest.mark.parametrize("relative", ["../outside.png", "/tmp/outside.png", "synthesis1_images/0.jpg"])
def test_unsafe_candidate_paths_rejected(tmp_path, relative):
    keys, csv_path = make_dataset(tmp_path)
    annotation = tmp_path / "synthesis1_annotation.txt"
    annotation.write_text(annotation.read_text().replace("synthesis1_images/0.png", relative))
    with pytest.raises(ValueError, match="普通相对 PNG 路径"):
        build_plan(csv_path, keys, tmp_path)


@pytest.mark.parametrize("contents, error", [
    ("字符,append_次数\n甲,1\n", "CSV 缺少必要列"),
    ("字符,train_次数,val_次数,append_次数,bench1_次数,bench2_次数\n甲,0,0,1,0,0\n", "CSV 缺少必要列"),
    ("字符,train_次数,val_次数,synthesis_次数\n甲,0,0,1\n", "CSV 缺少必要列"),
    ("字符,train_次数,val_次数,synthesis1_次数\n甲,0,0,1\n", "CSV 缺少必要列"),
    ("字符,train_次数,val_次数,synthesis2_次数\n甲,0,0,1\n", "CSV 缺少必要列"),
    ("字符,train_次数,val_次数,synthesis1_次数,synthesis2_次数\n甲,0,0,-1,0\n", "CSV 次数必须是非负整数"),
    ("字符,train_次数,val_次数,synthesis1_次数,synthesis2_次数\n甲,0,0,0,1.5\n", "CSV 次数必须是非负整数"),
    ("字符,train_次数,val_次数,synthesis1_次数,synthesis2_次数\n甲,0,0,1,0\n甲,0,0,1,0\n", "CSV 字符无效或重复"),
])
def test_invalid_csv_rejected(tmp_path, contents, error):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=error) as exc_info:
        load_statistics(path)
    assert str(path) in str(exc_info.value)
