import json
from pathlib import Path

import pytest

from recognition.utils.character_stats import build_statistics, write_statistics
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
        "append": ["保稀", "独" * 12, "界" * 5, "保 保"],
        "bench1": ["稀" * 8, "界" * 5], "bench2": ["保"],
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


@pytest.mark.parametrize("mode, expected", [
    ("1", {"稀", "零"}), ("2", {"独"}), ("both", {"稀", "独", "零"}),
])
def test_modes_use_occurrences_strict_threshold_and_dataset_presence(tmp_path, mode, expected):
    keys, csv_path = make_dataset(tmp_path)
    plan = build_plan(csv_path, keys, tmp_path, mode)
    assert set(plan.characters) == expected
    assert "界" not in plan.characters  # 5+5=10，不符合 <10 或单一来源。
    assert "保" not in plan.characters  # train 保护。
    assert "验" not in plan.characters  # val 保护。


def test_overlapping_rules_record_both_reasons():
    counts = {"甲": dict(zip(DATASETS, [0, 0, 3, 0, 0]))}
    assert select_characters(counts, "both") == {"甲": ("1", "2")}


def test_default_preview_writes_nothing(tmp_path, capsys):
    keys, csv_path = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(cli_args(tmp_path, keys, csv_path)) == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not (tmp_path / "backups").exists()
    assert "共移除 3 条标注、3 张图片" in capsys.readouterr().out


def test_apply_deletes_whole_rows_and_images_preserves_other_bytes_and_backs_up(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 0
    deleted = {tmp_path / p for p in (
        "append_images/0.png", "append_images/1.png", "bench1_images/0.png",
    )}
    for path in deleted:
        assert not path.exists()
    assert keys.read_bytes() == "保\r\n验\r\n界\r\n \r\n".encode()
    for name in ("train", "val", "bench2"):
        path = tmp_path / f"{name}_annotation.txt"
        assert path.read_bytes() == before[path]
    append = tmp_path / "append_annotation.txt"
    assert append.read_bytes() == b"".join(before[append].splitlines(keepends=True)[2:])
    for path, content in before.items():
        if path.suffix == ".png" and path not in deleted:
            assert path.read_bytes() == content
    manifest_path, = (tmp_path / "backups").glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        assert Path(entry["backup"]).read_bytes() == before[Path(entry["original"])]
    assert len(manifest["files"]) == 6  # 两份标注、keys、三张图片。
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
    image = tmp_path / "append_images/0.png"
    image.unlink()
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 1
    assert str(image) in capsys.readouterr().err
    assert not (tmp_path / "backups").exists()
    assert "稀" in keys.read_text()


def test_image_referenced_by_retained_row_is_protected(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    train = tmp_path / "train_annotation.txt"
    train.write_text("append_images/0.png\t保 保\n", encoding="utf-8")
    with pytest.raises(ValueError, match="仍被 train/val 或保留标注引用"):
        build_plan(csv_path, keys, tmp_path)


def test_duplicate_removed_image_is_only_deleted_once(tmp_path):
    keys, csv_path = make_dataset(tmp_path)
    annotation = tmp_path / "bench1_annotation.txt"
    annotation.write_text(annotation.read_text().replace("bench1_images/0.png", "append_images/0.png"))
    plan = build_plan(csv_path, keys, tmp_path)
    assert len(plan.removed_rows) == 3
    assert len(plan.images) == 2
    assert main([*cli_args(tmp_path, keys, csv_path), "--apply"]) == 0
    assert not (tmp_path / "append_images/0.png").exists()


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
    image = tmp_path / "append_images/0.png"
    image.unlink()
    image.symlink_to(tmp_path / "train_images/0.png")
    with pytest.raises(ValueError, match="普通相对 PNG 路径"):
        build_plan(csv_path, keys, tmp_path)


@pytest.mark.parametrize("relative", ["../outside.png", "/tmp/outside.png", "append_images/0.jpg"])
def test_unsafe_candidate_paths_rejected(tmp_path, relative):
    keys, csv_path = make_dataset(tmp_path)
    annotation = tmp_path / "append_annotation.txt"
    annotation.write_text(annotation.read_text().replace("append_images/0.png", relative))
    with pytest.raises(ValueError, match="普通相对 PNG 路径"):
        build_plan(csv_path, keys, tmp_path)


@pytest.mark.parametrize("contents", [
    "字符,append_次数\n甲,1\n",
    "字符,train_次数,val_次数,append_次数,bench1_次数,bench2_次数\n甲,0,0,-1,0,0\n",
    "字符,train_次数,val_次数,append_次数,bench1_次数,bench2_次数\n甲,0,0,1.5,0,0\n",
    "字符,train_次数,val_次数,append_次数,bench1_次数,bench2_次数\n甲,0,0,1,0,0\n甲,0,0,1,0,0\n",
])
def test_invalid_csv_rejected(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=str(path)):
        load_statistics(path)
