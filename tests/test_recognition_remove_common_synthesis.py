import json
from pathlib import Path

import pytest

from recognition.utils.character_stats import build_statistics, write_statistics
from recognition.utils import remove_common_synthesis as removal


@pytest.fixture
def dataset(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("A\nB\nr\ns\n", encoding="utf-8")
    texts = {
        "train": ["A" * 100 + "B" * 40 + "r" * 99], "val": ["B" * 60],
        "synthesis1": ["AB", "Ar", "ss", "?A"],
        "synthesis2": ["BA", "sA", "rA"],
    }
    annotations = []
    for name, labels in texts.items():
        directory = tmp_path / f"{name}_images"
        directory.mkdir()
        annotation = tmp_path / f"{name}_annotation.txt"
        lines = []
        for index, label in enumerate(labels):
            image = directory / f"{index}.png"
            image.write_bytes(f"image {name} {index}".encode())
            lines.append(f"{image.relative_to(tmp_path)}\t {label} \r\n")
        annotation.write_bytes("".join(lines).encode())
        annotations.append(annotation)
    rows, summary = build_statistics(keys, annotations)
    write_statistics(rows, summary, tmp_path / "stats")
    return tmp_path, tmp_path / "stats/character_stats.csv"


def args_for(dataset):
    root, csv_path = dataset
    return ["--csv", str(csv_path), "--data-dir", str(root),
            "--backup-dir", str(root / "backups")]


def test_all_rule_sums_train_val_includes_100_protects_99_and_unknown(dataset):
    root, csv_path = dataset
    plan = removal.build_plan(csv_path, root)
    assert set(plan.characters) == {"A", "B"}
    assert [row["label"] for row in plan.removed_rows] == ["AB", "BA"]
    assert set(plan.replacements) == {root / f"{name}_annotation.txt" for name in removal.TARGETS}
    assert "?A" in plan.replacements[root / "synthesis1_annotation.txt"]
    assert removal.removal_impact(plan, 100) == {
        "uncommon_occurrences": 0, "uncommon_characters": 0, "newly_below_threshold": 0,
    }


def test_any_rule_reports_loss_of_rare_and_unknown_characters(dataset):
    root, csv_path = dataset
    plan = removal.build_plan(csv_path, root, match="any")
    assert len(plan.removed_rows) == 6
    assert removal.removal_impact(plan, 100) == {
        "uncommon_occurrences": 4, "uncommon_characters": 3, "newly_below_threshold": 1,
    }


def test_preview_has_no_side_effects(dataset, capsys):
    root, _ = dataset
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert removal.main(args_for(dataset)) == 0
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = capsys.readouterr().out
    assert "共移除 2 条标注、2 张图片" in output
    assert "0 个字符、0 次出现" in output


def test_apply_removes_whole_samples_backs_up_and_preserves_keys_train_val(dataset):
    root, csv_path = dataset
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert removal.main([*args_for(dataset), "--apply"]) == 0
    changed = {root / f"{name}_annotation.txt" for name in removal.TARGETS}
    deleted = {root / f"{name}_images/0.png" for name in removal.TARGETS}
    for path, content in before.items():
        if path in deleted:
            assert not path.exists()
        elif path in changed:
            assert path.read_bytes() == b"".join(content.splitlines(keepends=True)[1:])
        else:
            assert path.read_bytes() == content
    manifest_path, = (root / "backups").glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["files"]) == 4
    for entry in manifest["files"]:
        assert Path(entry["backup"]).read_bytes() == before[Path(entry["original"])]
    assert (manifest_path.parent / "character_stats.csv").read_bytes() == before[csv_path]
    assert removal.main([*args_for(dataset), "--apply"]) == 1  # 旧 CSV 次数不再有效。


@pytest.mark.parametrize("name", removal.DATASETS)
def test_stale_statistics_stop_before_mutation(dataset, name):
    root, csv_path = dataset
    path = root / f"{name}_annotation.txt"
    path.write_text(path.read_text().replace("A", "s").replace("B", "s"))
    with pytest.raises(ValueError, match="CSV is stale"):
        removal.build_plan(csv_path, root)
    assert not (root / "backups").exists()


def test_missing_candidate_image_has_path_and_stops_before_apply(dataset):
    root, csv_path = dataset
    image = root / "synthesis1_images/0.png"
    image.unlink()
    with pytest.raises(FileNotFoundError, match="synthesis1_images/0.png"):
        removal.build_plan(csv_path, root)


@pytest.mark.parametrize("relative", ["../outside.png", "/tmp/outside.png", "synthesis1_images/0.jpg"])
def test_unsafe_candidate_paths_rejected(dataset, relative):
    root, csv_path = dataset
    path = root / "synthesis1_annotation.txt"
    path.write_text(path.read_text().replace("synthesis1_images/0.png", relative))
    with pytest.raises(ValueError, match="ordinary relative PNG"):
        removal.build_plan(csv_path, root)


def test_symlink_candidate_rejected(dataset):
    root, csv_path = dataset
    image = root / "synthesis1_images/0.png"
    image.unlink()
    image.symlink_to(root / "train_images/0.png")
    with pytest.raises(ValueError, match="ordinary relative PNG"):
        removal.build_plan(csv_path, root)


def test_retained_image_reference_prevents_deletion(dataset):
    root, csv_path = dataset
    train = root / "train_annotation.txt"
    train.write_text(train.read_text().replace("train_images/0.png", "synthesis1_images/0.png"))
    with pytest.raises(ValueError, match="still referenced"):
        removal.build_plan(csv_path, root)


def test_duplicate_deleted_images_are_deduplicated(dataset):
    root, csv_path = dataset
    path = root / "synthesis2_annotation.txt"
    path.write_text(path.read_text().replace("synthesis2_images/0.png", "synthesis1_images/0.png"))
    plan = removal.build_plan(csv_path, root)
    assert len(plan.removed_rows) == 2
    assert len(plan.images) == 1


def test_backup_failure_preserves_sources(dataset, monkeypatch):
    root, _ = dataset
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    def fail_copy(*args, **kwargs):
        raise OSError("backup failed")
    monkeypatch.setattr("recognition.utils.remove_characters_by_stats.shutil.copy2", fail_copy)
    assert removal.main([*args_for(dataset), "--apply"]) == 1
    for path, content in before.items():
        assert path.read_bytes() == content


def test_changes_after_planning_are_rejected(dataset):
    root, csv_path = dataset
    plan = removal.build_plan(csv_path, root)
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(ValueError, match="规划后文件发生变化"):
        removal.apply_plan(plan, csv_path, root / "backups")
    assert not (root / "backups").exists()


def test_no_matches_creates_no_backup(dataset):
    root, _ = dataset
    assert removal.main([*args_for(dataset), "--threshold", "1000", "--apply"]) == 0
    assert not (root / "backups").exists()
