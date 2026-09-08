import json
from pathlib import Path

import pytest

from recognition.utils import rename_synthesis_images as rename
from recognition.utils.annotation import split_annotation_fields


@pytest.fixture
def data(tmp_path):
    for batch in ("synthesis1", "synthesis2"):
        directory = tmp_path / f"{batch}_images"
        directory.mkdir()
        (directory / "old_1.png").write_bytes(b"first image " + batch.encode())
        (directory / "old_2.png").write_bytes(b"second image " + batch.encode())
        (tmp_path / f"{batch}_annotation.txt").write_bytes(
            f"{batch}_images/old_1.png\t 字,符 \r\n{batch}_images/old_2.png\t原文\n".encode())
    return tmp_path


def test_migration_preserves_image_bytes_labels_and_newlines_with_backup(data):
    before = {path: path.read_bytes() for path in data.rglob("*") if path.is_file()}
    plan = rename.build_plan(data)
    assert len(plan.files) == 4
    assert before == {path: path.read_bytes() for path in data.rglob("*") if path.is_file()}
    backup = rename.apply_plan(plan, data / "backups")
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for row in manifest["files"]:
        assert not Path(row["source"]).exists()
        assert Path(row["destination"]).read_bytes() == before[Path(row["source"])]
    for batch in ("synthesis1", "synthesis2"):
        annotation = data / f"{batch}_annotation.txt"
        assert (backup / annotation.name).read_bytes() == before[annotation]
        with annotation.open(encoding="utf-8", newline="") as handle:
            lines = handle.readlines()
        old = before[annotation].decode().splitlines(keepends=True)
        for number, (line, original) in enumerate(zip(lines, old), 1):
            relative, _ = split_annotation_fields(line, annotation, number)
            assert Path(relative).name == f"{batch}_{plan.date}_{number:05d}.png"
            assert line.partition("\t")[2] == original.partition("\t")[2]
    repeated = rename.build_plan(data)
    assert not repeated.files and not repeated.replacements
    assert rename.apply_plan(repeated, data / "backups") is None


def test_missing_source_and_existing_destination_fail_before_changes(data):
    plan = rename.build_plan(data)
    destination = Path(plan.files[0]["destination"])
    destination.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="destination already exists"):
        rename.build_plan(data)
    assert destination.read_bytes() == b"do not overwrite"
    destination.unlink()
    Path(plan.files[0]["source"]).unlink()
    with pytest.raises(FileNotFoundError, match="Image does not exist"):
        rename.build_plan(data)


def test_other_annotation_reference_blocks_migration(data):
    (data / "train_annotation.txt").write_text("synthesis1_images/old_1.png\t字\n")
    with pytest.raises(ValueError, match="another annotation"):
        rename.build_plan(data)


@pytest.mark.parametrize("change_image", [False, True])
def test_changes_after_plan_are_rejected(data, change_image):
    plan = rename.build_plan(data)
    path = Path(plan.files[0]["source"]) if change_image else next(iter(plan.originals))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after planning"):
        rename.apply_plan(plan, data / "backups")
    assert not (data / "backups").exists()


def test_link_failure_leaves_originals_intact(data, monkeypatch):
    plan = rename.build_plan(data)
    real_link = rename.os.link
    count = 0
    def fail_second(source, destination):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("link failed")
        real_link(source, destination)
    monkeypatch.setattr(rename.os, "link", fail_second)
    with pytest.raises(OSError, match="link failed"):
        rename.apply_plan(plan, data / "backups")
    for path, original in plan.originals.items():
        assert path.read_bytes() == original
    for row in plan.files:
        assert rename.digest(Path(row["source"])) == row["sha256"]
        assert not Path(row["destination"]).exists()


def test_second_annotation_failure_restores_first_and_all_old_names(data, monkeypatch):
    plan = rename.build_plan(data)
    real_write = rename.atomic_write
    count = 0
    def fail_second(path, text):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("annotation write failed")
        real_write(path, text)
    monkeypatch.setattr(rename, "atomic_write", fail_second)
    with pytest.raises(OSError, match="annotation write failed"):
        rename.apply_plan(plan, data / "backups")
    for path, original in plan.originals.items():
        assert path.read_bytes() == original
    for row in plan.files:
        assert rename.digest(Path(row["source"])) == row["sha256"]
        assert not Path(row["destination"]).exists()


def test_symlink_source_is_rejected(data):
    image = data / "synthesis1_images/old_1.png"
    image.unlink()
    image.symlink_to(data / "synthesis2_images/old_1.png")
    with pytest.raises(ValueError, match="ordinary relative PNG"):
        rename.build_plan(data)
