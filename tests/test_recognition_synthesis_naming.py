from pathlib import Path

import pytest

from recognition.utils.synthesis import generate as plain
from recognition.utils.synthesis import generate_balanced as balanced
from recognition.utils.synthesis import generate_red_stamp as stamp
from recognition.utils.synthesis import naming


def test_numbered_batches_skip_images_annotations_partials_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(plain, "DATA", tmp_path)
    monkeypatch.setattr(plain, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(naming, "REPORT_ROOT", tmp_path / "reports")
    first = plain.parse_args([])
    assert first.batch_name == "synthesis1"
    assert first.output_dir == tmp_path / "synthesis1_images"
    first.output_dir.mkdir()
    (tmp_path / "synthesis2_annotation.txt").touch()
    (tmp_path / "synthesis3_annotation.txt.partial").touch()
    (tmp_path / "reports" / f"synthesis4_{first.date}").mkdir(parents=True)
    args = plain.parse_args([])
    assert args.batch_name == "synthesis5"
    assert args.annotation == tmp_path / "synthesis5_annotation.txt"
    assert args.report_dir == tmp_path / "reports" / f"synthesis5_{args.date}"


def test_explicit_numbered_paths_determine_filename_batch(tmp_path):
    name, images, annotation = naming.numbered_outputs(
        tmp_path, "20260908", output_dir=tmp_path / "synthesis9_images")
    assert name == "synthesis9"
    assert annotation == tmp_path / "synthesis9_annotation.txt"
    assert naming.numbered_filename(name, "20260908", 1) == "synthesis9_20260908_00001.png"
    assert naming.numbered_filename(name, "20260908", 100000) == "synthesis9_20260908_100000.png"
    with pytest.raises(ValueError, match="different synthesis batch"):
        naming.numbered_outputs(tmp_path, "20260908", images, tmp_path / "synthesis2_annotation.txt")


def test_balanced_and_red_defaults_follow_document(tmp_path, monkeypatch):
    monkeypatch.setattr(balanced, "DATA", tmp_path)
    monkeypatch.setattr(balanced, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(stamp, "REPORT_ROOT", tmp_path / "reports")
    args = balanced.parse_args([])
    assert args.output_dir == tmp_path / f"synthesis_balanced_{args.date}_images"
    assert args.annotation == tmp_path / f"synthesis_balanced_{args.date}_annotation.txt"
    assert args.report_dir == tmp_path / "reports" / f"synthesis_balanced_{args.date}"
    args = stamp.parse_args(["--data-dir", str(tmp_path)])
    assert args.output_dir == tmp_path / f"synthesis_red_{args.date}_images"
    assert args.annotation == tmp_path / f"synthesis_red_{args.date}_annotation.txt"
    assert args.report_dir == tmp_path / "reports" / f"synthesis_red_{args.date}"


@pytest.mark.parametrize("module", [plain, balanced, stamp])
def test_explicit_output_paths_are_preserved(tmp_path, module):
    args = module.parse_args(["--output-dir", str(tmp_path / "custom_images"),
                              "--annotation", str(tmp_path / "custom.txt"),
                              "--report-dir", str(tmp_path / "custom_report")])
    assert args.output_dir == tmp_path / "custom_images"
    assert args.annotation == tmp_path / "custom.txt"
    assert args.report_dir == tmp_path / "custom_report"
