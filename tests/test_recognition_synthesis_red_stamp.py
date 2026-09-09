import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from recognition.utils.annotation import split_annotation_fields
from recognition.utils.synthesis import generate_red_stamp as stamp


@pytest.fixture
def setup(tmp_path):
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("DejaVuSans font unavailable")
    image = Image.new("RGB", (300, 36), "white")
    draw = ImageDraw.Draw(image)
    draw.text((3, 1), "ABab ABab ABab", font=ImageFont.truetype(str(font), 26), fill=(40, 40, 40))
    image.save(tmp_path / "gray.png")
    draw.ellipse((85, -30, 200, 85), outline=(210, 35, 55), width=4)
    image.save(tmp_path / "red.png")
    (tmp_path / "references.txt").write_text("red.png\ngray.png\n")
    (tmp_path / "keys.txt").write_text("A\nB\na\nb\n \n")
    (tmp_path / "train_annotation.txt").write_text("unused.png\tABab ABab ABab\n")
    args = stamp.parse_args([
        "--data-dir", str(tmp_path), "--references", str(tmp_path / "references.txt"),
        "--keys", str(tmp_path / "keys.txt"), "--fonts", str(font), "--stamp-text", "ABABABAB",
        "--images", "6", "--output-dir", str(tmp_path / "images"),
        "--annotation", str(tmp_path / "labels.txt"), "--report-dir", str(tmp_path / "report"),
    ])
    return args


def test_reference_manifest_contains_all_user_samples():
    manifest = Path(stamp.__file__).with_name("red_stamp_references.txt")
    entries = [line for line in manifest.read_text().splitlines() if line and not line.startswith("#")]
    assert len(entries) == len(set(entries)) == 161
    assert sum(line.startswith("val_images/") for line in entries) == 128
    assert sum(line.startswith("train_images/") for line in entries) == 33
    document = stamp.ROOT / "recognition/datadoc/analysisdata/errdata20260909.md"
    red_section = document.read_text().split("## 红章叠印（99）", 1)[1].split("\n## ", 1)[0]
    assert set(re.findall(r"\]\(../../data/([^)]*\.png)\)", red_section)) <= set(entries)
    assert "train_images/000480-000006.png" in entries
    assert "val_images/000584-000042.png" in entries
    assert "val_images/000872-000022.png" not in entries


def test_reference_styles_keep_statistics_and_handle_gray_reference(setup):
    args = setup
    paths = stamp.reference_paths(args.references, args.data_dir)
    styles = stamp.measure_styles(paths)
    assert styles[0].red_measured
    assert not styles[1].red_measured
    assert styles[0].color == styles[1].color
    assert styles[0].red_fraction > 0
    assert max(styles[0].body.foreground) - min(styles[0].body.foreground) < 10
    assert not any(isinstance(value, Image.Image) for value in vars(styles[0]).values())


def test_all_gray_references_fail_with_paths(setup):
    with pytest.raises(ValueError, match="gray.png"):
        stamp.measure_styles([setup.data_dir / "gray.png"])


def test_missing_reference_fails_with_path(setup):
    setup.references.write_text("missing.png\n")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        stamp.generate(setup)
    assert not setup.output_dir.exists()


def test_stamp_overlaps_body_preserves_dimensions_and_is_reproducible(setup):
    pytest.importorskip("text_renderer")
    args = setup
    style = stamp.measure_style(args.data_dir / "red.png")
    face = stamp.load_fonts(args.fonts, list("ABab "))[0]
    clean = stamp.render_image("ABab ABab ABab", face, style.body, random.Random(1))
    output, metadata = stamp.overlay_stamp(clean, style, face, args.stamp_text, random.Random(2))
    again, second = stamp.overlay_stamp(clean, style, face, args.stamp_text, random.Random(2))
    assert output.mode == "RGB" and output.size == clean.size
    assert metadata["body_overlap_fraction"] >= 0.01
    assert metadata["red_fraction"] > 0
    assert metadata == second and output.tobytes() == again.tobytes()
    assert stamp.red_mask(np.asarray(clean)).sum() == 0


def test_generation_labels_only_body_rgb_date_names_and_reports(setup):
    pytest.importorskip("text_renderer")
    args = setup
    args.images = len(stamp.SCENES) * 2
    source_before = args.corpus.read_bytes()
    report = stamp.generate(args)
    assert report["images"] == args.images
    assert args.corpus.read_bytes() == source_before
    assert len(list(args.output_dir.glob("*.png"))) == args.images
    assert set(report["references"].values()) == {args.images // 2}
    assert report["scenes"] == dict.fromkeys(stamp.SCENES, 2)
    for number, line in enumerate(args.annotation.read_text().splitlines(), 1):
        relative, text = split_annotation_fields(line, args.annotation, number)
        assert text == "ABab ABab ABab"
        assert args.stamp_text not in text
        assert Path(relative).name == f"{args.date}_{number:05d}.png"
        assert not Path(relative).is_absolute()
        with Image.open(args.annotation.parent / relative) as image:
            assert image.mode == "RGB"
            assert 16 <= image.height <= 64
            assert stamp.red_mask(np.asarray(image)).sum() >= 8
    records = [json.loads(line) for line in (args.report_dir / "samples.jsonl").read_text().splitlines()]
    assert len(records) == args.images
    assert all(row["stamp"]["body_overlap_fraction"] >= 0.002 for row in records)
    assert (args.report_dir / "preview.png").is_file()
    assert (args.report_dir / "summary.json").is_file()
    assert not args.annotation.with_name("labels.txt.partial").exists()
    before = args.annotation.read_bytes()
    with pytest.raises(FileExistsError):
        stamp.generate(args)
    assert args.annotation.read_bytes() == before


def test_stamp_still_overlaps_when_random_positions_miss_body(setup):
    pytest.importorskip("text_renderer")
    class OffCanvasRandom(random.Random):
        def uniform(self, low, high):
            if (low, high) == (0.12, 0.88):
                return 50.0
            return super().uniform(low, high)
    args = setup
    style = stamp.measure_style(args.data_dir / "red.png")
    face = stamp.load_fonts(args.fonts, list("ABab "))[0]
    clean = stamp.render_image("ABab ABab ABab", face, style.body, random.Random(1))
    output, metadata = stamp.overlay_stamp(clean, style, face, args.stamp_text, OffCanvasRandom(2))
    assert metadata["section"] == "middle"
    assert metadata["body_overlap_fraction"] >= 0.01
    assert stamp.red_mask(np.asarray(output)).sum() >= 8


def test_unknown_body_characters_fail_before_writing(setup):
    args = setup
    args.corpus.write_text("unused.png\tZ\n")
    with pytest.raises(ValueError, match="train_annotation.txt:1"):
        stamp.generate(args)
    assert not args.output_dir.exists()


def test_render_failure_does_not_publish_annotation(setup, monkeypatch):
    args = setup
    def fail(*unused):
        raise RuntimeError("render failed")
    monkeypatch.setattr(stamp, "render_image", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        stamp.generate(args)
    assert not args.annotation.exists()
    assert args.annotation.with_name("labels.txt.partial").is_file()


def test_partial_annotation_prevents_restart(setup):
    args = setup
    args.annotation.with_name("labels.txt.partial").write_text("keep")
    with pytest.raises(FileExistsError, match="Partial annotation"):
        stamp.generate(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("scene", stamp.SCENES)
@pytest.mark.parametrize("body_text", ["A", "ABab ABab ABab ABab ABab ABab"])
def test_scenes_support_single_character_and_long_lines(setup, scene, body_text):
    args = setup
    style = stamp.measure_style(args.data_dir / "red.png")
    face = stamp.load_fonts(args.fonts, list("ABab "))[0]
    font = ImageFont.truetype(str(args.fonts[0]), 26)
    width = round(font.getlength(body_text)) + 8
    clean = Image.new("RGB", (width, 36), "white")
    ImageDraw.Draw(clean).text((4, 1), body_text, font=font, fill=(40, 40, 40))
    before = clean.tobytes()
    for seed in (2, 7, 19):
        output, meta = stamp.overlay_stamp(clean, style, face, args.stamp_text,
                                          random.Random(seed), scene=scene)
        assert output.size == clean.size and output.mode == "RGB"
        assert clean.tobytes() == before
        assert stamp.red_mask(np.asarray(output)).sum() >= 8
        assert meta["body_overlap_fraction"] >= (0.002 if scene == "rim" else 0.01)
        if scene == "left_edge":
            assert meta["red_center_x"] <= 0.35
        elif scene == "right_edge":
            assert meta["red_center_x"] >= 0.65
        elif scene == "faded":
            assert 0.24 <= meta["opacity"] <= 0.42
        elif scene == "dense":
            assert 0.82 <= meta["opacity"] <= 0.98


def test_worn_mask_removes_stamp_ink_without_adding_pixels():
    mask = Image.new("L", (100, 100))
    ImageDraw.Draw(mask).rectangle((20, 20, 80, 80), fill=255)
    worn = np.asarray(stamp.weather_mask(mask, random.Random(1), True))
    clean = np.asarray(stamp.weather_mask(mask, random.Random(1), False))
    assert np.all(worn[np.asarray(mask) == 0] == 0)
    assert np.count_nonzero(worn) < np.count_nonzero(clean)
    assert worn.sum() < clean.sum()


def test_scene_subset_is_balanced_and_duplicates_rejected(setup):
    setup.scenes = ["faded", "right_edge"]
    report = stamp.generate(setup)
    assert report["scenes"] == {"faded": 3, "right_edge": 3}
    setup.scenes = ["faded", "faded"]
    with pytest.raises(ValueError, match="unique known scenes"):
        stamp.generate(setup)


def test_nearly_square_reference_selects_single_character_corpus(setup, monkeypatch):
    from dataclasses import replace
    style = replace(stamp.measure_style(setup.data_dir / "red.png"), aspect=1.2)
    monkeypatch.setattr(stamp, "measure_styles", lambda paths: [style])
    setup.corpus.write_text("one.png\tA\ntwo.png\tABab\n")
    setup.images = 1
    stamp.generate(setup)
    relative, text = split_annotation_fields(setup.annotation.read_text(), setup.annotation, 1)
    assert text == "A"
    assert (setup.annotation.parent / relative).is_file()
