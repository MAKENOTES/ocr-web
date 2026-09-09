import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from recognition.utils.annotation import split_annotation_fields
from recognition.utils.synthesis import generate_lines as lines
from recognition.utils.synthesis.style import Style


@pytest.mark.parametrize("pattern", lines.PATTERNS)
def test_lines_are_visible_on_requested_sides_and_preserve_body(pattern):
    body = Image.new("RGB", (60, 22), "white")
    ImageDraw.Draw(body).rectangle((3, 3, 56, 18), fill="black")
    style = Style((255, 255, 255), (40, 40, 40), 22, 0.7, 0.1, 3, "test")
    output, metadata = lines.add_lines(body, style, pattern, random.Random(5))
    assert output.mode == "RGB"
    assert output.crop(metadata["body_box"]).tobytes() == body.tobytes()
    pixels = np.asarray(output)
    edges = {"top": pixels[0], "bottom": pixels[-1],
             "left": pixels[:, 0], "right": pixels[:, -1]}
    for side, edge in edges.items():
        dark_fraction = (edge.mean(axis=-1) < 180).mean()
        assert (dark_fraction > 0.9) == (side in lines.PATTERNS[pattern])
    again, second = lines.add_lines(body, style, pattern, random.Random(5))
    assert again.tobytes() == output.tobytes() and second == metadata


@pytest.fixture
def args(tmp_path):
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        pytest.skip("DejaVuSans font unavailable")
    reference = Image.new("RGB", (100, 30), "white")
    draw = ImageDraw.Draw(reference)
    draw.text((4, 0), "AB12", font=ImageFont.truetype(str(font), 22), fill=(40, 40, 40))
    draw.line((0, 28, 99, 28), fill=(70, 70, 70))
    reference.save(tmp_path / "ref.png")
    (tmp_path / "references.txt").write_text("ref.png\n")
    (tmp_path / "keys.txt").write_text('A\nB\n1\n2\n"\n,\n')
    return lines.parse_args([
        "--data-dir", str(tmp_path), "--references", str(tmp_path / "references.txt"),
        "--keys", str(tmp_path / "keys.txt"), "--fonts", str(font), "--images", "10",
        "--output-dir", str(tmp_path / "images"), "--annotation", str(tmp_path / "labels.txt"),
        "--report-dir", str(tmp_path / "report"),
    ])


def test_generation_counts_labels_font_coverage_and_no_overwrite(args):
    pytest.importorskip("text_renderer")
    before = (args.data_dir / "ref.png").read_bytes()
    summary = lines.generate(args)
    assert summary["patterns"] == dict.fromkeys(lines.PATTERNS, 2)
    assert len(list(args.output_dir.glob("*.png"))) == 10
    records = [json.loads(row) for row in (args.report_dir / "samples.jsonl").read_text().splitlines()]
    with args.annotation.open() as handle:
        quoted = list(csv.reader(handle, skipinitialspace=True))
    training = Path(summary["training_annotation"])
    labels = [split_annotation_fields(row, training, i + 1)
              for i, row in enumerate(training.read_text().splitlines())]
    assert quoted == [list(row) for row in labels]
    for record, (relative, text) in zip(records, labels):
        assert record["text"] == text and set(text) <= set('AB12",')
        assert record["file"] == relative and not Path(relative).is_absolute()
        with Image.open(training.parent / relative) as image:
            image.load()
            assert image.mode == "RGB"
    assert (args.data_dir / "ref.png").read_bytes() == before
    with pytest.raises(FileExistsError):
        lines.generate(args)


def test_missing_reference_fails_before_writing(args):
    args.references.write_text("missing.png\n")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        lines.generate(args)
    assert not args.output_dir.exists()


def test_failed_generation_does_not_publish_labels(args, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("render failed")
    monkeypatch.setattr(lines, "render_image", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        lines.generate(args)
    assert not args.annotation.exists()
    assert not args.annotation.with_name("labels_training.txt").exists()


def test_reference_list_has_all_fifteen_unique_samples():
    entries = [row for row in Path(lines.__file__).with_name("line_references.txt").read_text().splitlines()
               if row and not row.startswith("#")]
    assert len(entries) == len(set(entries)) == 15
    assert all(row.startswith("val_images/") for row in entries)
