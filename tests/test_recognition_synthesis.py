from collections import Counter
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont
import pytest

from recognition.utils.annotation import split_annotation_fields
from recognition.utils.synthesis.generate import (
    FontFace, companion_pool, generate, load_fonts, make_text, parse_args, read_chars,
)
from recognition.utils.synthesis.style import sample_styles


def test_target_quota_excludes_other_targets_from_companions():
    targets = list("甲乙")
    face = FontFace(Path("fake.ttf"), 0, "test", frozenset("甲乙中文A"))
    pool = companion_pool(list("甲乙中文AB"), targets, face)
    rng = random.Random(42)
    labels = [make_text(target, pool, 5, rng) for target in targets for _ in range(100)]
    assert Counter(c for text in labels for c in text if c in targets) == {"甲": 100, "乙": 100}
    assert all(len(text) == 5 and set(text) <= face.chars for text in labels)


@pytest.mark.parametrize("text", ["甲\n甲\n", "甲乙\n", "\n", " \n"])
def test_invalid_character_files_fail_with_path(tmp_path, text):
    path = tmp_path / "keys.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="keys.txt"):
        read_chars(path)


def test_missing_font_coverage_fails_before_generation():
    with pytest.raises(ValueError, match="甲"):
        load_fonts([], ["甲"])


def test_existing_images_are_never_overwritten(tmp_path):
    output = tmp_path / "images"
    output.mkdir()
    existing = output / "existing.png"
    existing.write_bytes(b"keep")
    args = parse_args(["--output-dir", str(output)])
    with pytest.raises(FileExistsError, match="images"):
        generate(args)
    assert existing.read_bytes() == b"keep"


def test_synthesis_outputs_exact_quota_and_readable_rgb_annotations(tmp_path):
    pytest.importorskip("text_renderer")
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("DejaVuSans font unavailable")
    references = tmp_path / "references"
    references.mkdir()
    source = Image.new("RGB", (110, 30), "white")
    ImageDraw.Draw(source).text((3, 1), "Abc", font=ImageFont.truetype(str(font), 23), fill="blue")
    source.save(references / "sample.png")
    (tmp_path / "targets.txt").write_text("A\nB\n")
    (tmp_path / "keys.txt").write_text("A\nB\na\nb\nc\n")
    args = parse_args([
        "--targets", str(tmp_path / "targets.txt"), "--keys", str(tmp_path / "keys.txt"),
        "--fonts", str(font), "--references", str(references), "--per-char", "3",
        "--output-dir", str(tmp_path / "images"), "--annotation", str(tmp_path / "labels.txt"),
        "--report-dir", str(tmp_path / "report"),
    ])
    report = generate(args)
    assert report["counts"] == {"A": 3, "B": 3}
    counts = Counter()
    for number, line in enumerate(args.annotation.read_text().splitlines(), 1):
        relative, text = split_annotation_fields(line, args.annotation, number)
        assert Path(relative).name == f"{args.batch_name}_{args.date}_{number:05d}.png"
        assert not Path(relative).is_absolute()
        with Image.open(args.annotation.parent / relative) as image:
            assert image.mode == "RGB"
            assert image.height >= 16
            assert image.getextrema() != ((255, 255), (255, 255), (255, 255))
        counts.update(c for c in text if c in "AB")
    assert counts == {"A": 3, "B": 3}
    assert len(list(args.output_dir.glob("*.png"))) == 6
    assert not args.annotation.with_suffix(".txt.partial").exists()
    assert (args.report_dir / "preview.png").is_file()
    with pytest.raises(FileExistsError):
        generate(args)


def test_reference_style_does_not_retain_original_pixels(tmp_path):
    image = Image.new("RGB", (80, 24), "white")
    ImageDraw.Draw(image).rectangle((5, 3, 70, 20), fill=(0, 120, 250))
    image.save(tmp_path / "reference.png")
    style = sample_styles([tmp_path], 10, 42)[0]
    assert style.height == 24
    assert not any(isinstance(value, Image.Image) for value in style.to_dict().values())
    assert style.background != style.foreground
