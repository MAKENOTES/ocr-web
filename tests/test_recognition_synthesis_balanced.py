from collections import Counter
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pytest

from recognition.utils.annotation import split_annotation_fields
from recognition.utils.synthesis import generate_balanced as balanced
from recognition.utils.synthesis.generate import FontFace
from recognition.utils.synthesis.style import Style


def write_stats(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["字符", "忽略的中间列", "合计_次数"])
        writer.writerows((char, "12.34%", count) for char, count in rows)


def test_statistics_uses_first_and_last_columns_with_bom_and_quoted_comma(tmp_path):
    path = tmp_path / "stats.csv"
    write_stats(path, [("甲", 0), (",", 99), ("乙", 100), ("丙", 120)])
    counts = balanced.read_statistics(path)
    assert balanced.build_quotas(counts, 110, 10) == {"甲": 110, ",": 11}


@pytest.mark.parametrize("rows", [
    [("甲", 1), ("甲", 2)], [("甲乙", 2)], [("甲", -1)], [("甲", "1.5")],
])
def test_invalid_statistics_fail_with_source_path(tmp_path, rows):
    path = tmp_path / "bad.csv"
    write_stats(path, rows)
    with pytest.raises(ValueError, match="bad.csv:"):
        balanced.read_statistics(path)


@pytest.mark.parametrize("target_count", [100, 110, 120])
def test_plan_exact_image_count_and_final_occurrences(target_count):
    counts = {"甲": 0, "乙": 99, "A": 100}
    quotas = balanced.build_quotas(counts, target_count, 20)
    fonts = [FontFace(Path("fake.ttf"), 0, "test", frozenset(counts))]
    styles = [Style((255, 255, 255), (0, 0, 0), 30, 0.8, 0.1, 2, "reference.png")]
    samples = balanced.plan_samples(quotas, ["A"], fonts, styles, 20, 42)
    actual = Counter(char for sample in samples for char in sample.text if char in quotas)
    assert len(samples) == 20
    assert actual == quotas
    assert all(counts[char] + actual[char] == target_count for char in quotas)
    assert all(1 <= len(sample.text) <= 24 for sample in samples)
    assert samples == balanced.plan_samples(quotas, ["A"], fonts, styles, 20, 42)


def test_filler_images_do_not_increase_completed_targets():
    fonts = [FontFace(Path("fake.ttf"), 0, "test", frozenset("AB"))]
    styles = [Style((255, 255, 255), (0, 0, 0), 30, 0.8, 0.1, 5, "reference.png")]
    samples = balanced.plan_samples({"A": 1}, ["B"], fonts, styles, 10, 42)
    assert len(samples) == 10
    assert sum(sample.text.count("A") for sample in samples) == 1
    assert all(len(sample.text) == 5 for sample in samples)


def test_impossible_budget_and_out_of_range_counts_fail():
    with pytest.raises(ValueError, match="need >= 5"):
        balanced.build_quotas({"甲": 0}, 110, 4)
    with pytest.raises(ValueError, match="between 100 and 120"):
        balanced.build_quotas({"甲": 0}, 121, 100)
    with pytest.raises(ValueError, match="No target"):
        balanced.build_quotas({"甲": 100}, 110, 100)


@pytest.fixture
def generation_args(tmp_path):
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("DejaVuSans font unavailable")
    references = tmp_path / "references"
    references.mkdir()
    source = Image.new("RGB", (110, 30), "white")
    ImageDraw.Draw(source).text((3, 1), "Abc", font=ImageFont.truetype(str(font), 23), fill="blue")
    source.save(references / "sample.png")
    write_stats(tmp_path / "stats.csv", [("A", 98), ("B", 99), (",", 0), ("c", 100)])
    (tmp_path / "keys.txt").write_text("A\nB\n,\nc\n")
    return balanced.parse_args([
        "--stats", str(tmp_path / "stats.csv"), "--keys", str(tmp_path / "keys.txt"),
        "--fonts", str(font), "--references", str(references), "--target-count", "100",
        "--images", "8",
        "--output-dir", str(tmp_path / "images"), "--annotation", str(tmp_path / "labels.txt"),
        "--report-dir", str(tmp_path / "report"),
    ])


def test_outputs_date_names_rgb_labels_and_auditable_final_counts(generation_args):
    pytest.importorskip("text_renderer")
    args = generation_args
    report = balanced.generate(args)
    assert report["images"] == 8
    actual = Counter()
    lines = args.annotation.read_text().splitlines()
    assert len(lines) == 8
    for number, line in enumerate(lines, 1):
        relative, text = split_annotation_fields(line, args.annotation, number)
        assert Path(relative).name == f"{args.date}_{number:05d}.png"
        assert not Path(relative).is_absolute()
        with Image.open(args.annotation.parent / relative) as image:
            assert image.mode == "RGB"
            assert 16 <= image.height <= 64
            assert any(low != high for low, high in image.getextrema())
        actual.update(text)
    for char, before in [("A", 98), ("B", 99), (",", 0)]:
        assert actual[char] + before == 100
        assert report["counts"][char] == {"before": before, "added": actual[char], "after": 100}
    with (args.report_dir / "character_counts.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(int(row["最终次数"]) == 100 for row in rows)
    assert (args.report_dir / "preview.png").is_file()
    assert len((args.report_dir / "samples.jsonl").read_text().splitlines()) == 8
    assert not args.annotation.with_name("labels.txt.partial").exists()
    before = args.annotation.read_bytes()
    with pytest.raises(FileExistsError):
        balanced.generate(args)
    assert args.annotation.read_bytes() == before


def test_dry_run_validates_plan_without_outputs(generation_args):
    args = generation_args
    args.dry_run = True
    report = balanced.generate(args)
    assert report["added_occurrences"] == 103
    assert not args.output_dir.exists()
    assert not args.annotation.exists()
    assert not args.report_dir.exists()


def test_failed_render_does_not_publish_annotation(generation_args, monkeypatch):
    pytest.importorskip("text_renderer")
    args = generation_args
    def fail_render(*unused):
        raise ValueError("render failed")
    monkeypatch.setattr(balanced, "render_image", fail_render)
    with pytest.raises(ValueError, match="render failed"):
        balanced.generate(args)
    assert not args.annotation.exists()
    assert args.annotation.with_name("labels.txt.partial").exists()
    with pytest.raises(FileExistsError):
        balanced.generate(args)

