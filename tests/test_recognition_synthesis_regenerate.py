import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pytest

from recognition.utils.synthesis.regenerate import generate, load_texts, parse_args


@pytest.mark.parametrize("content", ["a.png\t\n", "a.png\tZ\n", "a.png A\n", ""])
def test_invalid_source_is_rejected_with_path(tmp_path, content):
    source = tmp_path / "source.txt"
    source.write_text(content)
    with pytest.raises(ValueError, match="source.txt"):
        load_texts(source, set("AB_"))


def test_regeneration_preserves_text_and_publishes_only_complete_output(tmp_path, monkeypatch):
    pytest.importorskip("text_renderer")
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        pytest.skip("DejaVuSans font unavailable")
    references = tmp_path / "references"
    references.mkdir()
    im = Image.new("RGB", (110, 32), "white")
    ImageDraw.Draw(im).text((3, 1), "A_B", font=ImageFont.truetype(str(font), 24), fill="black")
    im.save(references / "ref.png")
    source = tmp_path / "source.txt"
    source.write_text("old.png\tA_B\nsecond.png\tBA\n")
    keys = tmp_path / "keys.txt"
    keys.write_text("A\nB\n_\n")
    args = parse_args([
        "--source-annotation", str(source), "--keys", str(keys), "--prefix", "synthesis1",
        "--fonts", str(font), "--references", str(references),
        "--output-dir", str(tmp_path / "new"), "--annotation", str(tmp_path / "labels.txt"),
        "--report-dir", str(tmp_path / "report"),
    ])
    assert generate(args)["dry_run"]
    assert not args.output_dir.exists()
    args.apply = True
    summary = generate(args)
    assert summary["images"] == 2
    labels = [row.split("\t", 1) for row in args.annotation.read_text().splitlines()]
    assert [text for _, text in labels] == ["A_B", "BA"]
    records = [json.loads(row) for row in (args.report_dir / "samples.jsonl").read_text().splitlines()]
    for (relative, text), record in zip(labels, records):
        assert record["text"] == text
        with Image.open(args.annotation.parent / relative) as image:
            image.load()
            assert image.mode == "RGB" and image.width > 0
    assert source.read_text() == "old.png\tA_B\nsecond.png\tBA\n"
    with pytest.raises(FileExistsError):
        generate(args)

    args.output_dir = tmp_path / "failed_images"
    args.annotation = tmp_path / "failed.txt"
    args.report_dir = tmp_path / "failed_report"
    def fail(*args, **kwargs):
        raise RuntimeError("render failed")
    monkeypatch.setattr("recognition.utils.synthesis.regenerate.render_image", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        generate(args)
    assert not args.annotation.exists()
