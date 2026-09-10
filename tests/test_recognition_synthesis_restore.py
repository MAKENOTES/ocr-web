import json
from pathlib import Path
import random

from PIL import Image
import pytest

from recognition.utils.synthesis.restore import generate, parse_args, read_plan, render_record
from recognition.utils.synthesis.generate import load_fonts


def inputs(tmp_path, effect=None):
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("DejaVuSans font unavailable")
    annotation = tmp_path / "annotation.txt"
    annotation.write_text("images/keep_name.png\tA_B\n")
    keys = tmp_path / "keys.txt"
    keys.write_text("A\nB\n_\n")
    record = {"file": "old_dir/keep_name.png", "text": "A_B", "font": str(font), "font_index": 0,
              "style": {"background": [255, 255, 255], "foreground": [30, 30, 30],
                        "height": 32, "ink_fraction": .75, "side_fraction": .1,
                        "length": 3, "reference": "deleted.png"}}
    if effect:
        record.update(effect)
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record) + "\n")
    args = parse_args(["--annotation", str(annotation), "--records", str(records),
                       "--keys", str(keys), "--report-dir", str(tmp_path / "report")])
    return args, record


@pytest.mark.parametrize("effect", [None, {"lines": {"pattern": "bottom"}},
                                    {"effect": {"pattern": "low_resolution"}},
                                    {"effect": {"pattern": "tight_right"}},
                                    {"effect": {"pattern": "fragment", "fragment_character": "B"}}])
def test_restore_keeps_annotations_names_and_effects(tmp_path, effect):
    pytest.importorskip("text_renderer")
    args, record = inputs(tmp_path, effect)
    before = args.annotation.read_bytes()
    assert generate(args)["dry_run"]
    assert not (tmp_path / "images").exists()
    args.apply = True
    summary = generate(args)
    assert summary["images"] == 1
    assert args.annotation.read_bytes() == before
    with Image.open(tmp_path / "images/keep_name.png") as image:
        image.load()
        assert image.mode == "RGB" and min(image.size) > 0
    output = json.loads((args.report_dir / "samples.jsonl").read_text())
    assert output["text"] == "A_B" and output["file"] == "images/keep_name.png"
    if effect:
        field = next(iter(effect))
        assert output[field]["pattern"] == effect[field]["pattern"]
        if effect[field]["pattern"] == "fragment":
            assert output[field]["fragment_character"] == "B"
    args.report_dir = tmp_path / "another_report"
    with pytest.raises(FileExistsError, match="keep_name.png"):
        generate(args)


@pytest.mark.parametrize("line", ["../escape.png\tA_B\n", "images/keep_name.png\tAB\n",
                                  "images/keep_name.jpg\tA_B\n", "images/keep_name.png\tA_B\n" * 2])
def test_restore_rejects_unsafe_or_stale_sources(tmp_path, line):
    args, _ = inputs(tmp_path)
    args.annotation.write_text(line)
    with pytest.raises(ValueError):
        read_plan(args.annotation, args.records, set("A_B"))
    assert not (tmp_path / "images").exists()


def test_render_restores_red_stamp_using_existing_overlay(tmp_path):
    pytest.importorskip("text_renderer")
    _, record = inputs(tmp_path)
    record["text"] = "ABABAB"
    body = record["style"]
    record["style"] = {"body": body, "aspect": 5.0, "color": [220, 25, 40],
                       "opacity": .8, "red_fraction": .08, "diameter_ratio": 5,
                       "red_measured": True, "red_center_x": .5}
    record.update(stamp_font=record["font"], stamp_font_index=0,
                  stamp={"stamp_text": "AB", "scene": "rim"})
    faces = load_fonts([Path(record["font"])], list("AB_"))
    image, effect = render_record(record["text"], record, {(str(f.path), f.index): f for f in faces}, random.Random(12))
    assert image.mode == "RGB"
    assert effect["stamp"]["scene"] == "rim" and effect["stamp"]["red_fraction"] > 0
