"""保证 blank 评估可复现，并且不会静默吞掉坏数据。"""

import json

import numpy as np
from PIL import Image
import pytest

from docs.detection.evaluate_blank import (
    audit_dataset,
    evaluate_background,
    evaluate_ctc,
    evaluate_postprocess,
)


def test_ctc_repeated_characters_need_blank_and_space_is_preserved():
    result = evaluate_ctc()
    assert [case["decoded"] for case in result["cases"]] == ["A", "AA", "", "A A"]
    assert result["valid_paths_with_blank"] == 1
    assert result["valid_paths_without_blank"] == 0
    assert result["ctc_loss_with_blank"] < 1e-4
    assert result["ctc_loss_without_blank"] == "infinity"
    assert not result["target_contains_blank"]
    json.dumps(result, allow_nan=False)


def test_background_supervision_and_empty_batch_are_distinct():
    result = evaluate_background((256, 768))
    assert result["mixed_background_score_gradient_mean"] > 0
    assert result["mixed_foreground_score_gradient_mean"] < 0
    assert result["mixed_background_geo_gradient_max"] == 0
    assert result["positive_only_background_score_gradient_max"] == 0
    assert result["empty_batch_geo_loss"] == 0
    assert 0 < result["empty_batch_dice_gradient_mean"] < 1e-10
    assert result["empty_batch_bce_gradient_mean"] > 1e-5


def test_real_postprocessor_separates_boxes_using_background():
    result = evaluate_postprocess((256, 768))
    cases = result["cases"]
    np.testing.assert_allclose(cases["correct_background"]["boxes"],
                               result["ground_truth_boxes"], atol=1e-4)
    assert cases["all_text"]["connected_regions"] == 1
    assert cases["all_text"]["box_count"] == 1
    assert cases["all_background"]["box_count"] == 0


def test_dataset_audit_is_reproducible_and_rejects_bad_labels(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), "white").save(images / f"{i}.png")
        (labels / f"{i}.txt").write_text("4 4 28 28\n", encoding="utf-8")
    first = audit_dataset(images, labels, 2, 42, (32, 32))
    assert first == audit_dataset(images, labels, 2, 42, (32, 32))
    assert first["sampled_images"] == 2
    assert 0 < first["background_fraction"] < 1
    bad_label = labels / "0.txt"
    bad_label.write_text("broken annotation\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"0\.txt"):
        audit_dataset(images, labels, 3, 42, (32, 32))
    bad_label.write_text("4 4 28 28\n", encoding="utf-8")
    (labels / "gt_0.txt").write_text("4 4 28 28\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Ambiguous label"):
        audit_dataset(images, labels, 3, 42, (32, 32))
