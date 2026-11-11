"""验证小图检测的增强注册、框安全、忽略掩码和严格标签加载。"""
import numpy as np
import pytest
import torch
from PIL import Image

from detection_small.components.augmentation import (
    AUGMENTATIONS,
    POLICIES,
    Augmenter,
    _sanitize,
    get_geometry_augmenter,
    get_pixel_augmenter,
    normalize_stage_weights,
    policy_weights_for_epoch,
)
from detection_small.components.dataset import OCRDataset, read_quads
from detection_small.components.loss import EASTLoss
from detection_small.components.transforms import get_transforms


def make_sample(width=128, height=96):
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    image[20:40, 10:70] = 20
    quads = np.array([
        [[10, 20], [70, 20], [70, 40], [10, 40]],
        [[80, 50], [120, 50], [120, 80], [80, 80]],
    ], dtype=np.float32)
    return {
        "image": Image.fromarray(image, "RGB"),
        "quads": quads,
        "ignored": np.array([False, True]),
    }


def test_registry_contains_expected_operations():
    assert len(AUGMENTATIONS) == 29
    assert len({operation.name for operation in AUGMENTATIONS}) == len(AUGMENTATIONS)
    assert set(POLICIES) == {"light", "standard", "strong"}
    for operation in AUGMENTATIONS:
        assert 0 < operation.probability <= 1
        assert operation.category in {"geometry", "photometric", "occlusion"}
        assert callable(operation.function)
    # 紧凑 UI 裁剪不使用随机裁剪，避免系统性截断文字。
    assert "random_crop" not in {operation.name for operation in AUGMENTATIONS}


@pytest.mark.parametrize("operation", AUGMENTATIONS, ids=lambda operation: operation.name)
def test_each_operation_keeps_image_and_box_contract(operation):
    source = make_sample()
    rng = np.random.default_rng(20261111)
    image, quads, ignored = operation.function(
        np.asarray(source["image"]).copy(), source["quads"].copy(),
        source["ignored"].copy(), rng, 1.0)
    if operation.category == "geometry":
        quads, ignored = _sanitize(quads, ignored, image.shape[1], image.shape[0])

    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[0] > 0 and image.shape[1] > 0
    assert quads.ndim == 3 and quads.shape[1:] == (4, 2)
    assert len(quads) == len(ignored)
    if len(quads):
        height, width = image.shape[:2]
        assert np.isfinite(quads).all()
        assert quads[:, :, 0].min() >= 0 and quads[:, :, 0].max() <= width
        assert quads[:, :, 1].min() >= 0 and quads[:, :, 1].max() <= height
        assert (quads[:, :, 0].max(axis=1) - quads[:, :, 0].min(axis=1) > 0).all()
        assert (quads[:, :, 1].max(axis=1) - quads[:, :, 1].min(axis=1) > 0).all()


@pytest.mark.parametrize("operation", AUGMENTATIONS, ids=lambda operation: operation.name)
def test_each_operation_accepts_empty_boxes(operation):
    image = np.full((32, 48, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(7)
    output, quads, ignored = operation.function(
        image, np.empty((0, 4, 2), dtype=np.float32),
        np.empty(0, dtype=bool), rng, 1.0)
    assert output.shape[2] == 3
    assert quads.ndim == 3
    assert len(quads) == len(ignored)


def test_augmenter_returns_pil_and_keeps_labels_valid():
    augmenters = (get_geometry_augmenter({"standard": 1.0}),
                  get_pixel_augmenter({"standard": 1.0}))
    torch.manual_seed(20261111)
    for augmenter in augmenters:
        for _ in range(10):
            result = augmenter(make_sample())
            image, quads, ignored = result["image"], result["quads"], result["ignored"]
            assert image.mode == "RGB"
            assert quads.ndim == 3 and quads.shape[1:] == (4, 2)
            assert len(quads) == len(ignored)
            width, height = image.size
            if len(quads):
                assert quads[:, :, 0].min() >= 0 and quads[:, :, 0].max() <= width
                assert quads[:, :, 1].min() >= 0 and quads[:, :, 1].max() <= height


def test_sanitize_clips_and_drops_degenerate_boxes():
    quads = np.array([
        [[-5, 0], [10, 0], [10, 5], [-5, 5]],
        [[1, 1], [1.2, 1], [1.2, 1.2], [1, 1.2]],
    ], dtype=np.float32)
    ignored = np.array([True, False])
    clipped, kept = _sanitize(quads, ignored, 8, 8)
    assert clipped.shape == (1, 4, 2)
    assert clipped[0, :, 0].min() == 0 and clipped[0, :, 0].max() == 8
    assert kept.tolist() == [True]


def test_stage_weights_are_normalized_and_curriculum_moves_to_strong():
    for weights in (normalize_stage_weights(), normalize_stage_weights({"light": 2, "strong": 1})):
        assert abs(sum(weights.values()) - 1) < 1e-12
        assert set(weights) <= set(POLICIES)

    early = policy_weights_for_epoch(1, 400)
    middle = policy_weights_for_epoch(200, 400)
    late = policy_weights_for_epoch(400, 400)
    for weights in (early, middle, late):
        assert abs(sum(weights.values()) - 1) < 1e-12
    assert early["light"] > late["light"]
    assert late["strong"] > early["strong"]
    assert policy_weights_for_epoch(1, 1)["light"] > 0


@pytest.mark.parametrize("categories", [set(), {"geometry", "photometric"}, {"unknown"}])
def test_augmenter_rejects_invalid_category_sets(categories):
    with pytest.raises(ValueError):
        Augmenter({"standard": 1.0}, categories)


@pytest.mark.parametrize("weights", [{}, {"unknown": 1}, {"standard": -1}, {"standard": 0}])
def test_invalid_stage_weights(weights):
    with pytest.raises(ValueError):
        normalize_stage_weights(weights)


@pytest.mark.parametrize("epoch,epochs", [(0, 10), (1, 0)])
def test_invalid_curriculum_epoch(epoch, epochs):
    with pytest.raises(ValueError):
        policy_weights_for_epoch(epoch, epochs)


def test_validation_transform_is_deterministic_and_training_transform_is_connected():
    sample = make_sample()
    validation = get_transforms(False, (64, 64))
    first = validation(dict(sample))
    second = validation(dict(sample))
    for key in ("image", "score_map", "geo_map", "training_mask"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    torch.manual_seed(20261111)
    training = get_transforms(True, (64, 64), stage_weights={"standard": 1.0})
    batch = training(dict(sample))
    assert batch["image"].shape == (3, 64, 64)
    assert batch["score_map"].shape == (16, 16)
    assert batch["geo_map"].shape == (4, 16, 16)
    assert batch["training_mask"].shape == (16, 16)


def test_ignored_box_does_not_contribute_ground_truth_or_loss():
    image = np.full((64, 64, 3), 240, dtype=np.uint8)
    image[8:24, 8:56] = 30
    sample = {
        "image": Image.fromarray(image, "RGB"),
        "quads": np.array([[[8, 8], [56, 8], [56, 24], [8, 24]]], dtype=np.float32),
        "ignored": np.array([True]),
    }
    batch = get_transforms(False, (64, 64))(sample)
    assert float(batch["score_map"].max()) == 0
    assert float(batch["geo_map"].sum()) == 0
    assert float(batch["training_mask"].min()) == 0

    criterion = EASTLoss(lambda_iou=3)
    score_true = torch.zeros(1, 1, 16, 16)
    score_pred = torch.zeros(1, 1, 16, 16)
    geo = torch.zeros(1, 4, 16, 16)
    mask = batch["training_mask"].unsqueeze(0)
    loss, score_loss, geo_loss = criterion(
        score_true, score_pred, geo, geo, mask)
    assert torch.isfinite(loss) and torch.isfinite(score_loss) and torch.isfinite(geo_loss)


def _write_case(root, name, label_text, size=(80, 40)):
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    Image.new("RGB", size, "white").save(image_dir / f"{name}.png")
    (label_dir / f"gt_{name}.txt").write_text(label_text, encoding="utf-8")
    return image_dir, label_dir


def test_read_quads_parses_bbox4_quad8_and_ignore_marker(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text(
        "10 5 50 30\n"
        "10 5 50 5 50 30 10 30\n"
        "0 0 20 20 ###\n", encoding="utf-8")
    quads, ignored = read_quads(label, size=(80, 40))
    assert quads.shape == (3, 4, 2)
    assert ignored.tolist() == [False, False, True]


def test_read_quads_rejects_rotated_quad_and_out_of_bounds(tmp_path):
    rotated = tmp_path / "rotated.txt"
    rotated.write_text("10 5 50 10 50 30 10 35\n", encoding="utf-8")
    with pytest.raises(ValueError, match="水平矩形"):
        read_quads(rotated, size=(80, 40), label_format="quad8")

    outside = tmp_path / "outside.txt"
    outside.write_text("10 5 90 30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="超出图片范围"):
        read_quads(outside, size=(80, 40))


def test_dataset_requires_exactly_one_label_and_exposes_ignored(tmp_path):
    image_dir, label_dir = _write_case(tmp_path, "a", "10 5 50 30\n0 0 20 20 ###\n")
    dataset = OCRDataset(image_dir, label_dir)
    sample = dataset[0]
    assert sample["quads"].shape == (2, 4, 2)
    assert sample["ignored"].tolist() == [False, True]

    (label_dir / "a.txt").write_text("10 5 50 30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="多份标签"):
        OCRDataset(image_dir, label_dir)

    (label_dir / "gt_a.txt").unlink()
    (label_dir / "a.txt").write_text("10 5 50 30\n", encoding="utf-8")
    dataset = OCRDataset(image_dir, label_dir)
    assert len(dataset) == 1

    (label_dir / "a.txt").unlink()
    with pytest.raises(FileNotFoundError, match="缺少对应标签"):
        OCRDataset(image_dir, label_dir, require_labels=True)
    assert len(OCRDataset(image_dir, label_dir, require_labels=False)) == 1
