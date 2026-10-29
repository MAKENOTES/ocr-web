"""验证 30 种增强的注册、框安全、课程权重和训练/验证接线。"""
import numpy as np
import pytest
import torch
from PIL import Image

from detection.components.augmentation import (
    AUGMENTATIONS,
    POLICIES,
    Augmenter,
    _sanitize,
    get_geometry_augmenter,
    get_pixel_augmenter,
    normalize_stage_weights,
    policy_weights_for_epoch,
)
from detection.components.transforms import get_transforms


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


def test_registry_contains_expected_number_of_operations():
    assert len(AUGMENTATIONS) >= 30
    assert len({operation.name for operation in AUGMENTATIONS}) == len(AUGMENTATIONS)
    assert set(POLICIES) == {"light", "standard", "strong"}
    for operation in AUGMENTATIONS:
        assert 0 < operation.probability <= 1
        assert operation.category in {"geometry", "photometric", "occlusion"}
        assert callable(operation.function)


@pytest.mark.parametrize("operation", AUGMENTATIONS, ids=lambda operation: operation.name)
def test_each_operation_keeps_image_and_box_contract(operation):
    source = make_sample()
    rng = np.random.default_rng(20261028)
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
    assert quads.shape == (0, 4, 2) or quads.ndim == 3
    assert len(quads) == len(ignored)


def test_augmenter_returns_pil_and_keeps_labels_valid():
    augmenters = (get_geometry_augmenter({"standard": 1.0}),
                  get_pixel_augmenter({"standard": 1.0}))
    torch.manual_seed(20261028)
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

    early = policy_weights_for_epoch(1, 620)
    middle = policy_weights_for_epoch(310, 620)
    late = policy_weights_for_epoch(620, 620)
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
    validation = get_transforms(False, 64)
    first = validation(sample)
    second = validation(sample)
    for key in ("image", "score_map", "geo_map", "training_mask"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    torch.manual_seed(20261028)
    training = get_transforms(True, 64, stage_weights={"standard": 1.0})
    batch = training(sample)
    assert batch["image"].shape == (3, 64, 64)
    assert batch["score_map"].shape == (16, 16)
    assert batch["geo_map"].shape == (4, 16, 16)
    assert batch["training_mask"].shape == (16, 16)
