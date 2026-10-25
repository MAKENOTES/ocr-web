"""训练增强的标签安全性和随机可控性。

几何扰动不能裁掉字形；随机增强只在训练集启用；验证变换保持确定性。"""

import random

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from recognition.components.dataset import (
    InferenceDataset,
    TextRecognitionDataset,
)
from recognition.components.augmentation import (
    MAX_ROTATED_SPAN,
    RandomSourceGeometry,
    build_transform,
    ink_extent,
)
from recognition.components.preprocessing import border_fill_color

BACKGROUND = (237, 242, 250)  # 浅蓝灰纸面，与失败样本一致的背景色。
INK = (30, 30, 30)  # 深灰正文，和背景的通道差远大于墨迹阈值。


def _text_image(width=138, height=21, margin=1):
    """生成文字紧贴四边（只留 margin 像素）的 detection 框样本。"""
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for x in range(margin, width - margin - 4, 8):
        draw.rectangle((x, margin, x + 5, height - 1 - margin), fill=INK)
    return image


def _ink_mass(image):
    """相对纸面色差的总和；用于确认旋转前后没有墨迹被裁掉。"""
    pixels = np.asarray(image).astype(np.int16)
    return float(np.abs(pixels - np.asarray(BACKGROUND, dtype=np.int16)).sum())


def _operations(pad_px=3, **enabled):
    """只打开指定操作，其余概率为 0，避免单测依赖随机组合。"""
    params = {
        "scale_p": 0.0,
        "stretch_p": 0.0,
        "pad_p": 0.0,
        "rotate_p": 0.0,
        "shear_p": 0.0,
        "perspective_p": 0.0,
        "elastic_p": 0.0,
        "crop_p": 0.0,
    }
    params.update(enabled)
    return RandomSourceGeometry(pad_px=pad_px, **params)


def test_zero_probability_returns_the_same_pixels():
    image = _text_image()
    before = np.asarray(image).copy()

    result = _operations()(image)

    assert result.size == image.size
    assert np.array_equal(np.asarray(result), before)
    assert np.array_equal(np.asarray(image), before)  # 不修改调用方图片。


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_rotation_keeps_every_glyph(seed):
    """旋转换到扩大画布后按墨迹重裁，笔画不应被画布边缘截断。"""
    image = _text_image()
    before = _ink_mass(image)

    random.seed(seed)
    result = _operations(rotate_p=1.0)(image)

    # 双三次重采样会让边缘轻微过冲，只要求墨迹总量不减少。
    assert _ink_mass(result) >= before * 0.98
    assert result.width <= image.width + 1
    assert result.height <= image.height + 2 * MAX_ROTATED_SPAN * image.height + 1
    assert border_fill_color(result) != INK  # 新边缘是纸面色，不是黑边。


def test_rotation_angle_is_limited_by_aspect_ratio():
    """长行旋转后新增的垂直跨度不能超过文字高度的一小部分，否则文字会被缩小数倍。"""
    long_image = _text_image(width=1600, height=32)

    random.seed(0)
    result = _operations(rotate_p=1.0)(long_image)

    assert result.height - long_image.height <= MAX_ROTATED_SPAN * long_image.height + 1


def test_scale_changes_text_share_without_distorting_glyphs():
    image = _text_image()
    ink = ink_extent(image, border_fill_color(image))
    text_aspect = (ink[2] - ink[0]) / (ink[3] - ink[1])

    random.seed(0)
    result = _operations(scale_p=1.0)(image)
    result_ink = ink_extent(result, border_fill_color(result))
    result_aspect = (result_ink[2] - result_ink[0]) / (result_ink[3] - result_ink[1])

    # 等比缩放只改变文字占画布的比例，不改变字形宽高比。
    assert result_aspect == pytest.approx(text_aspect, rel=0.06)
    assert result.size != image.size


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stretch_changes_width_only(seed):
    image = _text_image()
    ink = ink_extent(image, border_fill_color(image))
    ink_width = ink[2] - ink[0]

    random.seed(seed)
    result = _operations(stretch_p=1.0)(image)
    result_ink = ink_extent(result, border_fill_color(result))

    assert ink_width * 0.89 <= result_ink[2] - result_ink[0] <= ink_width * 1.11
    assert result_ink[3] - result_ink[1] == ink[3] - ink[1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_padding_adds_background_around_the_text(seed):
    image = _text_image()
    ink = ink_extent(image, border_fill_color(image))
    ink_size = (ink[2] - ink[0], ink[3] - ink[1])

    random.seed(seed)
    result = _operations(pad_p=1.0)(image)

    assert ink_size[0] <= result.width <= ink_size[0] + 6
    assert ink_size[1] <= result.height <= ink_size[1] + 6
    assert result.getpixel((0, 0)) == BACKGROUND


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_zero_padding_still_keeps_the_crop_inside_the_canvas(seed):
    """pad_px=0 时重裁仍需外扩画布；越界部分会被 Pillow 补成黑色。"""
    image = _text_image()

    random.seed(seed)
    result = _operations(pad_px=0, rotate_p=1.0)(image)

    # 双三次重采样会在紧邻墨迹处轻微过冲，因此只要求是纸面色而非黑色。
    for corner in (
        (0, 0),
        (result.width - 1, 0),
        (0, result.height - 1),
        (result.width - 1, result.height - 1),
    ):
        assert min(result.getpixel(corner)) > 180


def test_same_seed_produces_the_same_image():
    image = _text_image()

    random.seed(42)
    first = np.asarray(_operations(scale_p=1.0, stretch_p=1.0, pad_p=1.0, rotate_p=1.0)(image))
    random.seed(42)
    second = np.asarray(_operations(scale_p=1.0, stretch_p=1.0, pad_p=1.0, rotate_p=1.0)(image))

    assert np.array_equal(first, second)


def test_geometry_augmentation_runs_only_for_training(tmp_path):
    image_path = tmp_path / "text.png"
    _text_image().save(image_path)
    annotation_path = tmp_path / "annotations.txt"
    annotation_path.write_text("text.png\t正文\n", encoding="utf-8")

    training = TextRecognitionDataset(annotation_path, 32, 240, 3, is_train=True)
    validation = TextRecognitionDataset(annotation_path, 32, 240, 3, is_train=False)
    inference = InferenceDataset([image_path], 32, 3, img_width=240)

    assert isinstance(training.source_augment, RandomSourceGeometry)
    assert validation.source_augment is None
    assert inference.source_augment is None


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_shear_perspective_and_elastic_keep_every_glyph(seed):
    """斜切、透视、弹性扭曲都要扩边，字形不能被画布截断或产生黑边。"""
    image = _text_image()
    before = _ink_mass(image)

    for kwargs in ({"shear_p": 1.0}, {"perspective_p": 1.0}, {"elastic_p": 1.0}):
        random.seed(seed)
        result = _operations(**kwargs)(image)
        assert _ink_mass(result) >= before * 0.95
        for corner in (
            (0, 0),
            (result.width - 1, 0),
            (0, result.height - 1),
            (result.width - 1, result.height - 1),
        ):
            assert min(result.getpixel(corner)) > 180


def test_only_one_shape_operation_runs_per_sample(monkeypatch):
    """旋转、斜切、透视同属页面倾斜，同一次增强最多施加其中一种。"""
    geometry = _operations(rotate_p=1.0, shear_p=1.0, perspective_p=1.0)
    counts = {"rotate": 0, "shear": 0, "perspective": 0}
    for name in counts:
        original = getattr(geometry, "_" + name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(geometry, "_" + name, wrapper)

    for _ in range(30):
        geometry(_text_image())

    assert sum(counts.values()) == 30
    assert all(0 < count < 30 for count in counts.values())


def test_trim_removes_one_background_column():
    image = _text_image()
    before = _ink_mass(image)

    trimmed = RandomSourceGeometry._trim_one_side(image, BACKGROUND)

    assert trimmed.size != image.size
    assert _ink_mass(trimmed) >= before * 0.92


def test_trim_keeps_a_column_holding_too_much_ink():
    """切入正文的比例超过上限时保留原图；10x10 实心块的单列占全部墨迹 10%。"""
    solid = Image.new("RGB", (10, 10), INK)

    result = RandomSourceGeometry._trim_one_side(solid, BACKGROUND)

    assert result.size == solid.size


def test_augmentation_rejects_non_rgb_input():
    with pytest.raises(ValueError, match="需要 RGB"):
        RandomSourceGeometry()(_text_image().convert("L"))


def test_validation_transform_is_deterministic():
    image = _text_image()
    transform = build_transform(is_train=False, num_channels=3)
    first, second = transform(image), transform(image)
    torch.testing.assert_close(first, second)
    assert first.shape[0] == 3


def test_transform_adapts_single_channel_and_rejects_others():
    tensor = build_transform(is_train=False, num_channels=1)(_text_image())
    assert tensor.shape[0] == 1
    with pytest.raises(ValueError, match="num_channels 仅支持"):
        build_transform(is_train=False, num_channels=2)
