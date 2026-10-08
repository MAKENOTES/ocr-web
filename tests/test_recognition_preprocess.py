from pathlib import Path

import torch
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from recognition import evaluate_val
from recognition.components.dataset import (
    CNN_RIGHT_CONTEXT,
    InferenceDataset,
    RandomBackgroundGeometry,
    TextRecognitionDataset,
    dynamic_padding_collate_fn,
    load_rgb_image,
    remove_red_seal,
    resolve_image_path,
)
from recognition.utils.checkpoint import infer_num_channels, infer_num_layers


@pytest.mark.parametrize("operation", ["rotation", "perspective"])
@pytest.mark.parametrize("mode,background,ink", [
    ("RGB", (255, 255, 255), (0, 0, 0)),
    ("RGB", (0, 0, 0), (255, 255, 255)),
    ("RGB", (20, 110, 180), (255, 255, 255)),
    ("L", 40, 255),
])
def test_geometry_fills_new_edges_with_image_background(monkeypatch, operation, mode, background, ink):
    image = Image.new(mode, (160, 32), background)
    ImageDraw.Draw(image).rectangle((40, 8, 110, 23), fill=ink)
    before = np.asarray(image).copy()
    monkeypatch.setattr(
        transforms.RandomRotation, "get_params",
        staticmethod(lambda degrees: 2.0 if operation == "rotation" else 0.0),
    )
    monkeypatch.setattr(
        transforms.RandomPerspective, "get_params",
        staticmethod(lambda width, height, distortion_scale: (
            [[0, 0], [159, 0], [159, 31], [0, 31]],
            [[4, 1], [155, 1], [155, 30], [4, 30]],
        )),
    )
    geometry = RandomBackgroundGeometry(p=1.0 if operation == "perspective" else 0.0)

    result = geometry(image)

    assert result.mode == mode and result.size == image.size
    assert result.getpixel((0, 0)) == background
    assert result.getpixel((159, 31)) == background
    assert result.getpixel((75, 16)) == ink
    assert not np.array_equal(np.asarray(result), before)
    assert np.array_equal(np.asarray(image), before)
    # 同一个增强实例处理下一张图片时，必须重新估计背景。
    next_background = (80, 60, 30) if mode == "RGB" else 120
    next_result = geometry(Image.new(mode, image.size, next_background))
    assert next_result.getpixel((0, 0)) == next_background


def _stamp_case(background=(245, 245, 245)):
    image = Image.new("RGB", (240, 80), background)
    draw = ImageDraw.Draw(image)
    for x in range(12, 220, 14):
        draw.rectangle((x, 29, x + 7, 50), fill=(35, 35, 35))
    draw.ellipse((74, 7, 145, 72), outline=(220, 35, 45), width=3)
    return image


def test_remove_red_seal_removes_outline_without_touching_other_pixels():
    image = _stamp_case()
    # 同一张盖章图中的合法红字必须保留，不能判出一个章就清空全图红色。
    ImageDraw.Draw(image).text((185, 32), "RED", fill=(220, 35, 45))
    before = np.asarray(image).copy()

    result = remove_red_seal(image)
    after = np.asarray(result)

    assert result.mode == "RGB" and result.size == image.size
    assert np.any(after != before)
    assert np.array_equal(after[:, 170:], before[:, 170:])
    neutral = before.max(axis=2) - before.min(axis=2) < 35
    assert np.array_equal(after[neutral], before[neutral])
    assert tuple(after[7, 109]) == (245, 245, 245)
    assert np.array_equal(np.asarray(image), before)  # 不修改调用方图片。


def test_remove_red_seal_handles_a_cropped_arc_and_preserves_black_strokes():
    image = Image.new("RGB", (360, 48), "white")
    draw = ImageDraw.Draw(image)
    for x in range(8, 350, 12):
        draw.rectangle((x, 13, x + 6, 34), fill="black")
    draw.ellipse((100, 3, 260, 140), outline=(220, 35, 45), width=3)
    before = np.asarray(image)
    after = np.asarray(remove_red_seal(image))
    dark = before.max(axis=2) < 140

    assert np.any(after != before)
    assert np.array_equal(after[dark], before[dark])


@pytest.mark.parametrize("background,ink,stamp", [
    ((255, 255, 255), (35, 35, 35), (220, 35, 45)),
    ((205, 210, 220), (155, 155, 155), (240, 195, 200)),
])
def test_remove_red_seal_handles_disconnected_stamp_text_on_scanned_paper(background, ink, stamp):
    image = Image.new("RGB", (320, 48), background)
    draw = ImageDraw.Draw(image)
    # 只有错位章字，没有圆弧；对应紧框内只剩文字笔画、且扫描正文偏灰的情况。
    for x in (90, 120, 150, 180):
        draw.line([(x, 0), (x + 15, 10), (x - 5, 28)], fill=stamp, width=3)
    for x in range(8, 310, 12):
        draw.rectangle((x, 12, x + 3, 34), fill=ink)
    before = np.asarray(image)
    after = np.asarray(remove_red_seal(image))
    stamp_pixels = np.all(before == stamp, axis=2)
    ink_pixels = np.all(before == ink, axis=2)

    # 碎章进入正文行内的小片段可保留，但主体应被去除。
    assert np.mean(np.any(after[stamp_pixels] != before[stamp_pixels], axis=1)) > 0.5
    assert np.array_equal(after[ink_pixels], before[ink_pixels])


def test_remove_red_seal_handles_dense_star_with_more_red_than_black():
    image = Image.new("RGB", (240, 48), "white")
    draw = ImageDraw.Draw(image)
    points = [
        (120 + (54 if i % 2 == 0 else 23) * np.cos(-np.pi / 2 + i * np.pi / 5),
         25 + (54 if i % 2 == 0 else 23) * np.sin(-np.pi / 2 + i * np.pi / 5))
        for i in range(10)
    ]
    draw.polygon(points, fill=(220, 35, 45))
    for x in range(6, 230, 12):
        draw.rectangle((x, 12, x + 1, 34), fill=(35, 35, 35))
    before = np.asarray(image)
    red = np.all(before == (220, 35, 45), axis=2)
    dark = before.max(axis=2) < 140
    assert red.sum() > dark.sum()

    after = np.asarray(remove_red_seal(image))
    assert np.mean(np.any(after[red] != before[red], axis=1)) > 0.75
    assert np.array_equal(after[dark], before[dark])


def test_remove_red_seal_preserves_adjacent_red_text_in_the_same_region():
    image = _stamp_case()
    draw = ImageDraw.Draw(image)
    draw.rectangle((147, 0, 239, 79), fill=(245, 245, 245))
    draw.text((148, 30), "RED", fill=(220, 35, 45), font=ImageFont.load_default(size=22))
    before = np.asarray(image)
    after = np.asarray(remove_red_seal(image))

    assert np.any(after[:, :147] != before[:, :147])
    assert np.array_equal(after[:, 147:], before[:, 147:])


def test_remove_red_seal_preserves_aligned_red_text_with_black_outline():
    image = Image.new("RGB", (240, 48), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    draw.text((5, 10), "BLACK", fill="black", font=font)
    draw.text((110, 10), "RED", fill="red", font=font, stroke_width=1, stroke_fill="black")

    assert np.array_equal(np.asarray(remove_red_seal(image)), np.asarray(image))


@pytest.mark.parametrize("case", [
    "red_text", "mixed_text", "red_background", "blue_background",
    "no_dark_text", "brown_outline", "orange_outline", "rectangle", "tall_red_text",
])
def test_remove_red_seal_preserves_normal_or_uncertain_content(case):
    if case == "no_dark_text":
        image = Image.new("RGB", (240, 80), "white")
        ImageDraw.Draw(image).ellipse((74, 7, 145, 72), outline="red", width=3)
    elif case in ("red_background", "blue_background"):
        image = _stamp_case((220, 30, 30) if case == "red_background" else (30, 80, 210))
    else:
        image = Image.new("RGB", (240, 80), "white")
        draw = ImageDraw.Draw(image)
        for x in range(12, 220, 14):
            color = "red" if case == "red_text" or (case == "mixed_text" and x > 170) else "black"
            draw.rectangle((x, 29, x + 7, 50), fill=color)
        if case in ("brown_outline", "orange_outline"):
            color = (130, 65, 35) if case == "brown_outline" else (240, 130, 20)
            draw.ellipse((74, 7, 145, 72), outline=color, width=3)
        elif case == "rectangle":
            draw.rectangle((74, 7, 145, 72), outline="red", width=3)
        elif case == "tall_red_text":
            draw.line([(85, 72), (108, 7), (131, 72)], fill="red", width=3)
            draw.line([(96, 45), (120, 45)], fill="red", width=3)

    assert np.array_equal(np.asarray(remove_red_seal(image)), np.asarray(image))


@pytest.mark.parametrize("num_channels", [1, 3])
def test_seal_preprocess_is_shared_by_training_validation_and_inference(tmp_path, num_channels):
    image = _stamp_case()
    image_path = tmp_path / "stamp.png"
    image.save(image_path)
    annotation_path = tmp_path / "annotations.txt"
    annotation_path.write_text("stamp.png\t正文\n", encoding="utf-8")
    expected = remove_red_seal(image)
    assert not np.array_equal(np.asarray(expected), np.asarray(image))

    # 拦截增强入口，验证训练也在原始 RGB 尺寸完成去章，且发生在随机增强之前。
    for is_train in (True, False):
        dataset = TextRecognitionDataset(annotation_path, 80, 240, num_channels, is_train=is_train)
        dataset.transform = lambda processed: np.asarray(processed)
        assert np.array_equal(dataset[0][0], np.asarray(expected))

    validation = TextRecognitionDataset(annotation_path, 32, 240, num_channels, is_train=False)
    inference = InferenceDataset([image_path], 32, num_channels, img_width=240)
    assert torch.equal(validation[0][0], inference[0][0])
    assert inference[0][0].shape == (num_channels, 32, 96)


def _write_detection_case(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "menu-item-current_000.png"
    annotation_path = tmp_path / "annotations.txt"

    image = Image.new("RGB", (180, 80), (20, 110, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 24, 119, 55), fill=(255, 255, 255))
    image.save(image_path)

    annotation_path.write_text(
        "menu-item-current_000.png\t首页274\n",
        encoding="utf-8",
    )
    return image_path, annotation_path


def test_training_dataset_reads_png_and_keeps_the_full_detection_box(tmp_path):
    image_path, annotation_path = _write_detection_case(tmp_path)

    dataset = TextRecognitionDataset(
        annotation_path=str(annotation_path),
        img_height=32,
        img_width=1800,
        num_channels=3,
        is_train=False,
    )

    image_tensor, label = dataset[0]

    assert dataset.image_paths == [str(image_path)]
    assert label == "首页274"
    assert image_tensor.shape == (3, 32, 72)  # 180 / 80 * 32，无二次裁剪


def test_training_dataset_combines_multiple_annotation_files(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_image = first_dir / "a.png"
    second_image = second_dir / "b.png"
    Image.new("RGB", (80, 32), "white").save(first_image)
    Image.new("RGB", (96, 32), "black").save(second_image)

    first_annotation = first_dir / "train.txt"
    second_annotation = second_dir / "append.txt"
    first_annotation.write_text("a.png\t甲\n", encoding="utf-8")
    second_annotation.write_text("b.png\t乙\n", encoding="utf-8")

    dataset = TextRecognitionDataset(
        annotation_path=[first_annotation, second_annotation],
        img_height=32,
        img_width=1800,
        num_channels=3,
        is_train=False,
    )

    assert dataset.image_paths == [str(first_image), str(second_image)]
    assert dataset.labels == ["甲", "乙"]


def test_rgb_preprocess_preserves_red_and_blue_channel_information(tmp_path):
    image_path = tmp_path / "colored-text.png"
    image = Image.new("RGB", (96, 32), (10, 80, 210))
    ImageDraw.Draw(image).rectangle((32, 8, 63, 23), fill=(235, 20, 30))
    image.save(image_path)

    dataset = InferenceDataset([str(image_path)], img_height=32, num_channels=3)
    tensor, returned_path = dataset[0]

    background = tensor[:, 16, 10]
    red_text = tensor[:, 16, 45]
    assert returned_path == str(image_path)
    assert background[2] > background[0]
    assert red_text[0] > red_text[2]


def test_transparent_png_is_composited_to_keep_dark_content_visible(tmp_path):
    image_path = tmp_path / "dark-on-transparent.png"
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((10, 5, 29, 14), fill=(0, 0, 0, 255))
    image.save(image_path)

    loaded = load_rgb_image(image_path)

    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((15, 10)) == (0, 0, 0)


def test_transparent_png_is_composited_to_keep_light_content_visible(tmp_path):
    image_path = tmp_path / "light-on-transparent.png"
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((10, 5, 29, 14), fill=(255, 255, 255, 255))
    image.save(image_path)

    loaded = load_rgb_image(image_path)

    assert loaded.getpixel((0, 0)) == (0, 0, 0)
    assert loaded.getpixel((15, 10)) == (255, 255, 255)


def test_dynamic_padding_uses_each_images_own_edge_color():
    blue = torch.tensor([-0.9, -0.3, 0.8]).view(3, 1, 1).expand(3, 32, 5).clone()
    red = torch.tensor([0.8, -0.8, -0.7]).view(3, 1, 1).expand(3, 32, 9).clone()

    padded, paths, widths = dynamic_padding_collate_fn([(blue, "blue"), (red, "red")])

    assert paths == ("blue", "red")
    assert widths.tolist() == [5, 9]
    expected_width = ((9 + CNN_RIGHT_CONTEXT + 7) // 8) * 8
    assert padded.shape == (2, 3, 32, expected_width)
    assert torch.allclose(padded[0, :, :, 5:], blue[:, :, -1:])
    assert torch.allclose(padded[1, :, :, 9:], red[:, :, -1:])


def test_evaluation_dataset_uses_same_no_crop_rgb_preprocess(tmp_path):
    image_path, _ = _write_detection_case(tmp_path)

    dataset = evaluate_val.create_evaluation_dataset([str(image_path)], num_channels=3)
    image_tensor, returned_path = dataset[0]

    assert returned_path == str(image_path)
    assert image_tensor.shape == (3, 32, 72)


def test_checkpoint_channel_detection_supports_old_grayscale_and_new_rgb_weights():
    assert infer_num_channels({"cnn.stem.0.weight": torch.empty(32, 1, 3, 3)}) == 1
    assert infer_num_channels({"cnn.stem.0.weight": torch.empty(32, 3, 3, 3)}) == 3


def test_checkpoint_layer_detection_counts_gru_layers_and_keeps_fallback():
    assert infer_num_layers({
        "rnn.weight_ih_l0": torch.empty(3, 3),
        "rnn.weight_ih_l0_reverse": torch.empty(3, 3),
    }) == 1
    # 反向键不参与计数：层数只由去掉 _reverse 后的最大层号加一决定。
    assert infer_num_layers({
        "rnn.weight_ih_l0": torch.empty(3, 3),
        "rnn.weight_ih_l0_reverse": torch.empty(3, 3),
        "rnn.weight_ih_l1": torch.empty(3, 6),
        "rnn.weight_ih_l1_reverse": torch.empty(3, 6),
    }) == 2
    # 没有递归权重的模型使用调用方给出的默认层数，无法推断时报错。
    assert infer_num_layers({"cnn.stem.0.weight": torch.empty(32, 3, 3, 3)}, default=2) == 2
    with pytest.raises(ValueError, match="BiGRU 层数"):
        infer_num_layers({"cnn.stem.0.weight": torch.empty(32, 3, 3, 3)})


def test_resolve_image_path_requires_the_exact_annotated_extension(tmp_path):
    png_path = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), "white").save(png_path)
    missing_jpg = png_path.with_suffix(".jpg")

    try:
        resolve_image_path(missing_jpg)
    except FileNotFoundError as exc:
        assert str(missing_jpg) in str(exc)
    else:
        raise AssertionError("图片扩展名不一致时应立即失败")
