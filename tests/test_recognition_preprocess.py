from pathlib import Path

import torch
from PIL import Image, ImageDraw

from recognition import evaluate_val
from recognition.components.dataset import (
    CNN_RIGHT_CONTEXT,
    InferenceDataset,
    TextRecognitionDataset,
    dynamic_padding_collate_fn,
    load_rgb_image,
    resolve_image_path,
)
from recognition.utils.checkpoint import infer_num_channels


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
