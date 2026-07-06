from PIL import Image, ImageDraw

from recognition.utils.white_text_crop import auto_crop_white_text_image, save_crop_debug_image


def test_auto_crop_white_text_image_crops_white_text_on_colored_background():
    image = Image.new("RGB", (160, 80), (30, 100, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((58, 28, 101, 50), fill=(255, 255, 255))

    cropped = auto_crop_white_text_image(image, margin_ratio=0.1, min_fg_ratio=0.001)

    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]
    assert cropped.size[0] >= 44
    assert cropped.size[1] >= 23


def test_auto_crop_white_text_image_keeps_image_when_no_white_text_found():
    image = Image.new("RGB", (160, 80), (30, 100, 170))

    cropped = auto_crop_white_text_image(image)

    assert cropped.size == image.size


def test_auto_crop_white_text_image_keeps_image_when_text_already_large():
    image = Image.new("RGB", (120, 60), (30, 100, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 12, 90, 48), fill=(255, 255, 255))

    cropped = auto_crop_white_text_image(image)

    assert cropped.size == image.size


def test_save_crop_debug_image_writes_crop_to_named_directory(tmp_path):
    image = Image.new("RGB", (40, 20), (30, 100, 170))
    source_path = tmp_path / "menu-item-current_000.jpg"
    output_dir = tmp_path / "crops"

    saved_path = save_crop_debug_image(image, str(source_path), str(output_dir))

    assert saved_path == output_dir / "menu-item-current_000.png"
    assert saved_path.exists()
    assert Image.open(saved_path).size == image.size


def test_auto_crop_white_text_image_defaults_to_tuned_x_y_margins():
    image = Image.new("RGB", (100, 60), (30, 100, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 69, 34), fill=(255, 255, 255))

    cropped = auto_crop_white_text_image(
        image,
        max_text_height_ratio=0.9,
    )

    assert cropped.size == (44, 23)
