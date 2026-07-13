import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from recognition import evaluate_val

from recognition.components.dataset import TextRecognitionDataset
from recognition.utils.white_text_crop import WhiteTextCropInferenceDataset


def _write_white_text_case(tmp_dir: Path) -> tuple[Path, Path]:
    image_path = tmp_dir / "menu-item-current_000.jpg"
    annotation_path = tmp_dir / "annotations.txt"

    image = Image.new("RGB", (180, 80), (30, 100, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 30, 110, 48), fill=(255, 255, 255))
    image.save(image_path)

    annotation_path.write_text(f"{image_path.name},首页274\n", encoding="utf-8")
    return image_path, annotation_path


def test_text_recognition_dataset_uses_auto_crop_before_resize():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        image_path, annotation_path = _write_white_text_case(tmp_dir)
        crops_dir = tmp_dir / "train_crops"

        dataset = TextRecognitionDataset(
            annotation_path=str(annotation_path),
            img_height=32,
            img_width=1800,
            num_channels=1,
            is_train=False,
            use_auto_crop=True,
            save_crops_dir=str(crops_dir),
        )

        image_tensor, label = dataset[0]
        saved_crop = crops_dir / f"{image_path.stem}.png"

        assert label == "首页274"
        assert image_tensor.shape[1] == 32
        assert saved_crop.exists()
        assert Image.open(saved_crop).size[0] < Image.open(image_path).size[0]


def test_white_text_crop_inference_dataset_uses_same_auto_crop_path():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        image_path, _ = _write_white_text_case(tmp_dir)
        crops_dir = tmp_dir / "inference_crops"

        dataset = WhiteTextCropInferenceDataset(
            image_paths=[str(image_path)],
            img_height=32,
            num_channels=1,
            save_crops_dir=str(crops_dir),
        )

        image_tensor, path = dataset[0]
        saved_crop = crops_dir / f"{image_path.stem}.png"

        assert path == str(image_path)
        assert image_tensor.shape[1] == 32
        assert saved_crop.exists()
        assert Image.open(saved_crop).size[0] < Image.open(image_path).size[0]


def test_evaluate_val_dataset_uses_unified_auto_crop():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        image_path, _ = _write_white_text_case(tmp_dir)
        crops_dir = tmp_dir / "evaluate_crops"

        dataset = evaluate_val.create_evaluation_dataset(
            image_paths=[str(image_path)],
            save_crops_dir=str(crops_dir),
        )

        _, path = dataset[0]
        saved_crop = crops_dir / f"{image_path.stem}.png"

        assert path == str(image_path)
        assert saved_crop.exists()
        assert Image.open(saved_crop).size[0] < Image.open(image_path).size[0]
