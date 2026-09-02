import numpy as np
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

import detection_export_onnx as export_script
import detection_inference_640 as inference_script
import detection_quantize_int8 as quantize_script
from detection.components.transforms import ResizeAndPad


INPUT_SIZE = (256, 768)


def test_detection_scripts_share_rectangular_input_size():
    for module in (export_script, inference_script, quantize_script):
        assert module.INPUT_SIZE == INPUT_SIZE
        assert module.normalize_input_size(640) == (640, 640)
        assert module.normalize_input_size("256x768") == INPUT_SIZE


def test_inference_and_quantization_preprocess_match_training(tmp_path):
    pixels = np.arange(73 * 211 * 3, dtype=np.uint8).reshape(73, 211, 3)
    image = Image.fromarray(pixels, "RGB")
    image_path = tmp_path / "sample.png"
    image.save(image_path)

    resized = ResizeAndPad(INPUT_SIZE)(
        {"image": image.copy(), "quads": np.array([])}
    )["image"]
    expected = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(F.to_tensor(resized)).numpy()[None]

    inference_input, inference_image, original_size = inference_script.preprocess(
        str(image_path), INPUT_SIZE
    )
    quantization_input = quantize_script.preprocess_image(
        str(image_path), INPUT_SIZE
    )

    assert inference_input.shape == (1, 3, 256, 768)
    assert quantization_input.shape == (1, 3, 256, 768)
    assert inference_image.size == (768, 256)
    assert original_size == (211, 73)
    np.testing.assert_allclose(inference_input, expected, rtol=0, atol=5e-7)
    np.testing.assert_allclose(quantization_input, expected, rtol=0, atol=5e-7)


def test_random_calibration_data_uses_rectangular_shape():
    reader = quantize_script.EASTCalibrationDataReader(
        input_name="input",
        calib_num=1,
        input_size=INPUT_SIZE,
    )

    assert reader.get_next()["input"].shape == (1, 3, 256, 768)
    assert reader.get_next() is None


def test_small_region_threshold_scales_by_rectangular_area():
    assert inference_script.scaled_small_region_area(INPUT_SIZE) == 3
    assert inference_script.scaled_small_region_area((1024, 1024)) == 15
