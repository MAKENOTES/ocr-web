import numpy as np
import pytest
import torch
from PIL import Image

from recognition.components.converter import LabelConverter
from recognition.components.dataset import InferenceDataset, dynamic_padding_collate_fn
from recognition.models.model import CRNN
from recognition_export_onnx import OnnxExportWrapper
from recognition_quantize_static import (
    CRNNCalibrationDataset, create_calibration_dataloader, make_validation_fn,
    transform_calibration, quantize_via_openvino_ir,
)
from recognition_benchmark_cpu import select_indices


@pytest.mark.parametrize('channels', [1, 3])
def test_calibration_matches_inference_on_transparent_long_and_dark_images(tmp_path, channels):
    Image.new('RGBA', (510, 32), (0, 0, 0, 0)).save(tmp_path / 'alpha.png')
    Image.new('RGB', (7, 32), (15, 20, 25)).save(tmp_path / 'dark.png')
    Image.new('RGB', (700, 32), (240, 235, 230)).save(tmp_path / 'long.png')
    calibration = CRNNCalibrationDataset(tmp_path, num_channels=channels)
    reference = InferenceDataset(calibration.image_paths, 32, channels)
    for i in range(len(reference)):
        expected_images, _, expected_widths = dynamic_padding_collate_fn([reference[i]])
        image, width = calibration[i]
        np.testing.assert_array_equal(image, expected_images[0].numpy())
        assert width == expected_widths[0].item()
        assert image.shape[-1] >= width + 32
    assert calibration[2][1] == 700  # 禁止旧校准逻辑把长图压缩到 256。
    loader = create_calibration_dataloader(tmp_path, num_channels=channels)
    feeds = transform_calibration(next(iter(loader)))
    assert feeds['input'].shape[:3] == (1, channels, 32)
    assert feeds['img_widths'].shape == (1,)
    assert feeds['img_widths'].dtype == np.int64


def test_logits_export_preserves_decoding_and_padding():
    torch.manual_seed(19)
    # 该导出包装器只支持单层 BiGRU；多层模型由 recognition_export_onnx_layer.py 导出。
    model = CRNN(32, 3, 11, num_layers=1).eval()
    images, _, widths = dynamic_padding_collate_fn([
        (torch.rand(3, 32, 48), 'short'), (torch.rand(3, 32, 153), 'long')])
    with torch.inference_mode():
        expected = model(images, widths)
        logits = OnnxExportWrapper(model, output_type='logits')(images, widths)
    torch.testing.assert_close(logits.log_softmax(2), expected)
    assert torch.equal(logits.argmax(2), expected.argmax(2))
    steps = model._calc_cnn_output_width(48)
    assert logits[steps:, 0].argmax(1).eq(0).all()


def test_accuracy_control_requires_labels_before_conversion():
    with pytest.raises(ValueError, match='validation_annotation'):
        quantize_via_openvino_ir('missing.onnx', 'missing', 'missing', accuracy_control=True)


def test_sampling_is_reproducible_and_does_not_take_filename_prefix():
    first = select_indices(1000, 30)
    assert first == select_indices(1000, 30)
    assert len(set(first)) == 30
    assert max(first) > 900
    assert select_indices(3, 20) == [0, 1, 2]


def test_accuracy_metric_uses_character_weighted_edit_distance(tmp_path):
    keys = tmp_path / 'keys.txt'
    keys.write_text('a\nb\n', encoding='utf-8')
    converter = LabelConverter(str(keys))
    class Port:
        def __init__(self, name, rank):
            self.any_name = name
            self.partial_shape = [1] * rank
    class Compiled:
        inputs = [Port('renamed_image', 4), Port('renamed_widths', 1)]
        def __call__(self, feeds):
            assert set(feeds) == {'renamed_image', 'renamed_widths'}
            out = np.zeros((3, 1, 4), dtype=np.float32)
            out[:, 0, 2] = 1  # decode -> 'a'
            return [out]
    images = torch.zeros(1, 3, 32, 80)
    widths = torch.tensor([48])
    # 总共 4 个字符、2 个编辑距离；不是按整行平均的 CER。
    metric, per_item = make_validation_fn(converter)(Compiled(), [
        (images, ['a'], widths), (images, ['aba'], widths)])
    assert metric == 0.5
    assert per_item is None
