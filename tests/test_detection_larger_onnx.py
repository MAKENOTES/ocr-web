"""验证大图 ONNX 导出、动态 batch、后端一致性和独立推理入口。"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detection_larger_export_onnx as exporter
from detection_larger.inference import DocumentDetector
from detection_larger.models.east_model import create_model, save_checkpoint


@pytest.mark.parametrize('fallback', [False, True])
def test_export_and_inference_roundtrip(tmp_path, monkeypatch, fallback):
    import onnx

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(exporter, 'BENCHMARK_WARMUP', 1)
        monkeypatch.setattr(exporter, 'BENCHMARK_RUNS', 2)
        model = create_model().eval()
        checkpoint, output = tmp_path / 'model.pth', tmp_path / 'model.onnx'
        save_checkpoint(model, checkpoint, (64, 96))
        original = torch.onnx.export
        exporters = []

        def track_export(*args, **kwargs):
            exporters.append(kwargs['dynamo'])
            if fallback and kwargs.get('dynamo'):
                raise RuntimeError('test exporter fallback')
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.onnx, 'export', track_export)
        config = exporter.export_model(checkpoint, output)
        assert exporters == ([True, False] if fallback else [True])
        detector = DocumentDetector(output)
        assert detector.config == config
        assert detector.session.get_inputs()[0].shape == ['batch', 3, 64, 96]
        graph = onnx.load(output)
        assert all(not tensor.external_data for tensor in graph.graph.initializer)
        image = Image.new('RGB', (180, 70), 'white')
        expected = DocumentDetector(checkpoint).predict(image, threshold=.1)
        observed = detector.predict(image, threshold=.1)
        np.testing.assert_allclose([p['box'] for p in observed], [p['box'] for p in expected], atol=.001)
        np.testing.assert_allclose([p['score'] for p in observed], [p['score'] for p in expected], atol=1e-5)
        if not fallback:
            ov_detector = DocumentDetector(output, backend='openvino')
            ov_detector.warmup(1)
            sample = torch.rand(1, 3, 64, 96)
            for expected_map, observed_map in zip(detector.infer(sample), ov_detector.infer(sample)):
                np.testing.assert_allclose(observed_map, expected_map, rtol=1e-4, atol=1e-5)
            ov_predictions = ov_detector.predict(image, threshold=.1)
            np.testing.assert_allclose([p['box'] for p in ov_predictions], [p['box'] for p in observed], atol=.001)
            assert ov_detector.last_infer_ms > 0
            check_quantized_inference(output, tmp_path / 'int8.onnx')
        with pytest.raises(FileExistsError, match=str(output)):
            exporter.export_model(checkpoint, output)
        graph.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 64
        onnx.save(graph, output)
        with pytest.raises(ValueError, match='Incompatible document ONNX inputs/outputs:.*model.onnx'):
            DocumentDetector(output)
        del graph.metadata_props[:]
        onnx.save(graph, output)
        with pytest.raises(ValueError, match='Invalid document ONNX config:.*model.onnx'):
            DocumentDetector(output)
    finally:
        torch.set_num_threads(previous)


def test_failed_export_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'load_checkpoint', lambda _: (object(), {'config': {'input_size': [64, 64]}}))

    def fail(*args):
        raise RuntimeError('export failed')

    monkeypatch.setattr(exporter, 'export_graph', fail)
    output = tmp_path / 'model.onnx'
    with pytest.raises(RuntimeError, match='export failed'):
        exporter.export_model('unused.pth', output)
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_onnx_entry_saves_images_labels_and_crops(tmp_path, monkeypatch):
    import detection_larger_inference_onnx as entry
    from detection_larger import inference

    image = tmp_path / 'page.png'
    Image.new('RGBA', (40, 30), (0, 0, 0, 0)).save(image)
    monkeypatch.setattr(entry, 'IMAGE_PATH', image)
    monkeypatch.setattr(entry, 'OUTPUT_DIR', tmp_path / 'results')
    monkeypatch.setattr(entry, 'CROP_DIR', tmp_path / 'crops')

    class Detector:
        def __init__(self, path, device, backend="auto"):
            self.backend, self.last_infer_ms = backend, 1.0
            assert Path(path).suffix == '.onnx' and device == 'cpu'

        def warmup(self, count):
            assert count == entry.WARMUP_RUNS

        def predict(self, image, threshold, iou):
            assert image.mode == 'RGB'
            return [{'box': [2, 3, 20, 15], 'score': .9}]

    monkeypatch.setattr(inference, 'DocumentDetector', Detector)
    entry.main()
    assert (tmp_path / 'results/images/page.png').exists()
    assert (tmp_path / 'results/labels/gt_page.txt').read_text() == '2.0 3.0 20.0 15.0\n'
    with Image.open(tmp_path / 'crops/page-00001.png') as crop:
        assert crop.size == (18, 12)



def check_quantized_inference(model_path, output):
    """用少量合成输入仅验证 QDQ INT8 可运行，不将其作为精度校准结论。"""
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.samples = iter([{'image': np.zeros((1, 3, 64, 96), dtype=np.float32)}])

        def get_next(self):
            return next(self.samples, None)

    quantize_static(str(model_path), str(output), Reader(),
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=['Conv'])
    detector = DocumentDetector(output)
    score, geometry = detector.infer(torch.zeros(1, 3, 64, 96))
    assert score.shape == (1, 1, 16, 24)
    assert geometry.shape == (1, 4, 16, 24)
    assert np.isfinite(score).all() and np.isfinite(geometry).all()


@pytest.mark.parametrize('backend,path', [('unknown', 'model.onnx'), ('onnx', 'model.pth'),
                                          ('pytorch', 'model.onnx'), ('auto', 'model.bin')])
def test_invalid_backend_or_extension(backend, path):
    with pytest.raises(ValueError, match='backend|requires'):
        DocumentDetector(path, backend=backend)


def test_expansion_clips_and_preserves_original():
    from detection_larger.inference import expand_predictions

    original = [{'box': [1, 2, 19, 12], 'score': .9}]
    result = expand_predictions(original, (20, 15), .2, .5)
    assert result == [{'box': [0, 0, 20, 15], 'score': .9}]
    assert original[0]['box'] == [1, 2, 19, 12]
    assert expand_predictions([], (20, 15), .1, .1) == []
    for ratio in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='ratios'):
            expand_predictions(original, (20, 15), ratio, 0)


def test_warmup_and_forward_timing():
    from detection_larger.models.east_model import checkpoint_config

    detector = DocumentDetector.__new__(DocumentDetector)
    detector.config = checkpoint_config(64)
    detector.session, detector.openvino, detector.device = None, None, 'cpu'
    calls = []

    def model(batch):
        calls.append(tuple(batch.shape))
        return torch.zeros(1, 1, 16, 16), torch.zeros(1, 4, 16, 16)

    detector.model = model
    detector.warmup(0)
    assert calls == []
    detector.warmup(3)
    assert calls == [(1, 3, 64, 64)] * 3
    assert detector.last_infer_ms >= 0
    with pytest.raises(ValueError, match='warmup'):
        detector.warmup(-1)


def test_run_summary_and_expanded_outputs(tmp_path, monkeypatch):
    from detection_larger import inference

    image = tmp_path / 'page.png'
    Image.new('RGB', (20, 15), 'white').save(image)

    class Detector:
        backend = 'onnx'
        last_infer_ms = 2.5

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args):
            return [{'box': [1, 2, 19, 12], 'score': .9}]

    monkeypatch.setattr(inference, 'DocumentDetector', Detector)
    summary = inference.run_inference('model.onnx', image, tmp_path / 'output',
                                      save_crops=True, crop_dir=tmp_path / 'crops',
                                      expand_ratio_w=.2, expand_ratio_h=.5)
    assert summary['images'] == summary['boxes'] == 1
    assert summary['mean_infer_ms'] == 2.5
    assert summary['wall_seconds'] > 0 and summary['images_per_second'] > 0
    assert (tmp_path / 'output/labels/gt_page.txt').read_text() == '0.0 0.0 20.0 15.0\n'
    with Image.open(tmp_path / 'crops/page-00001.png') as crop:
        assert crop.size == (20, 15)
