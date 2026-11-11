"""验证大图 ONNX 导出、动态 batch、后端一致性和独立推理入口。"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detection_larger_export_onnx as exporter
from detection_larger.inference import DocumentDetector
from detection_larger.models.east_model import create_model
from detection_larger.utils import save_checkpoint


@pytest.mark.parametrize('fallback', [False, True])
def test_export_and_inference_roundtrip(tmp_path, monkeypatch, fallback):
    import onnx
    from detection_larger import inference

    monkeypatch.setattr(inference, "INPUT_SIZE", (64, 96))
    monkeypatch.setattr(exporter, "INPUT_SIZE", (64, 96))
    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", not fallback)

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(exporter, 'BENCHMARK_WARMUP', 1)
        monkeypatch.setattr(exporter, 'BENCHMARK_RUNS', 2)
        model = create_model().eval()
        checkpoint, output = tmp_path / 'model.pth', tmp_path / 'model.onnx'
        save_checkpoint(model, checkpoint, 64)
        original = torch.onnx.export
        exporters = []

        def track_export(*args, **kwargs):
            exporters.append(kwargs['dynamo'])
            if fallback and kwargs.get('dynamo'):
                raise RuntimeError('test exporter fallback')
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.onnx, 'export', track_export)
        config = exporter.export_model(checkpoint, output)
        assert config["input_size"] == [64, 96]
        assert exporters == ([True, False] if fallback else [True])
        detector = DocumentDetector(output)
        assert detector.input_size == (64, 96)
        assert not hasattr(detector, "config")
        assert detector.session.get_inputs()[0].shape == ['batch', 3, 64, 96]
        graph = onnx.load(output)
        import json
        assert json.loads({p.key: p.value for p in graph.metadata_props}[exporter.ONNX_CONFIG_KEY]) == config
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
        monkeypatch.setattr(inference, "INPUT_SIZE", 64)
        # 固定高宽导出自动使用导出尺寸，模块推理常量不再参与校验。
        for backend in ("onnx", "openvino"):
            fixed = DocumentDetector(output, backend=backend)
            assert fixed.input_size == (64, 96)
            assert fixed.predict(image, threshold=.1)
        monkeypatch.setattr(inference, "INPUT_SIZE", (64, 96))
        # 导出目标已存在时直接覆盖，校验通过的模型替换旧文件。
        assert exporter.export_model(checkpoint, output) == config
        graph = onnx.load(output)
        graph.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 64
        onnx.save(graph, output)
        with pytest.raises(ValueError, match='does not match config.*model.onnx'):
            DocumentDetector(output)
        graph = onnx.load(output)
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


def test_onnx_entry_wires_runtime_cli(tmp_path, monkeypatch):
    """根目录入口只做常量到 runtime 命令行的转换。"""
    import detection_larger_inference_onnx as entry
    from detection_larger.runtime import cli

    image = tmp_path / 'page.png'
    Image.new('RGBA', (40, 30), (0, 0, 0, 0)).save(image)
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'onnn')

    class EmptyBackend:
        config = {'input_size': [64, 64], 'dynamic_spatial': False}
        static_input_size = (64, 64)

        def infer(self, batch):
            return (np.zeros((1, 1, 16, 16), dtype=np.float32),
                    np.zeros((1, 4, 16, 16), dtype=np.float32))

    monkeypatch.setattr(cli, 'load_backend', lambda path, backend: EmptyBackend())
    monkeypatch.setattr(cli, 'exported_shape',
                        lambda backend: (False, (64, 64), (64, 64)))
    entry.main(['--model_path', str(model), '--image_path', str(image),
                '--output_dir', str(tmp_path / 'results'), '--backend', 'onnx',
                '--warmup_runs', '0'])
    assert (tmp_path / 'results/images/page.png').exists()
    assert (tmp_path / 'results/labels/gt_page.txt').read_text() == ''


def test_runtime_engine_saves_images_labels_and_crops(tmp_path):
    """runtime 推理流水线处理透明图、bbox4 标签和裁片命名。"""
    from detection_larger.runtime.engine import run_inference

    image = tmp_path / 'page.png'
    Image.new('RGBA', (40, 30), (0, 0, 0, 0)).save(image)

    class BoxBackend:
        config = {'input_size': [64, 64], 'dynamic_spatial': False}
        static_input_size = (64, 64)

        def infer(self, batch):
            score = np.zeros((1, 1, 16, 16), dtype=np.float32)
            score[0, 0, 4:6, 4:8] = .9
            geometry = np.full((1, 4, 16, 16), np.log1p(3), dtype=np.float32)
            return score, geometry

    summary = run_inference(
        BoxBackend(), image, tmp_path / 'results', dynamic=False,
        max_input_size=(64, 64), fixed_input_size=(64, 64),
        threshold=.7, iou_threshold=.3, save_crops=True,
        crop_dir=tmp_path / 'crops')
    assert summary['images'] == 1 and summary['boxes'] == 1
    with Image.open(tmp_path / 'results/images/page.png') as preview:
        assert preview.mode == 'RGB' and preview.size == (40, 30)
    label = (tmp_path / 'results/labels/gt_page.txt').read_text().split()
    assert len(label) == 4
    assert (tmp_path / 'crops/page-00001.png').is_file()



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
    detector = DocumentDetector.__new__(DocumentDetector)
    detector.input_size = (64, 64)
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


def test_export_entry_uses_path_constants(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / 'source.pth'
    output = tmp_path / 'export.onnx'
    monkeypatch.setattr(exporter, 'CHECKPOINT_PATH', checkpoint)
    monkeypatch.setattr(exporter, 'OUTPUT_PATH', output)
    calls = []

    def export_model(checkpoint_path, output_path):
        calls.append((checkpoint_path, output_path))
        return {'input_size': [64, 96]}

    monkeypatch.setattr(exporter, 'export_model', export_model)
    exporter.main()
    assert calls == [(checkpoint, output)]
    assert f'导出及数值校验通过: {output}' in capsys.readouterr().out


def test_export_rejects_invalid_input_size(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'INPUT_SIZE', 65)
    output = tmp_path / 'invalid.onnx'
    with pytest.raises(ValueError, match='input_size'):
        exporter.export_model(tmp_path / 'unused.pth', output)
    assert not output.exists()


def test_runtime_is_self_contained_and_synced():
    """runtime 快照不能导入 detection_larger 包内模块，且必须与源文件同步。"""
    import ast

    from tools.sync_detection_larger_runtime import check_runtime

    assert check_runtime() == []
    runtime_dir = Path(__file__).resolve().parents[1] / "detection_larger" / "runtime"
    for path in sorted(runtime_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not (node.level == 0 and (
                    module == "detection_larger"
                    or module.startswith("detection_larger."))), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name == "detection_larger"
                               or alias.name.startswith("detection_larger.")
                               for alias in node.names), path


def test_build_bundle_copies_whitelist_and_verifies(tmp_path, monkeypatch):
    """部署包只带 runtime、入口和模型，并能在隔离路径下完成一次推理。"""
    import json

    from detection_larger.utils import MODEL_FORMAT
    from tools.build_detection_larger_bundle import build_bundle

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model().eval()
        checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
        save_checkpoint(model, checkpoint, 64)
        exporter.export_model(checkpoint, output, input_size=(64, 64), dynamic_spatial=False)

        bundle = tmp_path / "bundle"
        manifest = build_bundle(output, bundle, verify=True)
        assert (bundle / "detection_larger_inference_onnx.py").is_file()
        assert (bundle / "detection_larger/runtime/engine.py").is_file()
        assert manifest["model"]["file"] == "models/model.onnx"
        assert json.loads(manifest["model"]["config"])["format"] == MODEL_FORMAT
        assert not (bundle / "detection_larger/components").exists()
        assert not (bundle / "detection_larger/utils").exists()
    finally:
        torch.set_num_threads(previous)


def test_dynamic_spatial_export_uses_metadata_size_limit(tmp_path, monkeypatch):
    """动态导出按原图宽高比选尺寸，推理上限来自 ONNX 元数据。"""
    from detection_larger import runtime

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model().eval()
        checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
        save_checkpoint(model, checkpoint, 64)
        config = exporter.export_model(
            checkpoint, output, input_size=(64, 96), dynamic_spatial=True)
        assert config["dynamic_spatial"] is True

        backend = runtime.load_backend(output, "onnx")
        assert runtime.exported_shape(backend) == (True, (64, 96), None)
        detector = DocumentDetector(output)
        assert detector.dynamic_input is True
        assert detector.input_size == (64, 96)
        # 180×70 页面按宽高比缩放到不超过 64×96 的 32 倍数画布。
        assert isinstance(detector.predict(Image.new("RGB", (180, 70), "white")), list)
    finally:
        torch.set_num_threads(previous)


def test_inference_entry_is_thin_runtime_wrapper():
    """根目录 ONNX 入口只保留常量和 main，不再复制推理实现。"""
    import ast

    source = (Path(__file__).resolve().parents[1] / "detection_larger_inference_onnx.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] == ["main"]
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert "detection_larger.runtime.cli" in modules
    assert all(module in {"sys", "pathlib", "detection_larger.runtime.cli"}
               for module in modules), modules
