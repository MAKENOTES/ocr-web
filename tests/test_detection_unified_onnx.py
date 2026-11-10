"""验证统一检测 ONNX 动态/静态导出、后端一致性和根目录入口常量。"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detection_export_onnx as exporter
import detection_inference_onnx as inference_entry
from detection.components.inference_utils import suppress_boxes as shared_suppress_boxes
from detection.inference import DocumentDetector
from detection.models.east_model import create_model
from detection.utils import ONNX_CONFIG_KEY, save_checkpoint


@pytest.mark.parametrize("dynamic_spatial", [True, False])
def test_export_and_detector_roundtrip(tmp_path, monkeypatch, dynamic_spatial):
    import onnx

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model().eval()
        checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
        save_checkpoint(model, checkpoint, (64, 96))
        # 预置旧文件，验证导出会直接覆盖同名目标而不是报错。
        output.write_bytes(b"stale-onnx")
        config = exporter.export_model(checkpoint, output, input_size=(64, 96),
                                       dynamic_spatial=dynamic_spatial)
        assert config["input_size"] == [64, 96]
        assert config["dynamic_spatial"] is dynamic_spatial

        graph = onnx.load(output)
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        assert json.loads(properties[ONNX_CONFIG_KEY]) == config
        assert all(not tensor.external_data for tensor in graph.graph.initializer)

        detector = DocumentDetector(output)
        assert detector.backend == "onnx"
        assert detector.dynamic_input is dynamic_spatial
        assert detector.input_size == (64, 96)

        # 使用确定性噪声图，避免纯白图在所有后端都输出空框、对比失去意义。
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (70, 180, 3), dtype=np.uint8))
        expected = DocumentDetector(checkpoint, max_input_size=(64, 96),
                                    dynamic_input=dynamic_spatial).predict(image, threshold=.1)
        observed = detector.predict(image, threshold=.1)
        np.testing.assert_allclose([value["box"] for value in observed],
                                   [value["box"] for value in expected], atol=.001)
        np.testing.assert_allclose([value["score"] for value in observed],
                                   [value["score"] for value in expected], atol=1e-5)

        # 独立 ONNX 入口应与共享 DocumentDetector 输出一致，证明未依赖 GPU 推理模块。
        backend = inference_entry.load_backend(output, "onnx")
        dynamic, max_input_size, fixed_input_size = inference_entry.exported_shape(backend)
        standalone, _ = inference_entry.predict_image(
            backend, image, .1, .3, dynamic, max_input_size, fixed_input_size)
        np.testing.assert_allclose([value["box"] for value in standalone],
                                   [value["box"] for value in observed], atol=.001)
        np.testing.assert_allclose([value["score"] for value in standalone],
                                   [value["score"] for value in observed], atol=1e-5)

        ov_detector = DocumentDetector(output, backend="openvino")
        ov_detector.warmup(1)
        sample = torch.rand(1, 3, 64, 96)
        for expected_map, observed_map in zip(detector.infer(sample), ov_detector.infer(sample)):
            np.testing.assert_allclose(observed_map, expected_map, rtol=1e-4, atol=1e-5)
        ov_predictions = ov_detector.predict(image, threshold=.1)
        np.testing.assert_allclose([value["box"] for value in ov_predictions],
                                   [value["box"] for value in observed], atol=.001)
    finally:
        torch.set_num_threads(previous)


def test_detector_rejects_dynamic_flag_mismatch(tmp_path, monkeypatch):
    import onnx

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    torch.set_num_threads(1)
    model = create_model().eval()
    checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
    save_checkpoint(model, checkpoint, (64, 96))
    exporter.export_model(checkpoint, output, input_size=(64, 96), dynamic_spatial=False)

    graph = onnx.load(output)
    properties = {entry.key: entry.value for entry in graph.metadata_props}
    config = json.loads(properties[ONNX_CONFIG_KEY])
    config["dynamic_spatial"] = True
    properties[ONNX_CONFIG_KEY] = json.dumps(config)
    onnx.helper.set_model_props(graph, properties)
    onnx.save(graph, output)
    with pytest.raises(ValueError, match="dynamic_spatial"):
        DocumentDetector(output)


def test_failed_export_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "load_checkpoint",
                        lambda _: (object(), {"config": {"input_size": [64, 64]}}))

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(exporter, "export_graph", fail)
    output = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError, match="export failed"):
        exporter.export_model("unused.pth", output)
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_export_entry_uses_path_constants(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "source.pth"
    output = tmp_path / "export.onnx"
    monkeypatch.setattr(exporter, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(exporter, "OUTPUT_PATH", output)
    calls = []

    def fake_export(checkpoint_path, output_path):
        calls.append((checkpoint_path, output_path))
        return {"input_size": [64, 96]}

    monkeypatch.setattr(exporter, "export_model", fake_export)
    exporter.main()
    assert calls == [(checkpoint, output)]
    assert f"导出及数值校验通过: {output}" in capsys.readouterr().out


def test_onnx_inference_entry_parses_overrides(tmp_path):
    """命令行覆盖应只影响本次运行，默认值来自入口常量。"""
    args = inference_entry.build_parser().parse_args([
        "--model_path", str(tmp_path / "model.onnx"),
        "--image_path", str(tmp_path / "images"),
        "--output_dir", str(tmp_path / "output"),
        "--backend", "onnx",
        "--score_thr", "0.6",
        "--nms_iou", "0.4",
        "--save_crops",
        "--crop_dir", str(tmp_path / "crops"),
        "--warmup_runs", "0",
    ])
    assert args.model_path == tmp_path / "model.onnx"
    assert args.image_path == tmp_path / "images"
    assert args.output_dir == tmp_path / "output"
    assert args.backend == "onnx"
    assert args.score_thr == 0.6
    assert args.nms_iou == 0.4
    assert args.save_crops is True
    assert args.crop_dir == tmp_path / "crops"
    assert args.warmup_runs == 0


def test_onnx_inference_entry_rejects_missing_model(tmp_path):
    """模型路径不存在时应在加载后端前退出。"""
    with pytest.raises(SystemExit):
        inference_entry.main(["--model_path", str(tmp_path / "missing.onnx")])


def test_onnx_inference_entry_is_self_contained():
    """单文件部署入口不能导入 detection 包内的任何模块。"""
    source = (Path(__file__).resolve().parents[1] / "detection_inference_onnx.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(module == "detection" or module.startswith("detection.")
                   for module in modules), modules


def test_onnx_entry_numpy_nms_matches_shared_implementation():
    """单文件入口的 NumPy NMS 应与包内 torchvision NMS 选择相同索引。"""
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30],
                      [21, 21, 31, 31], [40, 40, 50, 50]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(
        inference_entry.suppress_boxes(boxes, scores, 0.3),
        shared_suppress_boxes(boxes, scores, 0.3))


def test_onnx_entry_preprocessing_matches_package(tmp_path):
    """单文件入口的图片读取和预处理应与包内实现一致。"""
    from detection.components.image import load_rgb_image as package_loader
    from detection.components.transforms import prepare_image as package_prepare

    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (70, 180, 3), dtype=np.uint8))
    tensor, metadata = inference_entry.prepare_image(image, (64, 96))
    expected, expected_metadata = package_prepare(image, (64, 96), dynamic=False)
    assert metadata == expected_metadata
    np.testing.assert_allclose(tensor, expected.numpy(), rtol=1e-6, atol=1e-6)

    rgba = Image.new("RGBA", (16, 12), (255, 0, 0, 0))
    transparent = tmp_path / "alpha.png"
    rgba.save(transparent)
    np.testing.assert_array_equal(
        np.asarray(inference_entry.load_rgb_image(transparent)),
        np.asarray(package_loader(transparent)))


def test_onnx_inference_entry_does_not_reuse_gpu_inference():
    """ONNX 入口只依赖 components，不导入 PyTorch/GPU 推理模块。"""
    source = (Path(__file__).resolve().parents[1] / "detection_inference_onnx.py").read_text(
        encoding="utf-8")
    assert "detection.inference" not in source
    assert "DocumentDetector" not in source
    assert "run_inference" not in source


def test_onnx_inference_entry_does_not_depend_on_exporter():
    """ONNX 推理入口只依赖 detection 包内推理，不引用根目录导出实现。"""
    source = (Path(__file__).resolve().parents[1] / "detection_inference_onnx.py").read_text(
        encoding="utf-8")
    assert "detection_export_onnx" not in source
    assert "export_model" not in source


def test_export_rejects_invalid_input_size(tmp_path):
    output = tmp_path / "invalid.onnx"
    with pytest.raises(ValueError, match="input_size"):
        exporter.export_model(tmp_path / "unused.pth", output, input_size=65)
    assert not output.exists()
