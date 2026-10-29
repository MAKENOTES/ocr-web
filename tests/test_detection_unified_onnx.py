"""验证统一检测 ONNX 动态/静态导出、后端一致性和根目录入口常量。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detection_export_onnx as exporter_entry
from detection import export_onnx as exporter
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

        image = Image.new("RGB", (180, 70), "white")
        expected = DocumentDetector(checkpoint, max_input_size=(64, 96),
                                    dynamic_input=dynamic_spatial).predict(image, threshold=.1)
        observed = detector.predict(image, threshold=.1)
        np.testing.assert_allclose([value["box"] for value in observed],
                                   [value["box"] for value in expected], atol=.001)
        np.testing.assert_allclose([value["score"] for value in observed],
                                   [value["score"] for value in expected], atol=1e-5)

        ov_detector = DocumentDetector(output, backend="openvino")
        ov_detector.warmup(1)
        sample = torch.rand(1, 3, 64, 96)
        for expected_map, observed_map in zip(detector.infer(sample), ov_detector.infer(sample)):
            np.testing.assert_allclose(observed_map, expected_map, rtol=1e-4, atol=1e-5)
        ov_predictions = ov_detector.predict(image, threshold=.1)
        np.testing.assert_allclose([value["box"] for value in ov_predictions],
                                   [value["box"] for value in observed], atol=.001)

        with pytest.raises(FileExistsError, match=str(output)):
            exporter.export_model(checkpoint, output, input_size=(64, 96),
                                  dynamic_spatial=dynamic_spatial)
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
    monkeypatch.setattr(exporter_entry, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(exporter_entry, "OUTPUT_PATH", output)
    calls = []

    def fake_export(checkpoint_path, output_path):
        calls.append((checkpoint_path, output_path))
        return {"input_size": [64, 96]}

    monkeypatch.setattr(exporter_entry, "export_model", fake_export)
    exporter_entry.main()
    assert calls == [(checkpoint, output)]
    assert f"导出及数值校验通过: {output}" in capsys.readouterr().out


def test_export_rejects_invalid_input_size(tmp_path):
    output = tmp_path / "invalid.onnx"
    with pytest.raises(ValueError, match="input_size"):
        exporter.export_model(tmp_path / "unused.pth", output, input_size=65)
    assert not output.exists()
