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
from detection import runtime as detection_runtime
from detection.utils.postprocess import suppress_boxes as shared_suppress_boxes
from detection.inference import DocumentDetector
from detection.models.east_model import create_model
from detection.utils import ONNX_CONFIG_KEY, save_checkpoint


def test_deployment_defaults_use_dynamic_canvas():
    """默认部署与导出都使用动态尺寸：长边上限 896，超宽图短边补到长边一半。"""
    from detection import inference
    import detection_inference_onnx as onnx_entry

    assert inference.DYNAMIC_INPUT is True
    assert inference.INPUT_SIZE == 896
    assert exporter.DYNAMIC_SPATIAL is True
    assert exporter.INPUT_SIZE == 896
    assert exporter.OUTPUT_PATH.name == "best_detection_dynamic.onnx"
    assert onnx_entry.MODEL_PATH.name == "best_detection_dynamic.onnx"


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

        # 使用确定性噪声图，避免纯白图在所有后端都输出空框、对比失去意义。
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (70, 180, 3), dtype=np.uint8))
        expected = DocumentDetector(checkpoint, max_input_size=(64, 96),
                                    dynamic_input=dynamic_spatial).predict(image, threshold=.1)

        # 独立 ONNX 入口应与包内 PyTorch 推理输出一致。
        backend = detection_runtime.load_backend(output, "onnx")
        dynamic, max_input_size, fixed_input_size = detection_runtime.exported_shape(backend)
        assert dynamic is dynamic_spatial
        assert (max_input_size if dynamic else fixed_input_size) == (64, 96)
        standalone, _ = detection_runtime.predict_image(
            backend, image, .1, .3, dynamic, max_input_size, fixed_input_size)
        np.testing.assert_allclose([value["box"] for value in standalone],
                                   [value["box"] for value in expected], atol=.001)
        np.testing.assert_allclose([value["score"] for value in standalone],
                                   [value["score"] for value in expected], atol=1e-5)

        # OpenVINO 后端应与 ONNX Runtime 输出一致。
        ov_backend = detection_runtime.load_backend(output, "openvino")
        dynamic, max_input_size, fixed_input_size = detection_runtime.exported_shape(ov_backend)
        ov_standalone, _ = detection_runtime.predict_image(
            ov_backend, image, .1, .3, dynamic, max_input_size, fixed_input_size)
        np.testing.assert_allclose([value["box"] for value in ov_standalone],
                                   [value["box"] for value in standalone], atol=.001)
        np.testing.assert_allclose([value["score"] for value in ov_standalone],
                                   [value["score"] for value in standalone], atol=1e-5)
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
    backend = detection_runtime.load_backend(output, "onnx")
    with pytest.raises(ValueError, match="dynamic_spatial"):
        detection_runtime.exported_shape(backend)


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
    from detection.runtime.cli import build_parser

    parser = build_parser(
        description="test",
        default_model_path=inference_entry.MODEL_PATH,
        default_image_path=inference_entry.IMAGE_PATH,
        default_output_dir=inference_entry.OUTPUT_DIR,
        default_backend=inference_entry.BACKEND,
        default_score_thr=inference_entry.SCORE_THRESHOLD,
        default_nms_iou=inference_entry.NMS_IOU_THRESHOLD,
        default_save_crops=inference_entry.SAVE_CROPS,
        default_crop_dir=inference_entry.CROP_DIR,
        default_warmup_runs=inference_entry.WARMUP_RUNS,
    )
    args = parser.parse_args([
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


def test_inference_entry_is_thin_runtime_wrapper():
    """根目录入口只保留常量和 main，不再复制推理实现。"""
    source = (Path(__file__).resolve().parents[1] / "detection_inference_onnx.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] == ["main"]
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert "detection.runtime.cli" in modules
    assert all(module in {"sys", "pathlib", "detection.runtime.cli"} for module in modules), modules


def test_runtime_is_self_contained_and_synced():
    """runtime 快照不能导入 detection 包内模块，且必须与源文件同步。"""
    from tools.sync_detection_runtime import check_runtime

    assert check_runtime() == []
    runtime_dir = Path(__file__).resolve().parents[1] / "detection" / "runtime"
    for path in sorted(runtime_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not (node.level == 0 and (
                    module == "detection" or module.startswith("detection."))), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name == "detection" or alias.name.startswith("detection.")
                               for alias in node.names), path


def test_runtime_suppress_boxes_matches_package():
    """runtime 的 NMS 应与包内实现选择相同索引。"""
    from detection.runtime.postprocess import suppress_boxes as runtime_suppress_boxes

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30],
                      [21, 21, 31, 31], [40, 40, 50, 50]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(
        runtime_suppress_boxes(boxes, scores, 0.3),
        shared_suppress_boxes(boxes, scores, 0.3))


def test_runtime_preprocessing_matches_package(tmp_path):
    """runtime 的图片读取和预处理应与包内实现一致。"""
    from detection.components.image import load_rgb_image as package_loader
    from detection.components.preprocessing import prepare_image as package_prepare
    from detection.runtime.image import load_rgb_image as runtime_loader
    from detection.runtime.preprocessing import prepare_image as runtime_prepare

    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (70, 180, 3), dtype=np.uint8))
    tensor, metadata = runtime_prepare(image, (64, 96))
    expected, expected_metadata = package_prepare(image, (64, 96), dynamic=False)
    assert metadata == expected_metadata
    np.testing.assert_allclose(tensor.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)

    rgba = Image.new("RGBA", (16, 12), (255, 0, 0, 0))
    transparent = tmp_path / "alpha.png"
    rgba.save(transparent)
    np.testing.assert_array_equal(
        np.asarray(runtime_loader(transparent)),
        np.asarray(package_loader(transparent)))


def test_runtime_does_not_reuse_gpu_inference():
    """runtime 不导入 PyTorch/GPU 推理模块，也不引用根目录导出实现。"""
    root = Path(__file__).resolve().parents[1]
    for name in ("detection_inference_onnx.py", "detection/runtime/engine.py",
                 "detection/runtime/backends.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "detection.inference" not in source, name
        assert "DocumentDetector" not in source, name
        assert "detection_export_onnx" not in source, name
        assert "export_model" not in source, name


def test_report_entries_reuse_runtime():
    """报告入口必须直接使用 runtime，不能依赖根目录薄入口的旧实现函数。"""
    from detection import report_inference, report_size_policy, runtime

    assert report_inference.onnx_runtime is runtime
    assert report_size_policy.onnx_runtime is runtime


def test_export_rejects_invalid_input_size(tmp_path):
    output = tmp_path / "invalid.onnx"
    with pytest.raises(ValueError, match="input_size"):
        exporter.export_model(tmp_path / "unused.pth", output, input_size=65)
    assert not output.exists()


def test_build_bundle_copies_whitelist_and_verifies(tmp_path, monkeypatch):
    """部署包只带 runtime、入口和模型，并能在隔离路径下完成一次推理。"""
    from tools.build_detection_bundle import build_bundle

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model().eval()
        checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
        save_checkpoint(model, checkpoint, (64, 96))
        exporter.export_model(checkpoint, output, input_size=(64, 96), dynamic_spatial=False)

        bundle = tmp_path / "bundle"
        manifest = build_bundle(output, bundle, verify=True)
        assert (bundle / "detection_inference_onnx.py").is_file()
        assert (bundle / "detection/runtime/engine.py").is_file()
        assert manifest["model"]["file"] == "models/model.onnx"
        assert not (bundle / "detection/components").exists()
        assert not (bundle / "detection/utils").exists()
    finally:
        torch.set_num_threads(previous)
