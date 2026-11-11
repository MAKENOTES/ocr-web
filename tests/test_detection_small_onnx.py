"""验证小图检测 ONNX 固定/动态导出、runtime 一致性、同步和部署包。"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detection_small_export_onnx as exporter
import detection_small_inference_onnx as inference_entry
from detection_small import runtime
from detection_small.components.postprocess import decode_maps, suppress_boxes
from detection_small.components.preprocessing import (
    dynamic_input_size,
    prepare_image,
    restore_boxes,
)
from detection_small.models.east_model import EASTModel
from detection_small.utils import save_checkpoint
from detection_small.utils.config import ONNX_CONFIG_KEY, MODEL_FORMAT


def test_deployment_defaults_use_fixed_rectangular_input():
    """默认导出和入口使用训练一致的固定矩形输入。"""
    assert exporter.INPUT_SIZE == (256, 768)
    assert exporter.DYNAMIC_SPATIAL is False
    assert inference_entry.MODEL_PATH.suffix == ".onnx"
    assert inference_entry.NMS_IOU_THRESHOLD == 0.3


@pytest.mark.parametrize("dynamic_spatial", [False, True])
def test_export_and_runtime_roundtrip(tmp_path, monkeypatch, dynamic_spatial):
    import onnx

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = EASTModel(pretrained=False).eval()
        checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
        save_checkpoint(model, checkpoint)
        output.write_bytes(b"stale-onnx")
        config = exporter.export_model(checkpoint, output, input_size=(64, 96),
                                       dynamic_spatial=dynamic_spatial)
        assert config["format"] == MODEL_FORMAT
        assert config["input_size"] == [64, 96]
        assert config["dynamic_spatial"] is dynamic_spatial

        graph = onnx.load(output)
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        assert json.loads(properties[ONNX_CONFIG_KEY]) == config
        assert {value.name for value in graph.graph.output} == {"score", "geometry"}
        assert not (tmp_path / "model.tmp.onnx").exists()

        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (70, 180, 3), dtype=np.uint8))
        backend = runtime.load_backend(output, "onnx")
        dynamic, max_input_size, fixed_input_size = runtime.exported_shape(backend)
        assert dynamic is dynamic_spatial
        standalone, _ = runtime.predict_image(
            backend, image, .1, .3, dynamic, max_input_size, fixed_input_size)
        actual_boxes = np.asarray([value["box"] for value in standalone], dtype=np.float32)

        if dynamic:
            target_size = dynamic_input_size(image.size, (64, 96))
        else:
            target_size = (64, 96)
        tensor, metadata = prepare_image(image, target_size)
        with torch.inference_mode():
            score, geometry = model(tensor[None])
        boxes, scores = decode_maps(score[0].numpy(), geometry[0].numpy(), .1)
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, .3)
        expected_boxes = boxes[keep]
        expected_sorted = expected_boxes[np.lexsort((expected_boxes[:, 0], expected_boxes[:, 1]))]
        actual_sorted = actual_boxes[np.lexsort((actual_boxes[:, 0], actual_boxes[:, 1]))]
        np.testing.assert_allclose(actual_sorted, expected_sorted, atol=1e-3)

        ov_backend = runtime.load_backend(output, "openvino")
        dynamic, max_input_size, fixed_input_size = runtime.exported_shape(ov_backend)
        ov_standalone, _ = runtime.predict_image(
            ov_backend, image, .1, .3, dynamic, max_input_size, fixed_input_size)
        np.testing.assert_allclose(
            [value["box"] for value in ov_standalone],
            [value["box"] for value in standalone], atol=1e-3)
        np.testing.assert_allclose(
            [value["score"] for value in ov_standalone],
            [value["score"] for value in standalone], atol=1e-5)
    finally:
        torch.set_num_threads(previous)


def test_dynamic_flag_mismatch_is_rejected(tmp_path, monkeypatch):
    import onnx

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    torch.set_num_threads(1)
    model = EASTModel(pretrained=False).eval()
    checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
    save_checkpoint(model, checkpoint)
    exporter.export_model(checkpoint, output, input_size=(64, 96), dynamic_spatial=False)

    graph = onnx.load(output)
    properties = {entry.key: entry.value for entry in graph.metadata_props}
    config = json.loads(properties[ONNX_CONFIG_KEY])
    config["dynamic_spatial"] = True
    properties[ONNX_CONFIG_KEY] = json.dumps(config)
    onnx.helper.set_model_props(graph, properties)
    onnx.save(graph, output)

    backend = runtime.load_backend(output, "onnx")
    with pytest.raises(ValueError, match="dynamic_spatial"):
        runtime.exported_shape(backend)


def test_failed_export_does_not_publish(tmp_path, monkeypatch):
    """数值/导出失败时不能留下可被误认为成功的目标或临时文件。"""
    monkeypatch.setattr(exporter, "load_model", lambda _: EASTModel(pretrained=False).eval())

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(exporter, "export_graph", fail)
    output = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError, match="export failed"):
        exporter.export_model("unused.pth", output, input_size=(64, 96))
    assert not output.exists()
    assert not (tmp_path / "model.tmp.onnx").exists()


def test_export_rejects_invalid_input_size(tmp_path):
    output = tmp_path / "invalid.onnx"
    with pytest.raises(ValueError, match="input_size"):
        exporter.export_model(tmp_path / "unused.pth", output, input_size=65)
    assert not output.exists()


def test_export_entry_uses_path_constants(tmp_path, monkeypatch, capsys):
    checkpoint, output = tmp_path / "source.pth", tmp_path / "export.onnx"
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
    from detection_small.runtime.cli import build_parser

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
    with pytest.raises(SystemExit):
        inference_entry.main(["--model_path", str(tmp_path / "missing.onnx")])


def test_inference_entry_is_thin_runtime_wrapper():
    """根目录入口只保留常量和 main，不再复制推理实现。"""
    source = (Path(__file__).resolve().parents[1] / "detection_small_inference_onnx.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] == ["main"]
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert "detection_small.runtime.cli" in modules
    assert all(module in {"sys", "pathlib", "detection_small.runtime.cli"}
               for module in modules), modules


def test_runtime_is_self_contained_and_synced():
    """runtime 快照不能导入 detection_small 包内模块，且必须与源文件同步。"""
    from tools.sync_detection_small_runtime import check_runtime

    assert check_runtime() == []
    runtime_dir = Path(__file__).resolve().parents[1] / "detection_small" / "runtime"
    for path in sorted(runtime_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not (node.level == 0 and (
                    module == "detection_small" or module.startswith("detection_small."))), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name == "detection_small"
                               or alias.name.startswith("detection_small.")
                               for alias in node.names), path


def test_runtime_preprocessing_and_postprocess_match_package(tmp_path):
    """runtime 的图片读取、预处理和后处理应与包内实现一致。"""
    from detection_small.components.image import load_rgb_image as package_loader
    from detection_small.components.preprocessing import prepare_image as package_prepare
    from detection_small.runtime.image import load_rgb_image as runtime_loader
    from detection_small.runtime.preprocessing import prepare_image as runtime_prepare

    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (70, 180, 3), dtype=np.uint8))
    tensor, metadata = runtime_prepare(image, (64, 96))
    expected, expected_metadata = package_prepare(image, (64, 96))
    assert metadata == expected_metadata
    np.testing.assert_allclose(tensor.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)

    rgba = Image.new("RGBA", (16, 12), (255, 0, 0, 0))
    transparent = tmp_path / "alpha.png"
    rgba.save(transparent)
    np.testing.assert_array_equal(
        np.asarray(runtime_loader(transparent)), np.asarray(package_loader(transparent)))


def test_runtime_engine_writes_labels_and_crops(tmp_path):
    """推理流水线应输出原图尺寸预览、bbox4 标签和可选裁片。"""
    from detection_small.runtime.engine import run_inference

    image = Image.new("RGB", (80, 40), "white")
    image_path = tmp_path / "sample.png"
    image.save(image_path)
    output_dir = tmp_path / "output"
    crop_dir = tmp_path / "crops"

    class EmptyBackend:
        config = {"input_size": [64, 96], "dynamic_spatial": False}
        static_input_size = (64, 96)

        def infer(self, batch):
            score = np.zeros((1, 1, 16, 24), dtype=np.float32)
            geometry = np.zeros((1, 4, 16, 24), dtype=np.float32)
            return score, geometry

    summary = run_inference(
        EmptyBackend(), image_path, output_dir, dynamic=False,
        max_input_size=(64, 96), fixed_input_size=(64, 96),
        threshold=.7, iou_threshold=.3, save_crops=True, crop_dir=crop_dir)
    assert summary["images"] == 1
    assert (output_dir / "images" / "sample.png").is_file()
    assert (output_dir / "labels" / "gt_sample.txt").read_text() == ""
    assert crop_dir.is_dir()


def test_build_bundle_copies_whitelist_and_verifies(tmp_path, monkeypatch):
    """部署包只带 runtime、入口和模型，并能在隔离路径下完成一次推理。"""
    from tools.build_detection_small_bundle import build_bundle

    monkeypatch.setattr(exporter, "BENCHMARK_ENABLED", False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = EASTModel(pretrained=False).eval()
        checkpoint, output = tmp_path / "model.pth", tmp_path / "model.onnx"
        save_checkpoint(model, checkpoint)
        exporter.export_model(checkpoint, output, input_size=(64, 96), dynamic_spatial=False)

        bundle = tmp_path / "bundle"
        manifest = build_bundle(output, bundle, verify=True)
        assert (bundle / "detection_small_inference_onnx.py").is_file()
        assert (bundle / "detection_small/runtime/engine.py").is_file()
        assert manifest["model"]["file"] == "models/model.onnx"
        assert json.loads(manifest["model"]["config"])["format"] == MODEL_FORMAT
        assert not (bundle / "detection_small/components").exists()
        assert not (bundle / "detection_small/utils").exists()
    finally:
        torch.set_num_threads(previous)
