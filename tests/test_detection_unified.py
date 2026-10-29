"""统一检测的数据、张量、模型、推理、训练和迁移契约。"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from detection.components import eval_metrics
from detection.components.dataset import OCRDataset, read_boxes
from detection.components.ground_truth import generate_ground_truth
from detection.components.image import load_rgb_image
from detection.components.loss import EASTLoss
from detection.components.transforms import (
    dynamic_input_size,
    get_transforms,
    normalize_input_size,
    prepare_image,
    restore_boxes,
)
from detection.inference import (
    DocumentDetector,
    decode_maps,
    expand_predictions,
    run_inference,
    suppress_boxes,
)
from detection.models.east_model import create_model
from detection.utils import (
    checkpoint_config,
    load_checkpoint,
    save_checkpoint,
)


def make_source(tmp_path, count=4, ext=".png"):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for index in range(count):
        Image.new("RGB", (96, 64), (index * 20, 100, 200)).save(images / f"{index}{ext}")
        (labels / f"gt_{index}.txt").write_text("8 8 40 24\n", encoding="utf-8")
    return {"images": str(images), "labels": str(labels)}


@pytest.mark.parametrize("line", ["NaN 2 20 30", "20 2 10 30", "0 0 101 20", "bad label", "1 2 3"])
def test_invalid_labels_include_path_and_line(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match=r"bad.txt:1"):
        read_boxes(path, (100, 100))


def test_labels_preserve_numeric_text_ignore_empty_and_explicit_quads(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text('1,2,30,40,"123,456"\n2 3 10 20 ###\n')
    boxes, ignored = read_boxes(path, (100, 100))
    assert boxes.shape == (2, 4)
    assert ignored.tolist() == [False, True]

    path.write_text("1 2 30 40 hello,world\n")
    assert read_boxes(path, (100, 100))[0].tolist() == [[1, 2, 30, 40]]
    path.write_text("1,2,30,2,30,40,1,40,text\n")
    assert read_boxes(path, (100, 100), "quad8")[0].tolist() == [[1, 2, 30, 40]]
    path.write_text("1,2,30,3,30,40,1,40,text\n")
    with pytest.raises(ValueError, match="rotated"):
        read_boxes(path, (100, 100), "quad8")
    path.write_text("")
    assert read_boxes(path, (100, 100))[0].shape == (0, 4)
    with pytest.raises(ValueError, match="Unsupported label format"):
        read_boxes(path, (100, 100), "polygon")


def test_directory_loader_includes_mixed_formats_and_precise_labels(tmp_path):
    source = make_source(tmp_path, count=1)
    root = Path(source["images"])
    Image.new("RGB", (96, 64), "white").save(root / "pdf.JPG")
    Image.new("RGBA", (96, 64), (0, 0, 0, 0)).save(root / "web.PNG")
    for name in ("pdf", "web"):
        (Path(source["labels"]) / f"{name}.txt").write_text("8 8 40 24")
    dataset = OCRDataset(source["images"], source["labels"], get_transforms(False, 64))
    assert {path.name for path in dataset.image_paths} == {"0.png", "pdf.JPG", "web.PNG"}
    for index in range(len(dataset)):
        assert dataset[index]["image"].shape == (3, 64, 64)
        assert dataset[index]["score_map"].sum() > 0
    raw = OCRDataset(source["images"], source["labels"])
    assert all(raw[index]["image"].mode == "RGB" for index in range(len(raw)))


def test_directory_loader_rejects_missing_or_ambiguous_labels(tmp_path):
    source = make_source(tmp_path, count=1)
    label_dir = Path(source["labels"])
    (label_dir / "0.txt").write_text("8 8 40 24")
    with pytest.raises(ValueError, match="exactly one label"):
        OCRDataset(source["images"], source["labels"])
    (label_dir / "0.txt").unlink()
    (label_dir / "gt_0.txt").unlink()
    with pytest.raises(ValueError, match="0.png"):
        OCRDataset(source["images"], source["labels"])


def test_transparent_images_choose_visible_background(tmp_path):
    path = tmp_path / "transparent.png"
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24] = (0, 0, 0, 255)  # 透明区域围绕黑色前景，应合成白底。
    Image.fromarray(rgba, "RGBA").save(path)
    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((16, 16)) == (0, 0, 0)

    rgba[:, :, :3] = 255  # 白色前景应合成黑底，避免白字消失。
    Image.fromarray(rgba, "RGBA").save(path)
    assert load_rgb_image(path).getpixel((0, 0)) == (0, 0, 0)

    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(path)
    assert load_rgb_image(path).getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("size", [(32, 17), (2632, 1114), (2480, 3509), (64, 400)])
def test_fixed_preprocess_matches_training_and_coordinates_roundtrip(size):
    image = Image.new("RGB", size, (243, 240, 238))
    boxes = np.array([[1, 1, size[0] - 1, size[1] - 1]], dtype=np.float32)
    quads = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]],
                      boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    actual = get_transforms(False, 1024)(
        {"image": image, "quads": quads.copy(), "ignored": np.array([False])})
    tensor, metadata = prepare_image(image, 1024)
    torch.testing.assert_close(tensor, actual["image"], rtol=0, atol=0)

    scale, offset = metadata
    transformed = boxes * scale + np.array([*offset, *offset])
    np.testing.assert_allclose(restore_boxes(transformed, metadata, size), boxes, atol=0.001)


@pytest.mark.parametrize("image_size,max_size,expected", [
    ((320, 128), 256, (128, 256)),
    ((2539, 57), 640, (64, 640)),
    ((20, 20), 640, (640, 640)),
    ((100000, 1), 640, (64, 640)),
])
def test_dynamic_input_size_aligns_to_network_stride(image_size, max_size, expected):
    result = dynamic_input_size(image_size, max_size)
    assert result == expected
    assert all(value >= 64 and value % 32 == 0 for value in result)
    assert result[0] <= max_size and result[1] <= max_size


def test_dynamic_preprocess_matches_equivalent_fixed_transform():
    image = Image.new("RGB", (320, 128), (243, 240, 238))
    dynamic_tensor, metadata = prepare_image(image, 256, dynamic=True)
    fixed_tensor = get_transforms(False, (128, 256))(
        {"image": image, "quads": np.empty((0, 4, 2), dtype=np.float32),
         "ignored": np.empty(0, dtype=bool)})["image"]
    torch.testing.assert_close(dynamic_tensor, fixed_tensor, rtol=0, atol=0)
    assert dynamic_tensor.shape == (3, 128, 256)
    assert metadata[0] > 0


@pytest.mark.parametrize("input_size", [63, 100, (64, 63), (64,), (64, 96, 128)])
def test_invalid_input_size(input_size):
    with pytest.raises(ValueError, match="input_size"):
        normalize_input_size(input_size)


def test_ground_truth_and_loss_match_frozen_reference_values():
    """统一实现必须与拆分前冻结的真值、损失和梯度数值一致。"""
    quads = np.array([[[8, 8], [72, 8], [72, 32], [8, 32]],
                      [[16, 48], [40, 48], [40, 64], [16, 64]]], dtype=np.float32)
    score, geometry = generate_ground_truth((96, 96), quads)
    with np.load(Path(__file__).parent / "fixtures" / "detection_larger_reference.npz") as data:
        np.testing.assert_array_equal(score, data["score"])
        np.testing.assert_array_equal(geometry, data["geometry"])
        expected = [torch.from_numpy(data[f"result_{index}"]) for index in range(5)]

    truth_score, truth_geo = torch.from_numpy(score)[None, None], torch.from_numpy(geometry)[None]
    pred_score = torch.full_like(truth_score, .7, requires_grad=True)
    pred_geo = torch.full_like(truth_geo, .3, requires_grad=True)
    actual = EASTLoss(lambda_iou=3)(
        truth_score, pred_score, truth_geo, pred_geo, torch.ones_like(truth_score))
    actual_grads = torch.autograd.grad(actual[0], (pred_score, pred_geo))
    for reference, result in zip(expected, (*actual, *actual_grads)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_ground_truth_and_loss_are_finite_and_ignore_regions():
    quads = np.array([[[8, 8], [72, 8], [72, 32], [8, 32]],
                      [[16, 48], [40, 48], [40, 64], [16, 64]]], dtype=np.float32)
    score, geometry = generate_ground_truth((96, 96), quads)
    assert score.shape == (24, 24) and geometry.shape == (4, 24, 24)
    assert np.isfinite(score).all() and np.isfinite(geometry).all()
    assert set(np.unique(score)) <= {0.0, 1.0}
    assert (geometry >= 0).all()

    sample = get_transforms(False, 64)({
        "image": Image.new("RGB", (64, 64), "white"),
        "quads": np.array([[[8, 8], [24, 8], [24, 24], [8, 24]]], dtype=np.float32),
        "ignored": np.array([True]),
    })
    assert sample["training_mask"][2:6, 2:6].eq(0).all()
    assert sample["score_map"].sum() == 0

    pred_score = torch.full((1, 1, 16, 16), .8, requires_grad=True)
    pred_geo = torch.zeros((1, 4, 16, 16), requires_grad=True)
    loss = EASTLoss(3)(sample["score_map"][None, None], pred_score,
                       sample["geo_map"][None], pred_geo,
                       sample["training_mask"][None])[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert pred_score.grad[:, :, 2:6, 2:6].eq(0).all()
    assert pred_score.grad.sum() > 0
    assert pred_geo.grad.eq(0).all()


@pytest.mark.parametrize("length", [1, 6, 10, 14])
@pytest.mark.parametrize("vertical", [False, True])
def test_decode_maps_keeps_confident_thin_text_regions(length, vertical):
    score = np.zeros((1, 256, 256), dtype=np.float32)
    geometry = np.zeros((4, 256, 256), dtype=np.float32)
    ys = np.arange(20, 20 + length) if vertical else np.full(length, 20)
    xs = np.full(length, 20) if vertical else np.arange(20, 20 + length)
    score[0, ys, xs] = .95
    x0, y0, x1, y1 = xs.min() - 1, ys.min() - 1, xs.max() + 2, ys.max() + 2
    geometry[:, ys, xs] = np.log1p(np.stack([
        ys + .5 - y0, x1 - xs - .5, y1 - ys - .5, xs + .5 - x0,
    ]))
    boxes, confidence = decode_maps(score, geometry)
    np.testing.assert_allclose(boxes, [[x0 * 4, y0 * 4, x1 * 4, y1 * 4]], atol=1e-4)
    np.testing.assert_allclose(confidence, [.95])


def test_decode_maps_keeps_all_regions_above_threshold_and_rejects_bad_threshold():
    score = np.zeros((1, 284, 284), dtype=np.float32)
    for y in range(0, 284, 4):
        for x in range(0, 284, 4):
            score[0, y:y + 2, x:x + 2] = .8
    geometry = np.full((4, 284, 284), np.log1p(1), dtype=np.float32)
    boxes, confidence = decode_maps(score, geometry, threshold=.7)
    assert len(boxes) == 71 * 71
    np.testing.assert_allclose(confidence, .8)
    assert len(decode_maps(score, geometry, threshold=.9)[0]) == 0
    for threshold in (0, 1, -.1):
        with pytest.raises(ValueError, match="threshold must be"):
            decode_maps(np.zeros((1, 4, 4)), np.zeros((4, 4, 4)), threshold)


def test_nms_and_expansion_keep_adjacent_cells():
    boxes = np.array([[0, 0, 30, 20], [1, 0, 31, 20], [32, 0, 62, 20]], dtype=np.float32)
    assert suppress_boxes(boxes, np.array([.9, .8, .7])).tolist() == [0, 2]
    dead, scores = decode_maps(np.zeros((1, 16, 16)), np.zeros((4, 16, 16)))
    assert dead.shape == (0, 4) and scores.shape == (0,)

    original = [{"box": [1, 2, 19, 12], "score": .9}]
    assert expand_predictions(original, (20, 15), .2, .5) == [
        {"box": [0, 0, 20, 15], "score": .9}]
    assert original[0]["box"] == [1, 2, 19, 12]
    assert expand_predictions([], (20, 15), .1, .1) == []
    for ratio in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ratios"):
            expand_predictions(original, (20, 15), ratio, 0)


def test_model_supports_dynamic_shapes_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model = create_model().eval()
    with torch.no_grad():
        for shape in [(2, 3, 64, 96), (1, 3, 96, 64), (1, 3, 64, 640)]:
            score, geometry = model(torch.randn(*shape))
            assert score.shape == (shape[0], 1, shape[2] // 4, shape[3] // 4)
            assert geometry.shape == (shape[0], 4, shape[2] // 4, shape[3] // 4)

    loss = EASTLoss(3)(torch.ones(1, 1, 16, 16), model(torch.randn(1, 3, 64, 64))[0],
                       torch.zeros(1, 4, 16, 16), model(torch.randn(1, 3, 64, 64))[1])[0]
    loss.backward()
    assert torch.isfinite(loss) and model.score_conv.weight.grad.abs().sum() > 0

    path = tmp_path / "model.pth"
    save_checkpoint(model, path, (64, 96))
    loaded, checkpoint = load_checkpoint(path)
    assert checkpoint["config"]["input_size"] == [64, 96]
    model.eval()
    same_input = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(loaded(same_input)[0], model(same_input)[0])

    with pytest.raises(ValueError, match="input_size"):
        checkpoint_config(63)
    bad = tmp_path / "bad.pth"
    torch.save({"config": {"format": "old-tiled"}, "model": model.state_dict()}, bad)
    with pytest.raises(ValueError, match="unified EAST checkpoint"):
        load_checkpoint(bad)


def test_whole_image_inference_restores_long_lines_and_cells():
    image = Image.new("RGB", (320, 128), "white")
    boxes = np.array([[8, 16, 310, 48], [32, 80, 64, 112], [160, 80, 192, 112]], dtype=np.float32)
    quads = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]],
                      boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    target = get_transforms(False, 256)(
        {"image": image, "quads": quads, "ignored": np.zeros(3, dtype=bool)})
    calls = []

    class PerfectModel:
        def __call__(self, tensor):
            calls.append(tensor.shape)
            return target["score_map"][None, None], target["geo_map"][None]

    detector = DocumentDetector.__new__(DocumentDetector)
    detector.max_input_size = (256, 256)
    detector.dynamic_input = False
    detector.static_input_size = (256, 256)
    detector.openvino = detector.session = None
    detector.device, detector.model = "cpu", PerfectModel()
    result = detector.predict(image)
    assert len(calls) == 1
    assert len(result) == 3
    assert eval_metrics.match_counts(result, boxes, np.zeros(3, dtype=bool)).tolist() == [3, 0, 0]
    np.testing.assert_allclose(result[0]["box"], boxes[0], atol=0.001)


@pytest.mark.parametrize("available,requested,expected", [
    (True, "auto", "cuda"), (False, "auto", "cpu"),
    (True, "cpu", "cpu"), (True, "cuda:0", "cuda:0"),
])
def test_detector_selects_pytorch_device(monkeypatch, available, requested, expected):
    from detection import inference

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    calls = []

    def fake_load(path, device):
        calls.append((path, device))
        return object(), {"config": checkpoint_config(64)}

    monkeypatch.setattr(inference, "load_checkpoint", fake_load)
    detector = DocumentDetector("model.pth", requested)
    assert detector.device == expected
    assert calls == [("model.pth", expected)]


def test_detector_dynamic_and_fixed_input_shapes(monkeypatch):
    from detection import inference

    calls = []

    class FakeModel:
        def __call__(self, tensor):
            calls.append(tuple(tensor.shape))
            height, width = tensor.shape[-2:]
            return (torch.zeros(1, 1, height // 4, width // 4),
                    torch.zeros(1, 4, height // 4, width // 4))

    monkeypatch.setattr(inference, "load_checkpoint",
                        lambda *args: (FakeModel(), {"config": checkpoint_config(64)}))
    image = Image.new("RGB", (300, 100), "white")

    dynamic = DocumentDetector("model.pth", max_input_size=(64, 256), dynamic_input=True)
    dynamic.warmup(1)
    assert sorted(calls) == sorted([(1, 3, 64, 256), (1, 3, 64, 128)])
    calls.clear()
    assert dynamic.predict(image) == []
    assert calls == [(1, 3, 64, 192)]

    calls.clear()
    fixed = DocumentDetector("model.pth", max_input_size=(64, 256), dynamic_input=False)
    fixed.warmup(1)
    assert fixed.predict(image) == []
    assert calls == [(1, 3, 64, 256)] * 2

    with pytest.raises(ValueError, match="input_size"):
        DocumentDetector("unused.pth", max_input_size=65)


def test_inference_main_uses_constants(tmp_path, monkeypatch):
    from detection import inference

    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (40, 30), "white").save(source / "page.png")
    (source / "notes.txt").write_text("非图片文件", encoding="utf-8")
    output = tmp_path / "output"
    model_path = tmp_path / "model.pth"
    for name, value in {
        "MODEL_PATH": model_path, "IMAGE_PATH": source, "OUTPUT_DIR": output,
        "CROP_DIR": tmp_path / "crops", "DEVICE": "cpu",
        "SCORE_THRESHOLD": .6, "NMS_IOU_THRESHOLD": .2, "SAVE_CROPS": True,
    }.items():
        monkeypatch.setattr(inference, name, value)

    class FakeDetector:
        def __init__(self, path, device, backend="auto"):
            self.backend, self.last_infer_ms = backend, 1.0
            assert (path, device) == (model_path, "cpu")

        def predict(self, image, threshold, iou_threshold):
            assert image.size == (40, 30)
            assert (threshold, iou_threshold) == (.6, .2)
            return [{"box": [2.25, 3, 20, 15], "score": .9}]

    monkeypatch.setattr(inference, "DocumentDetector", FakeDetector)
    inference.main()
    assert sorted(path.name for path in output.iterdir()) == ["images", "labels"]
    dataset = OCRDataset(output / "images", output / "labels")
    assert dataset.label_paths == [output / "labels" / "gt_page.txt"]
    boxes, ignored = read_boxes(dataset.label_paths[0], (40, 30))
    assert boxes.tolist() == [[2.25, 3, 20, 15]] and not ignored.any()
    with Image.open(tmp_path / "crops" / "page-00001.png") as crop:
        assert crop.size == (18, 12)


def test_inference_rejects_empty_directory_and_duplicate_stems(tmp_path, monkeypatch):
    from detection import inference

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(inference, "IMAGE_PATH", empty)
    monkeypatch.setattr(inference, "OUTPUT_DIR", tmp_path / "output")
    with pytest.raises(ValueError, match=str(empty)):
        inference.main()
    assert not (tmp_path / "output").exists()

    source = tmp_path / "images"
    source.mkdir()
    for name in ("page.jpg", "page.png"):
        Image.new("RGB", (40, 30), "white").save(source / name)
    monkeypatch.setattr(inference, "IMAGE_PATH", source)
    with pytest.raises(ValueError, match="Duplicate image stem for labels:.*page.jpg.*page.png"):
        inference.main()


def test_run_summary_and_expanded_outputs(tmp_path, monkeypatch):
    from detection import inference

    image = tmp_path / "page.png"
    Image.new("RGB", (20, 15), "white").save(image)

    class Detector:
        backend, last_infer_ms = "onnx", 2.5

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args):
            return [{"box": [1, 2, 19, 12], "score": .9}]

    monkeypatch.setattr(inference, "DocumentDetector", Detector)
    summary = run_inference("model.onnx", image, tmp_path / "output",
                            save_crops=True, crop_dir=tmp_path / "crops",
                            expand_ratio_w=.2, expand_ratio_h=.5)
    assert summary["images"] == summary["boxes"] == 1
    assert summary["mean_infer_ms"] == 2.5
    assert summary["wall_seconds"] > 0 and summary["images_per_second"] > 0
    assert (tmp_path / "output/labels/gt_page.txt").read_text() == "0.0 0.0 20.0 15.0\n"
    with Image.open(tmp_path / "crops/page-00001.png") as crop:
        assert crop.size == (20, 15)


def test_evaluation_reads_directories(tmp_path, monkeypatch, capsys):
    import json
    import sys

    source = make_source(tmp_path, count=2)

    class PerfectDetector:
        def predict(self, image, threshold, nms_iou):
            return [{"box": [8, 8, 40, 24], "score": 0.9}]

    monkeypatch.setattr(eval_metrics, "DocumentDetector", lambda *args: PerfectDetector())
    monkeypatch.setattr(sys, "argv", ["eval_metrics.py", "--model", "unused.pth",
                                     "--image-dir", source["images"],
                                     "--label-dir", source["labels"]])
    eval_metrics.main()
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["tp"] == 2
    assert report["overall"]["fp"] == report["overall"]["fn"] == 0
    assert report["formats"][".png"]["f1"] == 1


def test_evaluation_model_reports_overall_and_source_groups(tmp_path):
    from detection.components.eval_metrics import evaluate_model

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("ui_0.png", "doc_0.png"):
        Image.new("RGB", (64, 64), "white").save(images / name)
        (labels / f"gt_{Path(name).stem}.txt").write_text("8 8 40 24\n")

    class ZeroModel:
        def eval(self):
            return self

        def __call__(self, tensor):
            height, width = tensor.shape[-2:]
            return (torch.zeros(1, 1, height // 4, width // 4),
                    torch.zeros(1, 4, height // 4, width // 4))

    report = evaluate_model(ZeroModel(), OCRDataset(images, labels), "cpu", 64)
    assert report["overall"]["fn"] == 2
    assert report["ui"]["fn"] == 1
    assert report["doc"]["fn"] == 1
    assert report["overall"]["f1"] == 0


def test_training_entry_runs_reference_loop_and_saves_top_k(tmp_path, monkeypatch):
    from detection import train as training

    source = make_source(tmp_path)
    validation = make_source(tmp_path / "validation", count=2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "VAL_INTERVAL", 1)
    monkeypatch.setattr(training, "CHECKPOINT_INTERVAL", 1)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, value in {
            "TRAIN_IMAGE_DIR": source["images"], "TRAIN_LABEL_DIR": source["labels"],
            "VAL_IMAGE_DIR": validation["images"], "VAL_LABEL_DIR": validation["labels"],
            "INPUT_SIZE": 64, "BATCH_SIZE": 2, "EPOCHS": 1, "NUM_WORKERS": 0,
            "DEVICE": "cpu", "PRETRAINED": False, "LOG_GRAPH": False,
        }.items():
            monkeypatch.setattr(training, name, value)
        training.main(["--output", str(tmp_path / "run")])
        assert (tmp_path / "run" / "detection_epoch_1.pth").exists()
        best = list((tmp_path / "run").glob("best_detection_epoch_*.pth"))
        assert len(best) == 1
        model, checkpoint = load_checkpoint(best[0])
        assert checkpoint["config"]["input_size"] == [64, 64]
        with torch.no_grad():
            assert torch.isfinite(model(torch.rand(1, 3, 64, 64))[0]).all()
    finally:
        torch.set_num_threads(old_threads)


def test_training_options_only_accept_output_directory(tmp_path):
    from detection.components.training_options import parse_args

    args = parse_args(["--output", str(tmp_path / "new")])
    assert args.output == tmp_path / "new"
    for option in ("--epochs", "--input-size", "--train-images",
                   "--label-format", "--device"):
        with pytest.raises(SystemExit):
            parse_args([option, "1"])
    (tmp_path / "keep.pth").write_text("existing")
    with pytest.raises(FileExistsError, match="must be empty"):
        parse_args(["--output", str(tmp_path)])
    assert (tmp_path / "keep.pth").read_text() == "existing"


def test_migration_dedupes_and_keeps_train_val_disjoint(tmp_path, monkeypatch):
    from tools import prepare_unified_detection_data as migration

    def make_pairs(root, prefix, colors):
        image_dir, label_dir = root / "images", root / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        for index, color in enumerate(colors):
            Image.new("RGB", (32, 32), color).save(image_dir / f"{prefix}{index}.png")
            (label_dir / f"gt_{prefix}{index}.txt").write_text("2 2 12 12\n")
        return image_dir, label_dir

    ui_train = make_pairs(tmp_path / "ui_train", "a", [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
    ui_val = make_pairs(tmp_path / "ui_val", "v", [(1, 2, 3), (4, 5, 6)])
    doc_train = make_pairs(tmp_path / "doc_train", "d", [(21, 22, 23), (24, 25, 26)])
    doc_val = make_pairs(tmp_path / "doc_val", "e", [(21, 22, 23)])
    destination = tmp_path / "data"
    monkeypatch.setattr(migration, "SOURCES", (
        ("ui", 0, *ui_train), ("ui", 1, *ui_val),
        ("doc", 0, *doc_train), ("doc", 1, *doc_val)))
    monkeypatch.setattr(migration, "DESTINATION", destination)
    monkeypatch.setattr(migration, "MANIFEST", destination / "split_manifest.csv")
    monkeypatch.setattr(migration, "TEST_SOURCE", tmp_path / "missing")

    source_files = sorted(path.name for path in ui_train[0].iterdir())
    migration.main()
    assert sorted(path.name for path in ui_train[0].iterdir()) == source_files
    assert len(list((destination / "train_images").iterdir())) == 4
    assert len(list((destination / "val_images").iterdir())) == 2

    import csv
    with (destination / "split_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    by_hash = {}
    for row in rows:
        by_hash.setdefault(row["content_md5"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_hash.values())
