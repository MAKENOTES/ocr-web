"""统一检测部署推理报告的指标计算和渲染测试。"""
import json

import numpy as np
import pytest
from PIL import Image

from detection.components.eval_metrics import summarize_counts
from detection.components.inference_report import (
    build_report,
    compare_backends,
    evaluate_split,
    load_image_index,
    write_report,
)


def write_image_split(root, specs):
    """按 stem -> (GT 框列表, 预测框列表) 写出图片、真值和预测标签。"""
    image_dir = root / "images"
    gt_dir = root / "gt"
    pred_dir = root / "pred"
    for directory in (image_dir, gt_dir, pred_dir):
        directory.mkdir()
    for stem, (truth, predicted) in specs.items():
        Image.new("RGB", (64, 64), "white").save(image_dir / f"{stem}.png")
        for directory, boxes in ((gt_dir, truth), (pred_dir, predicted)):
            (directory / f"gt_{stem}.txt").write_text(
                "".join(" ".join(map(str, box)) + "\n" for box in boxes), encoding="utf-8")
    return image_dir, gt_dir, pred_dir


def test_evaluate_split_matches_training_metric_definition(tmp_path):
    """TP/FP/FN 应按 IoU 0.5 一对一匹配，并按前缀分到 doc/ui。"""
    specs = {
        "ui_a": ([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
        "ui_b": ([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]]),
        "doc_c": ([[0, 0, 10, 10], [20, 20, 30, 30]], [[0, 0, 10, 10]]),
    }
    image_dir, gt_dir, pred_dir = write_image_split(tmp_path, specs)
    index = load_image_index(image_dir)

    result = evaluate_split(index, gt_dir, pred_dir)

    overall = result["groups"]["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (3, 1, 1)
    assert result["groups"]["doc"]["fn"] == 1
    assert result["groups"]["ui"]["fp"] == 1
    assert result["error_analysis"]["summary"]["total_errors"] == 2
    assert len(result["pred_boxes"]["ui_b"]) == 2


def test_evaluate_split_rejects_missing_prediction(tmp_path):
    """缺少预测标签时应立即报错，不能把缺失当成空预测。"""
    image_dir, gt_dir, pred_dir = write_image_split(
        tmp_path, {"ui_a": ([[0, 0, 10, 10]], [[0, 0, 10, 10]])})
    (pred_dir / "gt_ui_a.txt").unlink()
    index = load_image_index(image_dir)

    with pytest.raises(FileNotFoundError):
        evaluate_split(index, gt_dir, pred_dir)


def test_compare_backends_reports_identical_predictions(tmp_path):
    """相同预测的框级一致率应为 1，框数和坐标一致计数覆盖全部图片。"""
    specs = {
        "ui_a": ([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
        "doc_b": ([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]]),
    }
    image_dir, gt_dir, pred_dir = write_image_split(tmp_path, specs)
    index = load_image_index(image_dir)
    reference = evaluate_split(index, gt_dir, pred_dir)
    observed = evaluate_split(index, gt_dir, pred_dir)

    agreement = compare_backends(index, reference, observed)

    assert agreement["same_box_count"] == 2
    assert agreement["same_coordinates"] == 2
    assert agreement["groups"]["overall"]["f1"] == 1.0


def test_write_report_outputs_lightweight_summary(tmp_path):
    """summary.json 不应嵌入错误图片的 data URI，HTML 应包含关键章节。"""
    specs = {
        "ui_a": ([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
        "doc_b": ([[0, 0, 10, 10], [20, 20, 30, 30]], [[0, 0, 10, 10]]),
    }
    image_dir, gt_dir, pred_dir = write_image_split(tmp_path, specs)
    index = load_image_index(image_dir)
    data = evaluate_split(index, gt_dir, pred_dir, collect_visuals=True)

    runs = {}
    for backend in ("pytorch", "onnx"):
        for split in ("train", "val"):
            runs[(backend, split)] = {
                "backend": backend,
                "split": split,
                "images": len(index),
                "boxes": 3,
                "input_size": 768,
                "threshold": 0.7,
                "groups": data["groups"],
                "records": data["records"],
                "pred_boxes": data["pred_boxes"],
                "image_index": index,
                "timing": {"images": 2, "boxes": 3, "mean_infer_ms": 1.0,
                           "mean_detect_ms": 2.0, "wall_seconds": 1.0,
                           "images_per_second": 2.0},
                "error_analysis": data["error_analysis"],
            }
    training = {
        "run": {"name": "fake", "epochs": 1, "best_f1": 0.9, "best_f1_epoch": 1,
                "best_loss": 0.1, "best_loss_epoch": 1, "last_f1": 0.9,
                "training_hours": 1.0, "throughput": 10.0},
        "primary": {"input_size": 768, "threshold": 0.7, "nms_iou": 0.3,
                    "match_iou": 0.5},
        "groups": {"overall": summarize_counts([10, 1, 1]),
                   "doc": summarize_counts([4, 1, 1]),
                   "ui": summarize_counts([6, 0, 0])},
        "best_workpoint": None,
    }
    config = {"checkpoint": "fake.pth", "onnx_model": "fake.onnx", "input_size": 768,
              "dynamic_input": True, "score_threshold": 0.7, "nms_iou": 0.3,
              "onnx_backend": "openvino"}

    report = build_report(runs, training, {}, config, "fake_run")
    html_path, summary_path, csv_paths = write_report(report, tmp_path / "report")

    html = html_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "统一检测部署推理报告" in html
    assert "后端一致性" in html
    assert "data:image" not in summary_text
    assert json.loads(summary_text)["runs"]["pytorch_val"]["groups"]["overall"]["tp"] == 2
    assert {path.name for path in csv_paths} == {"errors_train.csv", "errors_val.csv"}
    train_csv = (tmp_path / "report" / "errors_train.csv").read_text(encoding="utf-8-sig")
    assert train_csv.startswith("image,path,split,source")
    assert "doc_b" in train_csv
