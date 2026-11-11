"""小图检测分析报告、检查点解析和框匹配的单元测试。"""
import json

import numpy as np
import pytest
from PIL import Image

from detection_small.components.eval_metrics import (
    decode_predictions,
    match_box_status,
    match_counts,
    summarize_counts,
)
from detection_small.models.east_model import EASTModel
from detection_small.utils import save_checkpoint
from detection_small.utils.training_eval import (
    build_workpoints,
    collect_report,
    describe_checkpoint,
    json_safe,
    render_report,
    report_summary,
    scan_checkpoints,
    select_checkpoints,
)


def test_match_box_status_preserves_counts_and_marks_fp():
    """逐框状态应和原 match_counts 计数一致，并能指出误检框。"""
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    ignored = np.array([False, False])
    predictions = [
        {"box": [0, 0, 10, 10], "score": 0.9},
        {"box": [50, 50, 60, 60], "score": 0.8},
        {"box": [20, 20, 30, 30], "score": 0.7},
    ]
    status, used, counts = match_box_status(predictions, boxes, ignored, 0.5)
    assert status.tolist() == [1, 0, 1]
    assert used == {0, 1}
    assert counts.tolist() == [2, 1, 0]
    assert match_counts(predictions, boxes, ignored, 0.5).tolist() == [2, 1, 0]


def test_match_box_status_skips_prediction_inside_ignore_region():
    """主要落在 ### 内的预测不计误检，但对应真值仍会计入漏检。"""
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    ignored = np.array([False, True])
    predictions = [{"box": [20, 20, 30, 30], "score": 0.9}]
    status, used, counts = match_box_status(predictions, boxes, ignored, 0.5)
    assert status.tolist() == [-1]
    assert used == set()
    assert counts.tolist() == [0, 0, 1]


def test_decode_predictions_uses_four_connectivity_and_scaled_small_area():
    """对角相邻的两个区域在四连通下应拆成两个框，而不是合并成一个。"""
    score = np.zeros((16, 16), dtype=np.float32)
    score[2, 2:4] = 0.9
    score[3, 4:6] = 0.9
    geo = np.zeros((4, 16, 16), dtype=np.float32)
    geo[:, 2, 2:4] = np.log1p(1)
    geo[:, 3, 4:6] = np.log1p(1)
    boxes, scores = decode_predictions(score, np.expm1(geo), 0.7, small_region_area=3)
    assert len(boxes) == 2
    assert np.all(scores > 0.8)


def test_scan_and_select_checkpoints(tmp_path):
    """文件名解析应区分最佳 F1、最佳 loss 和最后一轮，兼容历史命名。"""
    names = [
        "best_f1_detection_epoch_360_f1_0.94158.pth",
        "best_f1_detection_epoch_320_f1_0.94117.pth",
        "best_detection_epoch_194_loss_0.00516.pth",
        "detection_epoch_400.pth",
        "detection_epoch_380_early_stop.pth",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"x")

    checkpoints = scan_checkpoints(tmp_path)
    assert len(checkpoints) == 5
    assert describe_checkpoint(tmp_path / names[0])["kind"] == "best_f1"
    chosen = select_checkpoints(checkpoints)
    assert chosen["best_f1"]["name"] == names[0]
    assert chosen["best_loss"]["name"] == names[2]
    assert chosen["last"]["name"] == names[3]
    assert chosen["primary"]["name"] == names[0]


def test_build_workpoints_selects_highest_overall_f1():
    """工作点对比应选中总体 F1 最高的输入尺寸和阈值组合。"""
    rows = {}
    for size, threshold, f1 in (((256, 768), 0.7, 0.941), ((320, 960), 0.85, 0.946)):
        metrics = summarize_counts([100, 2, 2])
        metrics["f1"] = f1
        rows[(size, threshold)] = {"overall": metrics}
    workpoints, best = build_workpoints(rows)
    assert len(workpoints) == 2
    assert best["input_size"] == (320, 960)
    assert best["threshold"] == 0.85


def sample_report():
    """构造渲染测试所需的最小报告数据。"""
    metrics = summarize_counts([10, 2, 3])
    point = {"input_size": (256, 768), "threshold": 0.7, "overall": metrics}
    return {
        "generated_at": "2026-01-01 00:00",
        "run": {
            "name": "<script>alert(1)</script>",
            "epochs": 400,
            "best_f1": 0.94158,
            "best_f1_epoch": 360,
            "best_loss": 0.00516,
            "best_loss_epoch": 194,
            "last_f1": 0.93723,
            "training_hours": 6.5,
            "throughput": 152.5,
            "has_curves": True,
        },
        "curves": {},
        "dataset": {
            "train": {"images": 10, "label_boxes": 100, "ignored_boxes": 2},
            "val": {"images": 5, "label_boxes": 50, "ignored_boxes": 1},
        },
        "primary": {
            "checkpoint": "best_f1_detection_epoch_360_f1_0.94158.pth",
            "input_size": [256, 768],
            "threshold": 0.7,
            "nms_iou": 0.3,
            "match_iou": 0.5,
        },
        "groups": {"overall": metrics},
        "workpoints": [point],
        "best_workpoint": point,
        "checkpoints": [{
            "name": "best_f1_detection_epoch_360_f1_0.94158.pth",
            "path": "/tmp/model.pth",
            "kind": "best_f1",
            "epoch": 360,
            "f1": 0.94158,
            "loss": None,
            "size_mb": 48.0,
            "mtime": "2026-11-11 12:00",
            "eval": {"overall": metrics},
        }],
        "error_analysis": {
            "summary": {
                "images": 5, "gt": 50, "total_errors": 5, "error_images": 2,
                "zero_error_images": 3, "top_ten_errors": 5, "top_ten_share": 1.0,
            },
            "orientation": {"wide": {"images": 3, "errors": 5, "fp": 2, "fn": 3}},
            "images": [],
        },
        "deployment": [{
            "status": "warn",
            "title": "推理入口检查点未更新",
            "detail": "当前指向旧检查点。",
            "action": "更新 MODEL_PATH。",
        }],
    }


def test_render_report_escapes_dynamic_text_and_contains_sections():
    """静态报告应包含关键章节，并转义运行名里的 HTML。"""
    html = render_report(sample_report())
    assert "小图检测训练分析报告" in html
    assert "部署检查清单" in html
    assert "指标口径" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_report_summary_drops_curves_and_image_data():
    """summary.json 不应包含曲线和图片 data URI，但保留逐图统计摘要。"""
    report = sample_report()
    summary = report_summary(report)
    assert "curves" not in summary
    assert "error_analysis" not in summary
    assert summary["error_summary"]["summary"]["total_errors"] == 5


def test_json_safe_converts_numpy_values():
    """numpy 标量和数组应转换为 Python 原生类型。"""
    result = json_safe({"count": np.int64(3), "value": np.float32(0.5),
                        "array": np.array([1, 2])})
    assert result == {"count": 3, "value": 0.5, "array": [1, 2]}


def test_collect_report_evaluates_a_small_run(tmp_path):
    """用一张小图和随机权重跑通报告收集，验证运行配置、指标和渲染输出。"""
    image_dir, label_dir = tmp_path / "val_images", tmp_path / "val_labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (80, 40), "white").save(image_dir / "ui_1.png")
    (label_dir / "gt_ui_1.txt").write_text("10 5 60 30\n", encoding="utf-8")

    train_images, train_labels = tmp_path / "train_images", tmp_path / "train_labels"
    train_images.mkdir()
    train_labels.mkdir()

    run_dir = tmp_path / "ocr_experiment_test"
    run_dir.mkdir()
    model = EASTModel(pretrained=False)
    save_checkpoint(model, run_dir / "best_f1_detection_epoch_1_f1_0.5.pth")
    (run_dir / "run_config.json").write_text(json.dumps({
        "input_size": [64, 64],
        "train_image_dir": str(train_images),
        "train_label_dir": str(train_labels),
        "val_image_dir": str(image_dir),
        "val_label_dir": str(label_dir),
    }), encoding="utf-8")

    report = collect_report(run_dir, "cpu")
    assert report["primary"]["input_size"] == [64, 64]
    assert report["groups"]["overall"]["tp"] >= 0
    assert report["workpoints"]
    assert "<html" in render_report(report)
