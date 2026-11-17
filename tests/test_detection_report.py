"""统一检测分析报告和框匹配状态的单元测试。"""
import csv

import numpy as np
from PIL import Image

from detection.components.eval_metrics import match_box_status, match_counts, summarize_counts
from detection.utils.training_eval import (
    build_error_analysis,
    build_workpoints,
    describe_checkpoint,
    json_safe,
    render_report,
    report_summary,
    scan_checkpoints,
    select_checkpoints,
    write_error_details,
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


def test_scan_and_select_checkpoints(tmp_path):
    """文件名解析应区分最佳 F1、最佳 loss 和最后一轮。"""
    names = [
        "best_f1_detection_epoch_560_f1_0.97158.pth",
        "best_f1_detection_epoch_510_f1_0.97117.pth",
        "best_detection_epoch_597_loss_0.04613.pth",
        "detection_epoch_620.pth",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"x")

    checkpoints = scan_checkpoints(tmp_path)
    assert len(checkpoints) == 4
    assert describe_checkpoint(tmp_path / names[0])["kind"] == "best_f1"
    chosen = select_checkpoints(checkpoints)
    assert chosen["best_f1"]["name"] == names[0]
    assert chosen["best_loss"]["name"] == names[2]
    assert chosen["last"]["name"] == names[3]
    assert chosen["primary"]["name"] == names[0]


def test_build_workpoints_selects_highest_overall_f1():
    """工作点对比应选中总体 F1 最高的输入尺寸和阈值组合。"""
    rows = {}
    for size, threshold, f1 in ((768, 0.7, 0.971), (896, 0.85, 0.976)):
        metrics = summarize_counts([100, 2, 2])
        metrics["f1"] = f1
        rows[(size, threshold)] = {"overall": metrics, "doc": metrics, "ui": metrics}
    workpoints, best = build_workpoints(rows)
    assert len(workpoints) == 2
    assert best["input_size"] == 896
    assert best["threshold"] == 0.85


def build_error_analysis_fixture(split, preview_name):
    """构造一个数据划分的逐图错误摘要。"""
    return {
        "summary": {
            "images": 5, "gt": 50, "total_errors": 5, "error_images": 2,
            "zero_error_images": 3, "top_ten_errors": 5, "top_ten_share": 1.0,
        },
        "groups": {"doc": {"images": 1, "errors": 5, "fp": 2, "fn": 3,
                           "zero_error_images": 0}},
        "orientation": {"wide": {"images": 1, "errors": 5, "fp": 2, "fn": 3}},
        "images": [],
        "records": [{
            "image": "doc_sample", "name": "doc_sample.png",
            "preview": f"error_previews/{split}/{preview_name}",
            "path": "/tmp/doc_sample.png", "split": split, "source": "doc",
            "group": "doc", "width": 120, "height": 60, "gt": 2,
            "tp": 1, "fp": 1, "fn": 1, "error_count": 2, "error_rate": 0.5,
            "input_size": 768, "score_threshold": 0.7, "nms_iou": 0.3,
            "median_text_height": 10.0,
        }],
    }


def sample_report():
    """构造渲染测试所需的最小报告数据。"""
    metrics = summarize_counts([10, 2, 3])
    train_metrics = summarize_counts([200, 5, 4])
    point = {"input_size": 768, "threshold": 0.7, "overall": metrics,
             "doc": metrics, "ui": metrics}
    train_point = {"input_size": 768, "threshold": 0.7, "overall": train_metrics,
                   "doc": train_metrics, "ui": train_metrics}
    return {
        "generated_at": "2026-01-01 00:00",
        "run": {
            "name": "<script>alert(1)</script>",
            "epochs": 620,
            "best_f1": 0.97158,
            "best_f1_epoch": 560,
            "best_loss": 0.04613,
            "best_loss_epoch": 597,
            "last_f1": 0.96723,
            "training_hours": 18.5,
            "throughput": 52.5,
            "has_curves": True,
        },
        "curves": {},
        "dataset": {
            "train": {"images": 10, "doc": 1, "ui": 9, "other": 0, "label_lines": 100},
            "val": {"images": 5, "doc": 1, "ui": 4, "other": 0, "label_lines": 50},
        },
        "primary": {
            "checkpoint": "best_f1_detection_epoch_560_f1_0.97158.pth",
            "input_size": 768,
            "threshold": 0.7,
            "nms_iou": 0.3,
            "match_iou": 0.5,
        },
        "groups": {"overall": metrics, "doc": metrics, "ui": metrics},
        "train_groups": {"overall": train_metrics, "doc": train_metrics,
                         "ui": train_metrics},
        "workpoints": [point],
        "best_workpoint": point,
        "train_workpoints": [train_point],
        "train_best_workpoint": train_point,
        "checkpoints": [{
            "name": "best_f1_detection_epoch_560_f1_0.97158.pth",
            "path": "/tmp/model.pth",
            "kind": "best_f1",
            "epoch": 560,
            "f1": 0.97158,
            "loss": None,
            "size_mb": 48.0,
            "mtime": "2026-10-30 15:02",
            "eval": {"overall": metrics, "doc": metrics, "ui": metrics},
        }],
        "error_analysis": build_error_analysis_fixture("val", "doc_sample.jpg"),
        "train_error_analysis": build_error_analysis_fixture("train", "doc_train.jpg"),
        "review": None,
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
    assert "统一检测训练分析报告" in html
    assert "部署检查清单" in html
    assert "指标口径" in html
    assert "训练集（in-sample）" in html
    assert html.count("逐图错误明细") == 2
    assert "error_previews/val/doc_sample.jpg" in html
    assert "error_previews/train/doc_train.jpg" in html
    assert '<table class="error-detail-table" data-split="train"' in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_report_summary_drops_curves_and_image_data():
    """summary.json 不应包含曲线、图片 data URI 和逐图错误明细。"""
    report = sample_report()
    summary = report_summary(report)
    assert "curves" not in summary
    assert "error_analysis" not in summary
    assert "train_error_analysis" not in summary
    assert summary["error_summary"]["summary"]["total_errors"] == 5
    assert summary["train_error_summary"]["summary"]["total_errors"] == 5
    assert "records" not in summary["error_summary"]
    assert "records" not in summary["train_error_summary"]
    assert summary["train_groups"]["overall"]["tp"] == 200


def test_build_error_analysis_writes_preview_and_details(tmp_path):
    """有 FP/FN 的图片应生成原尺寸预览、逐图明细和两份 CSV，并清理旧预览。"""
    image_path = tmp_path / "doc_sample.png"
    Image.new("RGB", (120, 60), "white").save(image_path)
    combo = (768, 0.7)
    entry = {
        "path": str(image_path),
        "group": "doc",
        "width": 120,
        "height": 60,
        "gt": 2,
        "truth_boxes": np.array([[0, 0, 20, 10], [30, 0, 50, 10]], dtype=np.float32),
        "median_gt_height": 10.0,
        "metadata": {768: (1.0, (0, 0))},
        "combos": {combo: {
            "predictions": [{"box": [0, 0, 20, 10], "score": 0.9},
                            {"box": [80, 40, 110, 58], "score": 0.8}],
            "status": [1, 0],
            "used": [0],
            "counts": [1, 1, 1],
        }},
    }
    preview_dir = tmp_path / "error_previews" / "train"
    analysis = build_error_analysis({"doc_sample.png": entry}, combo, "train",
                                    preview_dir=preview_dir, preview_root=tmp_path)

    assert analysis["summary"]["total_errors"] == 2
    record = analysis["records"][0]
    assert record["image"] == "doc_sample"
    assert record["split"] == "train"
    assert (record["fp"], record["fn"], record["error_count"]) == (1, 1, 2)
    assert record["preview"] == "error_previews/train/doc_sample.jpg"
    assert (tmp_path / record["preview"]).is_file()

    paths = write_error_details(analysis["records"], tmp_path, "train")
    assert [path.name for path in paths] == ["errors_train.csv", "error_previews_train.csv"]
    with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["split"] == "train"
    assert rows[0]["fp"] == "1" and rows[0]["fn"] == "1"

    stale = preview_dir / "old_checkpoint.jpg"
    stale.write_bytes(b"stale")
    build_error_analysis({"doc_sample.png": entry}, combo, "train",
                         preview_dir=preview_dir, preview_root=tmp_path)
    assert not stale.exists()


def test_json_safe_converts_numpy_values():
    """numpy 标量和数组应转换为 Python 原生类型。"""
    result = json_safe({"count": np.int64(3), "value": np.float32(0.5),
                        "array": np.array([1, 2])})
    assert result == {"count": 3, "value": 0.5, "array": [1, 2]}
