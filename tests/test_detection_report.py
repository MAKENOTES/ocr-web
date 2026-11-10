"""统一检测分析报告和框匹配状态的单元测试。"""
import numpy as np

from detection.components.eval_metrics import match_box_status, match_counts, summarize_counts
from detection.components.training_report import (
    build_workpoints,
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


def sample_report():
    """构造渲染测试所需的最小报告数据。"""
    metrics = summarize_counts([10, 2, 3])
    point = {"input_size": 768, "threshold": 0.7, "overall": metrics,
             "doc": metrics, "ui": metrics}
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
        "workpoints": [point],
        "best_workpoint": point,
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
        "error_analysis": {
            "summary": {
                "images": 5, "gt": 50, "total_errors": 5, "error_images": 2,
                "zero_error_images": 3, "top_ten_errors": 5, "top_ten_share": 1.0,
            },
            "groups": {"doc": {"images": 1, "errors": 5, "fp": 2, "fn": 3,
                               "zero_error_images": 0}},
            "orientation": {"wide": {"images": 1, "errors": 5, "fp": 2, "fn": 3}},
            "images": [],
        },
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
