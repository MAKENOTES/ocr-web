"""输入尺寸策略报告的尺寸判定和耗时汇总测试。"""
from detection.utils.size_policy_eval import select_input_size, timing_summary


def test_select_input_size_uses_256_only_for_small_both_sides():
    """宽高都小于 256 才用 256；任一边达到 256 仍用 768。"""
    assert select_input_size(100, 50) == 256
    assert select_input_size(255, 255) == 256
    assert select_input_size(256, 100) == 768
    assert select_input_size(100, 256) == 768
    assert select_input_size(300, 50) == 768


def test_timing_summary_computes_mean_and_group_split():
    """总耗时、吞吐和小图/大图均值按累计值计算。"""
    summary = timing_summary({
        "images": 2, "total_ms": 12.0,
        "small_images": 1, "small_ms": 4.0,
        "large_images": 1, "large_ms": 8.0,
    })

    assert summary["mean_detect_ms"] == 6.0
    assert summary["total_seconds"] == 0.012
    assert summary["images_per_second"] == 2 / 0.012
    assert summary["small_mean_detect_ms"] == 4.0
    assert summary["large_mean_detect_ms"] == 8.0
