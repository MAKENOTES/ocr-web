"""对比“小图 256、其余 768”与全固定 768 的精度和耗时，生成静态 HTML 报告。

用法：
    .venv/bin/python detection/size_policy_report.py
    .venv/bin/python detection/size_policy_report.py --use-cache

测试集为 detection/data/train_images 和 detection/data/val_images。PyTorch 覆盖
训练集和验证集；ONNX/OpenVINO 只在验证集上复算 CPU 耗时和精度。小图判定为宽和高
都小于 256px，其他图片仍使用 768×768。输出在 detection/reports/<run>/size_policy/。
--use-cache 复用上一轮写出的 raw_metrics.json，只重新渲染报告。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detection_export_onnx as exporter  # 复用固定尺寸 ONNX 导出。
import detection_inference_onnx as onnx_entry  # 复用独立 ONNX/OpenVINO 推理。

from detection.components.dataset import read_boxes
from detection.components.eval_metrics import match_box_status, summarize_counts
from detection.components.image import load_rgb_image
from detection.components.inference_report import load_image_index
from detection.components.size_policy_report import (
    BASELINE_POLICY,
    LARGE_INPUT_SIZE,
    MIXED_POLICY,
    SMALL_INPUT_SIZE,
    SMALL_MAX_SIDE,
    timing_summary,
    write_report,
)
from detection.components.training_report import format_percent
from detection.inference import DocumentDetector


DETECTION_DIR = REPO_ROOT / "detection"  # 统一检测包目录。
RUN_NAME = "ocr_experiment_1793283532819507511"  # 对照的训练运行名。
CHECKPOINT_PATH = (DETECTION_DIR / "checkpoints" / RUN_NAME
                   / "best_f1_detection_epoch_560_f1_0.97158.pth")  # 统一检测检查点。
ONNX_768 = REPO_ROOT / "best_detection.onnx"  # 已导出的固定 768×768 ONNX。
ONNX_256 = DETECTION_DIR / "data" / "inference_eval" / "models" / "best_detection_256.onnx"  # 小图固定 256×256 ONNX，本地产物。
REPORT_DIR = DETECTION_DIR / "reports" / RUN_NAME / "size_policy"  # 报告输出目录，不提交。
CACHE_PATH = REPORT_DIR / "raw_metrics.json"  # 推理指标缓存；--use-cache 时只重新渲染报告。
SPLITS = {
    "train": (DETECTION_DIR / "data" / "train_images",
              DETECTION_DIR / "data" / "train_labels"),
    "val": (DETECTION_DIR / "data" / "val_images",
            DETECTION_DIR / "data" / "val_labels"),
}  # 图片目录和严格配对的真值标签目录。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，与部署入口一致。
NMS_IOU = 0.3  # 重复框抑制阈值，与部署入口一致。
MATCH_IOU = 0.5  # 预测框与真值框匹配阈值，与训练期指标一致。
WARMUP_RUNS = 3  # PyTorch 每个输入尺寸的预热次数。
ONNX_BACKEND = "openvino"  # ONNX CPU 后端；与部署入口默认一致。
GROUP_KEYS = ("overall", "small", "large", "ui", "doc", "other")  # 需要汇总的分组。


def source_group(name):
    """按文件名前缀划分业务来源，未知前缀归入 other。"""
    prefix = name.split("_", 1)[0]
    return prefix if prefix in {"ui", "doc"} else "other"


def size_group(width, height):
    """按原图宽高判定小图/大图分组，与输入尺寸策略使用同一阈值。"""
    return "small" if width < SMALL_MAX_SIDE and height < SMALL_MAX_SIDE else "large"


def new_totals():
    """初始化各分组的 TP/FP/FN 计数。"""
    return {group: {"counts": [0, 0, 0], "images": 0} for group in GROUP_KEYS}


def accumulate(totals, counts, *groups):
    """把一张图片的计数累加到总体和给定分组；overall 只加一次。"""
    for key in ("overall", *groups):
        totals[key]["counts"][0] += int(counts[0])
        totals[key]["counts"][1] += int(counts[1])
        totals[key]["counts"][2] += int(counts[2])
        totals[key]["images"] += 1


def summarize_totals(totals):
    """把累计计数转换为报告指标，跳过没有图片的分组。"""
    return {group: summarize_counts(item["counts"])
            for group, item in totals.items() if item["images"] > 0}


def dataset_summary(image_index):
    """统计图片数、小图数和策略下的像素计算量比例。"""
    small = sum(1 for _, (_, size) in image_index.items()
                if size[0] < SMALL_MAX_SIDE and size[1] < SMALL_MAX_SIDE)
    images = len(image_index)
    pixel_ratio = ((small * SMALL_INPUT_SIZE ** 2
                    + (images - small) * LARGE_INPUT_SIZE ** 2)
                   / (images * LARGE_INPUT_SIZE ** 2)) if images else 0.0
    return {"images": images, "small": small, "large": images - small,
            "small_ratio": small / images if images else 0.0,
            "pixel_ratio": pixel_ratio}


def counts_for(predictions, gt_path, image_size):
    """读取真值并返回预测的 TP/FP/FN。"""
    boxes, ignored = read_boxes(gt_path, image_size)
    return match_box_status(predictions, boxes, ignored, MATCH_IOU)[2]


def run_pytorch_split(split, image_index, gt_label_dir):
    """在单个划分上同时跑全 768 和混合策略，返回指标和耗时。"""
    detector = DocumentDetector(CHECKPOINT_PATH, "auto", dynamic_input=False)
    for size in (LARGE_INPUT_SIZE, SMALL_INPUT_SIZE):
        detector.max_input_size = (size, size)
        detector.warmup(WARMUP_RUNS)

    totals = {BASELINE_POLICY: new_totals(), MIXED_POLICY: new_totals()}
    timing = {policy: {"images": 0, "total_ms": 0.0, "small_images": 0,
                       "small_ms": 0.0, "large_images": 0, "large_ms": 0.0}
              for policy in (BASELINE_POLICY, MIXED_POLICY)}

    for index, (stem, (path, image_size)) in enumerate(image_index.items(), 1):
        image = load_rgb_image(path)
        gt_path = gt_label_dir / f"gt_{stem}.txt"
        group = size_group(*image_size)
        source = source_group(stem)

        detector.max_input_size = (LARGE_INPUT_SIZE, LARGE_INPUT_SIZE)
        started = time.perf_counter()
        baseline_predictions = detector.predict(image, SCORE_THRESHOLD, NMS_IOU)
        baseline_ms = (time.perf_counter() - started) * 1000
        baseline_counts = counts_for(baseline_predictions, gt_path, image.size)
        accumulate(totals[BASELINE_POLICY], baseline_counts, group, source)
        timing[BASELINE_POLICY]["images"] += 1
        timing[BASELINE_POLICY]["total_ms"] += baseline_ms
        timing[BASELINE_POLICY][f"{group}_images"] += 1
        timing[BASELINE_POLICY][f"{group}_ms"] += baseline_ms

        if group == "small":
            detector.max_input_size = (SMALL_INPUT_SIZE, SMALL_INPUT_SIZE)
            started = time.perf_counter()
            policy_predictions = detector.predict(image, SCORE_THRESHOLD, NMS_IOU)
            policy_ms = (time.perf_counter() - started) * 1000
            policy_counts = counts_for(policy_predictions, gt_path, image.size)
        else:
            policy_counts, policy_ms = baseline_counts, baseline_ms
        accumulate(totals[MIXED_POLICY], policy_counts, group, source)
        timing[MIXED_POLICY]["images"] += 1
        timing[MIXED_POLICY]["total_ms"] += policy_ms
        timing[MIXED_POLICY][f"{group}_images"] += 1
        timing[MIXED_POLICY][f"{group}_ms"] += policy_ms

        if index % 500 == 0:
            print(f"[PyTorch {split}] {index}/{len(image_index)}", flush=True)

    return {policy: {"groups": summarize_totals(totals[policy]),
                     "latency": timing_summary(timing[policy])}
            for policy in (BASELINE_POLICY, MIXED_POLICY)}


def load_onnx_backends():
    """加载固定 768 和固定 256 的 OpenVINO 后端；缺少 256 模型时先导出。"""
    if not ONNX_256.is_file():
        print(f"[导出] 固定 {SMALL_INPUT_SIZE}×{SMALL_INPUT_SIZE} ONNX: {ONNX_256}", flush=True)
        exporter.BENCHMARK_ENABLED = False
        exporter.export_model(CHECKPOINT_PATH, ONNX_256,
                              input_size=SMALL_INPUT_SIZE, dynamic_spatial=False)
    backends = {}
    for name, model_path in ((LARGE_INPUT_SIZE, ONNX_768), (SMALL_INPUT_SIZE, ONNX_256)):
        runtime = onnx_entry.load_backend(model_path, ONNX_BACKEND)
        dynamic, max_input_size, fixed_input_size = onnx_entry.exported_shape(runtime)
        if dynamic or tuple(fixed_input_size) != (name, name):
            raise ValueError(f"ONNX 模型不是固定 {name}×{name}: {model_path}")
        onnx_entry.warmup_backend(runtime, dynamic, max_input_size, fixed_input_size, 2)
        backends[name] = (runtime, dynamic, max_input_size, fixed_input_size)
    return backends


def run_onnx_val(image_index, gt_label_dir):
    """在验证集上同时跑 ONNX 全 768 和混合策略，返回指标和 CPU 耗时。"""
    backends = load_onnx_backends()
    totals = {BASELINE_POLICY: new_totals(), MIXED_POLICY: new_totals()}
    timing = {policy: {"images": 0, "total_ms": 0.0, "small_images": 0,
                       "small_ms": 0.0, "large_images": 0, "large_ms": 0.0}
              for policy in (BASELINE_POLICY, MIXED_POLICY)}

    for index, (stem, (path, image_size)) in enumerate(image_index.items(), 1):
        image = load_rgb_image(path)
        gt_path = gt_label_dir / f"gt_{stem}.txt"
        group = size_group(*image_size)
        source = source_group(stem)

        runtime, dynamic, max_input_size, fixed_input_size = backends[LARGE_INPUT_SIZE]
        started = time.perf_counter()
        baseline_predictions, _ = onnx_entry.predict_image(
            runtime, image, SCORE_THRESHOLD, NMS_IOU, dynamic,
            max_input_size, fixed_input_size)
        baseline_ms = (time.perf_counter() - started) * 1000
        baseline_counts = counts_for(baseline_predictions, gt_path, image.size)
        accumulate(totals[BASELINE_POLICY], baseline_counts, group, source)
        timing[BASELINE_POLICY]["images"] += 1
        timing[BASELINE_POLICY]["total_ms"] += baseline_ms
        timing[BASELINE_POLICY][f"{group}_images"] += 1
        timing[BASELINE_POLICY][f"{group}_ms"] += baseline_ms

        if group == "small":
            runtime, dynamic, max_input_size, fixed_input_size = backends[SMALL_INPUT_SIZE]
            started = time.perf_counter()
            policy_predictions, _ = onnx_entry.predict_image(
                runtime, image, SCORE_THRESHOLD, NMS_IOU, dynamic,
                max_input_size, fixed_input_size)
            policy_ms = (time.perf_counter() - started) * 1000
            policy_counts = counts_for(policy_predictions, gt_path, image.size)
        else:
            policy_counts, policy_ms = baseline_counts, baseline_ms
        accumulate(totals[MIXED_POLICY], policy_counts, group, source)
        timing[MIXED_POLICY]["images"] += 1
        timing[MIXED_POLICY]["total_ms"] += policy_ms
        timing[MIXED_POLICY][f"{group}_images"] += 1
        timing[MIXED_POLICY][f"{group}_ms"] += policy_ms

        if index % 200 == 0:
            print(f"[ONNX val] {index}/{len(image_index)}", flush=True)

    return {policy: {"groups": summarize_totals(totals[policy]),
                     "latency": timing_summary(timing[policy])}
            for policy in (BASELINE_POLICY, MIXED_POLICY)}


def build_conclusions(report):
    """根据报告数据生成结论和上线建议。"""
    dataset = report["dataset"]["val"]
    pytorch_val = report["runs"]["pytorch"]["val"]
    base = pytorch_val[BASELINE_POLICY]
    policy = pytorch_val[MIXED_POLICY]
    base_f1 = base["groups"]["overall"]["f1"]
    policy_f1 = policy["groups"]["overall"]["f1"]
    delta = policy_f1 - base_f1
    small_base = base["groups"]["small"]["f1"]
    small_policy = policy["groups"]["small"]["f1"]
    base_ms = base["latency"]["mean_detect_ms"]
    policy_ms = policy["latency"]["mean_detect_ms"]
    conclusions = [
        f"小图（宽高都小于 {SMALL_MAX_SIDE}px）占验证集 "
        f"{format_percent(dataset['small_ratio'], 1)}（{dataset['small']}/{dataset['images']}），"
        f"策略的像素计算量约为全 768 的 {dataset['pixel_ratio']:.2f} 倍。",
        f"验证集 F1：固定 768 为 <strong>{base_f1:.5f}</strong>，混合策略为 "
        f"<strong>{policy_f1:.5f}</strong>，Δ {delta:+.5f}；小图组 "
        f"{small_base:.5f} → {small_policy:.5f}，大图组保持 768 不变。",
        f"PyTorch 验证集检测均值 {base_ms:.1f} → {policy_ms:.1f} ms/张"
        f"（{base_ms / policy_ms:.2f}×）；训练集 "
        f"{base['latency']['mean_detect_ms']:.1f} → "
        f"{policy['latency']['mean_detect_ms']:.1f} ms/张。",
    ]
    onnx_val = report["runs"].get("onnx", {}).get("val")
    if onnx_val:
        onnx_base = onnx_val[BASELINE_POLICY]
        onnx_policy = onnx_val[MIXED_POLICY]
        conclusions.append(
            f"ONNX/OpenVINO 验证集 F1：{onnx_base['groups']['overall']['f1']:.5f} → "
            f"{onnx_policy['groups']['overall']['f1']:.5f}；检测均值 "
            f"{onnx_base['latency']['mean_detect_ms']:.1f} → "
            f"{onnx_policy['latency']['mean_detect_ms']:.1f} ms/张"
            f"（{onnx_base['latency']['mean_detect_ms'] / onnx_policy['latency']['mean_detect_ms']:.2f}×）。")
    if delta >= -0.005:
        conclusions.append(
            "精度损失在 0.5 个百分点以内；如果业务接受该损失，可优先在 UI 小图场景启用，"
            "并用未参与调参的独立留出集复核后再上线。")
    else:
        conclusions.append(
            "精度损失超过 0.5 个百分点；不建议直接替换固定 768。若必须降低小图耗时，"
            "可只对低风险场景启用，或继续保留 768。")
    conclusions.append(
        "验证集参与过检查点选择，以上 F1 存在乐观偏差；ONNX 训练集未重跑，"
        "ONNX 结论来自验证集样本。")
    return conclusions


def main(argv=None):
    """按固定路径跑三种组合并生成报告；--use-cache 只重新渲染。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cache", action="store_true",
                        help="复用 raw_metrics.json，跳过推理")
    args = parser.parse_args(argv)
    for path in (CHECKPOINT_PATH, ONNX_768):
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要文件: {path}")

    if args.use_cache:
        if not CACHE_PATH.is_file():
            parser.error(f"缓存不存在: {CACHE_PATH}")
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        dataset, runs = cached["dataset"], cached["runs"]
        print(f"[缓存] 复用 {CACHE_PATH}", flush=True)
    else:
        print("[加载] 图片索引", flush=True)
        image_indexes = {split: load_image_index(image_dir)
                         for split, (image_dir, _) in SPLITS.items()}
        dataset = {split: dataset_summary(index)
                   for split, index in image_indexes.items()}
        runs = {"pytorch": {}, "onnx": {}}
        for split, (_, gt_dir) in SPLITS.items():
            print(f"[PyTorch] {split}: {len(image_indexes[split])} 张", flush=True)
            runs["pytorch"][split] = run_pytorch_split(split, image_indexes[split], gt_dir)
        print(f"[ONNX] val: {len(image_indexes['val'])} 张", flush=True)
        runs["onnx"]["val"] = run_onnx_val(image_indexes["val"], SPLITS["val"][1])
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps({"dataset": dataset, "runs": runs}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name": RUN_NAME,
        "config": {
            "checkpoint": CHECKPOINT_PATH.name,
            "onnx_768": ONNX_768.name,
            "onnx_256": ONNX_256.name,
            "small_max_side": SMALL_MAX_SIDE,
            "small_input_size": SMALL_INPUT_SIZE,
            "large_input_size": LARGE_INPUT_SIZE,
            "score_threshold": SCORE_THRESHOLD,
            "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU,
        },
        "dataset": dataset,
        "runs": runs,
    }
    report["conclusions"] = build_conclusions(report)
    html_path, summary_path = write_report(report, REPORT_DIR)
    print(f"报告已生成: {html_path}")
    print(f"数值摘要: {summary_path}")
    return report


if __name__ == "__main__":
    main()
