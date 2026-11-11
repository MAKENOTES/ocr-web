"""在训练/验证集上运行两个部署推理入口，生成与训练指标对比的静态 HTML 报告。

用法：
    .venv/bin/python detection/inference_report.py
    .venv/bin/python detection/inference_report.py --skip-inference
    .venv/bin/python detection/inference_report.py --error-previews

输出：
    detection/data/inference_eval/<backend>_<split>/{images,labels,run_summary.json}
    detection/reports/<run>/inference/{index.html,summary.json}
    detection/reports/<run>/inference/{errors_train.csv,errors_val.csv}

--error-previews 会对 errors_*.csv 中的图片再用检测包 PyTorch 入口推理，输出：
    detection/reports/<run>/inference/error_previews/<split>/<原文件名>
    detection/reports/<run>/inference/error_previews_<split>.csv

预览图直接复用 detection/inference.py 的红色预测框、原尺寸和原格式，与单独运行推理
入口得到的预览效果一致。默认行为会先检查已有预测标签，数量匹配时跳过，便于中断后
继续；预览同样按已有 CSV 和图片判断完整性，--force-previews 只重算预览，--force
在重跑四次推理时一并重算预览。报告按入口默认配置（固定 768×768、阈值 0.7、NMS 0.3）
评估，不修改任何入口常量。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录；报告内所有默认路径都相对它解析。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detection_inference_onnx as onnx_entry  # 根目录独立 ONNX/OpenVINO 入口。

from detection.components.dataset import read_boxes
from detection.components.eval_metrics import match_box_status
from detection.components.image import IMAGE_EXTENSIONS, load_rgb_image
from detection.utils.inference_report import (
    ERROR_PREVIEW_DIRNAME,
    MATCH_IOU,
    build_report,
    evaluate_split,
    load_image_index,
    read_error_rows,
    save_error_preview,
    source_group,
    write_preview_csv,
    write_report,
)
from detection.utils.training_report import load_curves
from detection.components.transforms import normalize_input_size
from detection.inference import (
    DYNAMIC_INPUT as PYTORCH_DYNAMIC_INPUT,
    INPUT_SIZE as PYTORCH_INPUT_SIZE,
    DocumentDetector,
    run_inference,
)


DETECTION_DIR = REPO_ROOT / "detection"  # 统一检测包目录。
RUN_NAME = "ocr_experiment_1793283532819507511"  # 本次对照的训练运行名。
CHECKPOINT_PATH = (DETECTION_DIR / "checkpoints" / RUN_NAME
                   / "best_f1_detection_epoch_560_f1_0.97158.pth")  # PyTorch 推理检查点。
ONNX_MODEL = REPO_ROOT / "best_detection.onnx"  # 已导出的 ONNX 模型。
OUTPUT_ROOT = DETECTION_DIR / "data" / "inference_eval"  # 预测图片和标签的输出根目录，不提交。
REPORT_DIR = DETECTION_DIR / "reports" / RUN_NAME / "inference"  # 报告输出目录，不提交。
TRAINING_SUMMARY = DETECTION_DIR / "reports" / RUN_NAME / "summary.json"  # 训练期指标摘要。
TRAIN_LOG_DIR = DETECTION_DIR / "runs" / RUN_NAME  # TensorBoard 事件目录。
SPLITS = {
    "train": (DETECTION_DIR / "data" / "train_images",
              DETECTION_DIR / "data" / "train_labels"),
    "val": (DETECTION_DIR / "data" / "val_images",
            DETECTION_DIR / "data" / "val_labels"),
}  # 图片目录与严格配对的真值标签目录。
BACKENDS = ("pytorch", "onnx")  # 两次推理分别使用包内 PyTorch 入口和独立 ONNX 入口。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，与两个入口的部署默认值一致。
NMS_IOU = 0.3  # 重复框抑制的交并比阈值，与两个入口一致。
ONNX_BACKEND = "openvino"  # 独立入口默认的 CPU 后端；ONNX Runtime 在当前机器上明显更慢。
WARMUP_RUNS = 3  # 正式计时前的模型前向预热次数。
TIMING_KEYS = ("images", "boxes", "mean_infer_ms", "mean_detect_ms",
               "wall_seconds", "images_per_second")  # 写入报告的入口耗时字段。


def count_images(directory):
    """统计目录内支持的图片数量，用于判断已有预测是否完整。"""
    return sum(1 for path in Path(directory).iterdir()
               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def run_inference_entries(image_indexes, force):
    """按后端和划分调用两个部署入口；已有完整预测时跳过。"""
    for split, (image_dir, _) in SPLITS.items():
        expected = len(image_indexes[split])
        for backend in BACKENDS:
            output_dir = OUTPUT_ROOT / f"{backend}_{split}"
            labels_dir = output_dir / "labels"
            cached = (len(list(labels_dir.glob("gt_*.txt"))) if labels_dir.is_dir() else 0)
            summary_path = output_dir / "run_summary.json"
            if not force and cached == expected and summary_path.is_file():
                print(f"[跳过] {backend} {split}: 已缓存 {cached}/{expected} 张", flush=True)
                continue
            print(f"[推理] {backend} {split}: {expected} 张 -> {output_dir}", flush=True)

            if backend == "pytorch":
                summary = run_inference(
                    CHECKPOINT_PATH, image_dir, output_dir, "auto",
                    threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU,
                    backend="pytorch", warmup_runs=WARMUP_RUNS)
            else:
                runtime = onnx_entry.load_backend(ONNX_MODEL, ONNX_BACKEND)
                dynamic, max_input_size, fixed_input_size = onnx_entry.exported_shape(runtime)
                summary = onnx_entry.run_onnx_inference(
                    runtime, image_dir, output_dir,
                    dynamic=dynamic, max_input_size=max_input_size,
                    fixed_input_size=fixed_input_size,
                    threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU,
                    warmup_runs=WARMUP_RUNS)

            summary.update({"backend": backend, "split": split,
                            "score_threshold": SCORE_THRESHOLD, "nms_iou": NMS_IOU,
                            "model": str(CHECKPOINT_PATH if backend == "pytorch"
                                         else ONNX_MODEL),
                            "image_dir": str(image_dir)})
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                    encoding="utf-8")


def check_exported_size():
    """确认 ONNX 导出的高宽模式与尺寸和 PyTorch 入口一致，避免对比工作点错配。"""
    pytorch_size = normalize_input_size(PYTORCH_INPUT_SIZE)
    runtime = onnx_entry.load_backend(ONNX_MODEL, ONNX_BACKEND)
    dynamic, max_input_size, fixed_input_size = onnx_entry.exported_shape(runtime)
    onnx_size = tuple(max_input_size if dynamic else fixed_input_size)
    if dynamic != PYTORCH_DYNAMIC_INPUT or onnx_size != pytorch_size:
        raise ValueError(
            f"ONNX 导出动态={dynamic}、尺寸={onnx_size}，与 PyTorch 入口 "
            f"动态={PYTORCH_DYNAMIC_INPUT}、尺寸={pytorch_size} 不一致")


def collect_runs(image_indexes):
    """读取入口写出的标签并评估，返回报告所需的四次运行数据。"""
    runs = {}
    for split, (image_dir, gt_label_dir) in SPLITS.items():
        for backend in BACKENDS:
            output_dir = OUTPUT_ROOT / f"{backend}_{split}"
            summary_path = output_dir / "run_summary.json"
            labels_dir = output_dir / "labels"
            if not summary_path.is_file() or not labels_dir.is_dir():
                raise FileNotFoundError(
                    f"缺少 {backend} {split} 的推理产物: {output_dir}；"
                    f"先去掉 --skip-inference 重新运行")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            index = image_indexes[split]
            print(f"[评估] {backend} {split}: {len(index)} 张", flush=True)
            data = evaluate_split(
                index, gt_label_dir, labels_dir,
                collect_visuals=(backend == "pytorch" and split == "val"))
            runs[(backend, split)] = {
                "backend": backend,
                "split": split,
                "images": len(index),
                "boxes": int(summary["boxes"]),
                "input_size": max(normalize_input_size(PYTORCH_INPUT_SIZE)),
                "dynamic_input": PYTORCH_DYNAMIC_INPUT,
                "threshold": SCORE_THRESHOLD,
                "groups": data["groups"],
                "records": data["records"],
                "pred_boxes": data["pred_boxes"],
                "image_index": index,
                "timing": {key: summary[key] for key in TIMING_KEYS},
                "error_analysis": data["error_analysis"],
            }
    return runs


def preview_cache_complete(rows, output_csv):
    """已有预览 CSV 覆盖同一批图片且预览文件齐全时视为完整。"""
    if not output_csv.is_file():
        return False
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        cached = list(csv.DictReader(handle))
    if {row.get("image") for row in cached} != {row["image"] for row in rows}:
        return False
    return all((output_csv.parent / (row.get("preview") or "")).is_file()
               for row in cached)


def remove_stale_previews(preview_dir, keep_names):
    """删除本次错误清单之外的旧预览图，避免不同推理批次的产物混杂。"""
    preview_dir = Path(preview_dir)
    if not preview_dir.is_dir():
        return 0
    removed = 0
    for path in preview_dir.iterdir():
        if (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.name not in keep_names):
            path.unlink()
            removed += 1
    return removed


def run_error_previews(csv_paths, force):
    """对错误明细 CSV 中的图片用检测包 PyTorch 入口重新推理并输出预览图。

    预览图写入 CSV 同层的 error_previews/<split>/，并把本轮推理的 TP/FP/FN、
    耗时和相对预览路径记录到 error_previews_<split>.csv。已有完整产物时跳过，
    force=True 强制重算。
    """
    detector = DocumentDetector(CHECKPOINT_PATH, "auto", backend="pytorch")
    detector.warmup(WARMUP_RUNS)
    input_size = max(normalize_input_size(PYTORCH_INPUT_SIZE))
    outputs = []

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        split = csv_path.stem.removeprefix("errors_")
        if split not in SPLITS:
            raise ValueError(f"无法从文件名确定数据划分: {csv_path}")
        rows = read_error_rows(csv_path)
        preview_dir = csv_path.parent / ERROR_PREVIEW_DIRNAME / split
        removed = remove_stale_previews(
            preview_dir, {Path(row["path"]).name for row in rows})
        if removed:
            print(f"[清理] 错误预览 {split}: 删除 {removed} 张旧图", flush=True)
        output_csv = csv_path.parent / f"error_previews_{split}.csv"
        if not force and preview_cache_complete(rows, output_csv):
            print(f"[跳过] 错误预览 {split}: 已有 {len(rows)} 张", flush=True)
            outputs.append(output_csv)
            continue
        print(f"[预览] {split}: {len(rows)} 张 -> {preview_dir}", flush=True)

        gt_label_dir = SPLITS[split][1]
        records = []
        for index, row in enumerate(rows, 1):
            image_path = Path(row["path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"错误明细中的图片不存在: {image_path}")

            image = load_rgb_image(image_path)
            started = time.perf_counter()
            predictions = detector.predict(image, SCORE_THRESHOLD, NMS_IOU)
            detect_ms = (time.perf_counter() - started) * 1000

            gt_path = gt_label_dir / f"gt_{row['image']}.txt"
            if not gt_path.is_file():
                raise FileNotFoundError(f"缺少真值标签: {gt_path}")
            boxes, ignored = read_boxes(gt_path, image.size)
            _, _, counts = match_box_status(predictions, boxes, ignored, MATCH_IOU)
            truth = boxes[~ignored]
            tp, fp, fn = (int(value) for value in counts)

            preview_path = save_error_preview(image, preview_dir / image_path.name, predictions)
            records.append({
                "image": row["image"],
                "preview": preview_path.relative_to(csv_path.parent).as_posix(),
                "path": str(image_path),
                "split": split,
                "source": source_group(row["image"]),
                "width": image.size[0],
                "height": image.size[1],
                "gt": int(len(truth)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "error_count": fp + fn,
                "error_rate": round((fp + fn) / max(1, len(truth)), 6),
                "input_size": input_size,
                "score_threshold": SCORE_THRESHOLD,
                "nms_iou": NMS_IOU,
                "infer_ms": round(detector.last_infer_ms, 2),
                "detect_ms": round(detect_ms, 2),
            })
            if index % 20 == 0 or index == len(rows):
                print(f"[预览] {split}: {index}/{len(rows)}", flush=True)

        outputs.append(write_preview_csv(records, output_csv))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--skip-inference", action="store_true",
                        help="复用已有预测标签，只重新生成报告")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有预测标签，强制重跑四次推理")
    parser.add_argument("--error-previews", action="store_true",
                        help="报告生成后对 errors_*.csv 中的图片重新推理并输出预览图")
    parser.add_argument("--force-previews", action="store_true",
                        help="忽略已有预览 CSV 和图片，只强制重算错误预览")
    args = parser.parse_args(argv)
    if args.skip_inference and args.force:
        parser.error("--skip-inference 与 --force 不能同时使用")
    if args.force_previews and not args.error_previews:
        parser.error("--force-previews 需要与 --error-previews 同时使用")

    for path in (CHECKPOINT_PATH, ONNX_MODEL, TRAINING_SUMMARY):
        if not path.is_file():
            parser.error(f"缺少必要文件: {path}")

    image_indexes = {split: load_image_index(image_dir)
                     for split, (image_dir, _) in SPLITS.items()}
    check_exported_size()
    if not args.skip_inference:
        run_inference_entries(image_indexes, args.force)

    runs = collect_runs(image_indexes)
    training = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8"))
    curves = load_curves(TRAIN_LOG_DIR)
    config = {
        "checkpoint": CHECKPOINT_PATH.name,
        "onnx_model": ONNX_MODEL.name,
        "input_size": max(normalize_input_size(PYTORCH_INPUT_SIZE)),
        "dynamic_input": PYTORCH_DYNAMIC_INPUT,
        "score_threshold": SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "onnx_backend": ONNX_BACKEND,
    }
    report = build_report(runs, training, curves, config, RUN_NAME)
    html_path, summary_path, csv_paths = write_report(report, REPORT_DIR)
    print(f"报告已生成: {html_path}")
    print(f"数值摘要: {summary_path}")
    for csv_path in csv_paths:
        print(f"错误图片: {csv_path}")
    if args.error_previews:
        for preview_csv in run_error_previews(csv_paths,
                                             args.force or args.force_previews):
            print(f"错误预览记录: {preview_csv}")
    return report


if __name__ == "__main__":
    main()
