"""在训练/验证集上运行两个部署推理入口，生成与训练指标对比的静态 HTML 报告。

用法：
    .venv/bin/python detection/inference_report.py
    .venv/bin/python detection/inference_report.py --skip-inference

输出：
    detection/data/inference_eval/<backend>_<split>/{images,labels,run_summary.json}
    detection/reports/<run>/inference/{index.html,summary.json}

默认行为会先检查已有预测标签，数量匹配时跳过，便于中断后继续；需要强制重跑时
使用 --force。报告按入口默认配置（768px 动态、阈值 0.7、NMS 0.3）评估，不修改
任何入口常量。
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录；报告内所有默认路径都相对它解析。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detection_inference_onnx as onnx_entry  # 根目录独立 ONNX/OpenVINO 入口。

from detection.components.image import IMAGE_EXTENSIONS
from detection.components.inference_report import (
    build_report,
    evaluate_split,
    load_image_index,
    write_report,
)
from detection.components.training_report import load_curves
from detection.components.transforms import normalize_input_size
from detection.inference import (
    DYNAMIC_INPUT as PYTORCH_DYNAMIC_INPUT,
    INPUT_SIZE as PYTORCH_INPUT_SIZE,
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
    """确认 ONNX 导出尺寸与 PyTorch 入口一致，避免对比工作点错配。"""
    pytorch_size = normalize_input_size(PYTORCH_INPUT_SIZE)
    if not PYTORCH_DYNAMIC_INPUT:
        raise ValueError("detection/inference.py 未启用动态输入，报告口径不一致")
    runtime = onnx_entry.load_backend(ONNX_MODEL, ONNX_BACKEND)
    dynamic, max_input_size, _ = onnx_entry.exported_shape(runtime)
    if not dynamic or tuple(max_input_size) != pytorch_size:
        raise ValueError(
            f"ONNX 导出动态={dynamic}、最大尺寸={max_input_size}，与 PyTorch 入口 "
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
                "threshold": SCORE_THRESHOLD,
                "groups": data["groups"],
                "records": data["records"],
                "pred_boxes": data["pred_boxes"],
                "image_index": index,
                "timing": {key: summary[key] for key in TIMING_KEYS},
                "error_analysis": data["error_analysis"],
            }
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--skip-inference", action="store_true",
                        help="复用已有预测标签，只重新生成报告")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有预测标签，强制重跑四次推理")
    args = parser.parse_args(argv)
    if args.skip_inference and args.force:
        parser.error("--skip-inference 与 --force 不能同时使用")

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
    return report


if __name__ == "__main__":
    main()
