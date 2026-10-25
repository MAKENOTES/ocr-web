"""直接读取验证目录，报告总体及按图片格式分组的文字框指标。"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# 评估入口位于 components，直接运行时向上两级找到仓库根目录。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recognition.components.preprocessing import load_rgb_image
from detection_larger.components.dataset import OCRDataset, read_boxes
from detection_larger.inference import DocumentDetector


def box_overlap(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    other_area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + other_area - intersection, 1e-8), intersection / max(area, 1e-8)


def match_counts(predictions, boxes, ignored, iou_threshold=0.5):
    """按置信度一对一匹配；预测主要落在 ### 内时不计误检。"""
    truth, ignore_boxes = boxes[~ignored], boxes[ignored]
    used = set()
    true_positive, false_positive = 0, 0
    for prediction in sorted(predictions, key=lambda value: -value["score"]):
        box = np.asarray(prediction["box"], dtype=np.float32)
        overlap, _ = box_overlap(box, truth)
        candidates = [int(index) for index in np.argsort(-overlap) if index not in used and overlap[index] >= iou_threshold]
        if candidates:
            used.add(candidates[0])
            true_positive += 1
        elif len(ignore_boxes) and np.any(box_overlap(box, ignore_boxes)[1] >= 0.5):
            continue
        else:
            false_positive += 1
    return np.array([true_positive, false_positive, len(truth) - true_positive], dtype=np.int64)


def summarize_counts(counts):
    tp, fp, fn = map(int, counts)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-dir", default="detection_larger/data/val_images", help="验证图片目录")
    parser.add_argument("--label-dir", default="detection_larger/data/val_labels", help="验证标签目录")
    parser.add_argument("--label-format", choices=("bbox4", "quad8"), default="bbox4")
    parser.add_argument("--model", required=True, help="文档模型 checkpoint 或 ONNX")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--threshold", type=float, default=0.7, help="文字置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="重复框抑制阈值")
    parser.add_argument("--match-iou", type=float, default=0.5, help="预测框与真值的匹配阈值")
    args = parser.parse_args()
    if not 0 < args.match_iou <= 1:
        parser.error("--match-iou must be in (0, 1]")
    dataset = OCRDataset(args.image_dir, args.label_dir, label_format=args.label_format)
    detector = DocumentDetector(args.model, args.device)
    totals = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for image_path, label_path in zip(dataset.image_paths, dataset.label_paths):
        image = load_rgb_image(image_path)
        boxes, ignored = read_boxes(label_path, image.size, args.label_format)
        predictions = detector.predict(image, args.threshold, args.nms_iou)
        # 扩展名是可核实的分组字段，不把文件格式推断成业务来源。
        totals[image_path.suffix.lower()] += match_counts(predictions, boxes, ignored, args.match_iou)
    report = {"formats": {name: summarize_counts(counts) for name, counts in totals.items()},
              "overall": summarize_counts(sum(totals.values()))}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
