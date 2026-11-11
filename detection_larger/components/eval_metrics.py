"""大图检测的框级评估：解码后按置信度一对一匹配，统计 Precision/Recall/F1。

训练期指标、目录级评估和报告共用本模块；坐标在还原到原图后与标签比较，
因此不同输入尺寸和补边方式可以直接对比。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# 评估入口位于 components，直接运行时向上两级找到仓库根目录。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detection_larger.components.dataset import OCRDataset, read_boxes
from detection_larger.components.image import load_rgb_image
from detection_larger.components.preprocessing import prepare_image, restore_boxes
from detection_larger.inference import DocumentDetector
from detection_larger.utils.postprocess import (
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    decode_maps,
    suppress_boxes,
)


DEFAULT_MATCH_IOU = 0.5  # 预测框与真值框匹配的 IoU 下限。


def format_group(name):
    """按图片扩展名返回可核实的分组字段，不把格式推断成业务来源。"""
    return Path(name).suffix.lower() or "other"


def box_overlap(box, boxes):
    """返回 (IoU, 交集占预测框面积比)；boxes 为空时返回空数组。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    intersection = np.maximum(
        0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    other_area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return (intersection / np.maximum(area + other_area - intersection, 1e-8),
            intersection / max(area, 1e-8))


def match_box_status(predictions, boxes, ignored, iou_threshold=DEFAULT_MATCH_IOU):
    """按置信度一对一匹配，返回逐框状态、命中真值索引和 TP/FP/FN。

    状态与输入 predictions 同序：1 表示 TP，0 表示 FP，-1 表示预测主要
    落在 ### 忽略区内、既不计 TP 也不计 FP。命中真值索引对应 boxes 去掉
    忽略区域后的顺序，供报告标出漏检框。
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    ignored = np.asarray(ignored, dtype=bool)
    truth, ignore_boxes = boxes[~ignored], boxes[ignored]
    used = set()
    status = np.zeros(len(predictions), dtype=np.int8)
    order = sorted(range(len(predictions)), key=lambda index: -predictions[index]["score"])

    for index in order:
        box = np.asarray(predictions[index]["box"], dtype=np.float32)
        overlap, _ = box_overlap(box, truth)
        candidates = [int(truth_index) for truth_index in np.argsort(-overlap)
                      if truth_index not in used and overlap[truth_index] >= iou_threshold]
        if candidates:
            used.add(candidates[0])
            status[index] = 1
        elif len(ignore_boxes) and np.any(box_overlap(box, ignore_boxes)[1] >= 0.5):
            status[index] = -1
        else:
            status[index] = 0

    counts = np.array([int(np.count_nonzero(status == 1)),
                       int(np.count_nonzero(status == 0)),
                       len(truth) - len(used)], dtype=np.int64)
    return status, used, counts


def match_counts(predictions, boxes, ignored, iou_threshold=DEFAULT_MATCH_IOU):
    """按置信度一对一匹配；预测主要落在 ### 内时不计误检。"""
    return match_box_status(predictions, boxes, ignored, iou_threshold)[2]


def summarize_counts(counts):
    """由 TP/FP/FN 计算 Precision、Recall 和 F1；无真值且无预测时记为 1。"""
    tp, fp, fn = (int(value) for value in counts)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def predict_boxes(model, image, device, input_size,
                  score_threshold=DEFAULT_SCORE_THRESHOLD,
                  nms_iou=DEFAULT_NMS_IOU_THRESHOLD,
                  split_word_gaps=False):
    """在原图坐标系中推理一张图片，返回预测框、分数和缩放元数据。"""
    tensor, metadata = prepare_image(image, input_size)
    score, geometry = model(tensor[None].to(device))
    score = score[0].detach().cpu().numpy()
    geometry = geometry[0].detach().cpu().numpy()
    boxes, scores = decode_maps(score, geometry, score_threshold,
                                split_word_gaps=split_word_gaps)
    if len(boxes):
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, nms_iou)
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores, metadata


@torch.no_grad()
def evaluate_model(model, dataset, device, input_size,
                   threshold=DEFAULT_SCORE_THRESHOLD,
                   nms_iou=DEFAULT_NMS_IOU_THRESHOLD,
                   match_iou=DEFAULT_MATCH_IOU):
    """在内存模型上评估目录数据集，返回总体和按扩展名分组的指标。

    训练期验证直接复用该函数，避免为了 F1 再加载一次检查点。模型会切到
    eval 模式，调用方在训练循环中按需恢复 train 模式。推理使用固定输入尺寸，
    与部署默认行为一致。
    """
    if not 0 < threshold < 1 or not 0 < nms_iou <= 1 or not 0 < match_iou <= 1:
        raise ValueError("Invalid detection metric thresholds")
    model.eval()
    totals = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for image_path, label_path in zip(dataset.image_paths, dataset.label_paths):
        image = load_rgb_image(image_path)
        boxes, ignored = read_boxes(label_path, image.size, dataset.label_format)
        decoded, scores, _ = predict_boxes(
            model, image, device, input_size, threshold, nms_iou)
        predictions = [{"box": decoded[index].tolist(), "score": float(scores[index])}
                       for index in range(len(decoded))]

        group = format_group(image_path.name)
        counts = match_counts(predictions, boxes, ignored, match_iou)
        totals[group] += counts
        totals["overall"] += counts
    return {name: summarize_counts(counts) for name, counts in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-dir", default="detection_larger/data/val_images", help="验证图片目录")
    parser.add_argument("--label-dir", default="detection_larger/data/val_labels", help="验证标签目录")
    parser.add_argument("--label-format", choices=("bbox4", "quad8"), default="bbox4")
    parser.add_argument("--model", required=True, help="文档模型 checkpoint 或 ONNX")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD, help="文字置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU_THRESHOLD, help="重复框抑制阈值")
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU, help="预测框与真值的匹配阈值")
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
        totals[format_group(image_path.name)] += match_counts(
            predictions, boxes, ignored, args.match_iou)

    report = {"formats": {name: summarize_counts(counts) for name, counts in totals.items()},
              "overall": summarize_counts(sum(totals.values()))}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
