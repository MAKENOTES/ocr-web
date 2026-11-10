"""直接读取验证目录，报告总体及按图片格式分组的文字框指标。"""
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

from detection.components.dataset import OCRDataset, read_boxes
from detection.components.image import load_rgb_image
from detection.components.transforms import prepare_image, restore_boxes
from detection.inference import (
    NMS_IOU_THRESHOLD,
    SCORE_THRESHOLD,
    DocumentDetector,
    decode_maps,
    suppress_boxes,
)

SOURCE_PREFIXES = {"ui", "doc"}  # 统一数据按文件名前缀分组；其他名称归入 other。



def box_overlap(box, boxes):
    """返回交集并集比与交集占预测框面积比；boxes 为空时返回空数组。"""
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])
                              - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    other_area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return (intersection / np.maximum(area + other_area - intersection, 1e-8),
            intersection / max(area, 1e-8))


def match_box_status(predictions, boxes, ignored, iou_threshold=0.5):
    """按置信度一对一匹配，返回逐框状态、命中真值索引和 TP/FP/FN。

    状态与输入 predictions 同序：1 表示 TP，0 表示 FP，-1 表示预测主要
    落在 ### 忽略区内、既不计 TP 也不计 FP。命中真值索引对应 boxes 去掉
    忽略区域后的顺序，供可视化标出漏检框。
    """
    truth, ignore_boxes = boxes[~ignored], boxes[ignored]
    used = set()
    status = np.zeros(len(predictions), dtype=np.int8)
    # 稳定排序保证同分预测按输入顺序匹配，与逐框循环的判定顺序一致。
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


def match_counts(predictions, boxes, ignored, iou_threshold=0.5):
    """按置信度一对一匹配；预测主要落在 ### 内时不计误检。"""
    return match_box_status(predictions, boxes, ignored, iou_threshold)[2]


def summarize_counts(counts):
    """由 TP/FP/FN 计算 Precision、Recall 和 F1。"""
    tp, fp, fn = map(int, counts)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


@torch.no_grad()
def evaluate_model(model, dataset, device, input_size, threshold=SCORE_THRESHOLD,
                   nms_iou=NMS_IOU_THRESHOLD, match_iou=0.5):
    """在内存模型上评估目录数据集，返回总体和按来源前缀分组的指标。

    训练期验证直接复用该函数，避免为了 F1 再加载一次检查点。模型会切到
    eval 模式，调用方在训练循环中按需恢复 train 模式。input_size 是动态
    推理的最大边长，与部署入口按宽高比选尺寸的行为一致。
    """
    if not 0 < threshold < 1 or not 0 < nms_iou <= 1 or not 0 < match_iou <= 1:
        raise ValueError("Invalid detection metric thresholds")
    model.eval()
    totals = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for image_path, label_path in zip(dataset.image_paths, dataset.label_paths):
        image = load_rgb_image(image_path)
        boxes, ignored = read_boxes(label_path, image.size)
        tensor, metadata = prepare_image(image, input_size, dynamic=True)
        score, geometry = model(tensor[None].to(device))
        score, geometry = score[0].cpu().numpy(), geometry[0].cpu().numpy()

        decoded, scores = decode_maps(score, geometry, threshold)
        decoded = restore_boxes(decoded, metadata, image.size)
        valid = (decoded[:, 2] > decoded[:, 0]) & (decoded[:, 3] > decoded[:, 1])
        decoded, scores = decoded[valid], scores[valid]
        keep = suppress_boxes(decoded, scores, nms_iou)
        predictions = [{"box": decoded[index].tolist(), "score": float(scores[index])}
                       for index in keep]

        prefix = image_path.name.split("_", 1)[0]
        group = prefix if prefix in SOURCE_PREFIXES else "other"
        counts = match_counts(predictions, boxes, ignored, match_iou)
        totals[group] += counts
        totals["overall"] += counts
    return {name: summarize_counts(counts) for name, counts in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-dir", default="detection/data/val_images", help="验证图片目录")
    parser.add_argument("--label-dir", default="detection/data/val_labels", help="验证标签目录")
    parser.add_argument("--label-format", choices=("bbox4", "quad8"), default="bbox4")
    parser.add_argument("--model", required=True, help="统一检测 checkpoint 或 ONNX")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD, help="文字置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESHOLD, help="重复框抑制阈值")
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
