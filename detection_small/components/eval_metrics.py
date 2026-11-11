"""小图检测的独立评估：解码后按置信度一对一匹配，统计 Precision/Recall/F1。

解码、NMS 和输入尺寸口径统一放在 components/postprocess.py，训练期指标、
报告和 ONNX/OpenVINO 部署共用同一套实现。坐标在还原到原图后与标签比较，
因此不同输入尺寸和补边方式可以直接对比。
"""
import numpy as np
import torch

from .dataset import read_quads
from .image import load_rgb_image
from .postprocess import (
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    decode_maps,
    decode_predictions,
    scaled_small_region_area,
    suppress_boxes,
)
from .preprocessing import prepare_image, restore_boxes


DEFAULT_MATCH_IOU = 0.5  # 预测框与真值框匹配的 IoU 下限。


def compute_iou_matrix(boxes_a, boxes_b):
    """计算两组 xyxy 框的 IoU 矩阵，返回形状 (len(a), len(b))。"""
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_area = (np.maximum(0, inter_x2 - inter_x1)
                  * np.maximum(0, inter_y2 - inter_y1))
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter_area
    return np.where(union > 0, inter_area / union, 0.0).astype(np.float32)


def box_overlap(box, boxes):
    """返回 (IoU, 交集占预测框面积比)；boxes 为空时返回空数组。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    intersection = np.maximum(
        0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    other_area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return (intersection / np.maximum(area + other_area - intersection, 1e-8),
            intersection / max(area, 1e-8))


def match_box_status(predictions, boxes, ignored, iou_threshold=DEFAULT_MATCH_IOU):
    """按置信度一对一匹配，返回逐框状态、命中真值索引和 TP/FP/FN。

    状态与输入 predictions 同序：1 表示 TP，0 表示 FP，-1 表示预测主要落在
    `###` 忽略区内、既不计 TP 也不计 FP。命中真值索引对应 boxes 去掉忽略区域
    后的顺序，供报告标出漏检框。
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


def quads_to_aabb(quads):
    """把 `(N,4,2)` 四边形转换为 `(N,4)` 水平外接框。"""
    quads = np.asarray(quads, dtype=np.float32)
    if quads.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.stack([quads[:, :, 0].min(axis=1), quads[:, :, 1].min(axis=1),
                     quads[:, :, 0].max(axis=1), quads[:, :, 1].max(axis=1)], axis=1)


def predict_boxes(model, image, device, input_size,
                  score_threshold=DEFAULT_SCORE_THRESHOLD,
                  nms_iou=DEFAULT_NMS_IOU_THRESHOLD):
    """在原图坐标系中推理一张图片，返回预测框、分数和缩放元数据。

    坐标先减补边偏移、除以缩放比例，裁到原图边界后再按置信度做 NMS。
    """
    tensor, metadata = prepare_image(image, input_size)
    score, geometry = model(tensor[None].to(device))
    score = score[0].detach().cpu().numpy()
    geometry = geometry[0].detach().cpu().numpy()
    boxes, scores = decode_maps(score, geometry, score_threshold)
    if len(boxes):
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, nms_iou)
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores, metadata


def evaluate_single_image(score_map_np, geo_map_np, gt_boxes, gt_ignored=None,
                          score_threshold=DEFAULT_SCORE_THRESHOLD,
                          iou_threshold=DEFAULT_MATCH_IOU,
                          nms_iou=DEFAULT_NMS_IOU_THRESHOLD,
                          small_region_area=None):
    """对单张画布空间的结果计算框级指标，供轻量测试和自定义流程使用。"""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    if gt_ignored is None:
        gt_ignored = np.zeros(len(gt_boxes), dtype=bool)
    boxes, scores = decode_predictions(
        score_map_np, geo_map_np, score_threshold=score_threshold,
        small_region_area=small_region_area)
    if len(boxes):
        keep = suppress_boxes(boxes, scores, nms_iou)
        boxes, scores = boxes[keep], scores[keep]
    predictions = [{"box": boxes[index].tolist(), "score": float(scores[index])}
                   for index in range(len(boxes))]
    status, used, counts = match_box_status(predictions, gt_boxes, gt_ignored, iou_threshold)
    metrics = summarize_counts(counts)
    tp_indices = np.where(status == 1)[0]
    if len(tp_indices):
        overlap = compute_iou_matrix(
            boxes[tp_indices], gt_boxes[~np.asarray(gt_ignored, dtype=bool)])
        metrics["avg_iou"] = float(overlap.max(axis=1).mean())
    else:
        metrics["avg_iou"] = 0.0
    metrics.update({"predictions": predictions, "status": status, "used": used})
    return metrics


@torch.no_grad()
def evaluate_detection(model, dataset, device, input_size,
                       score_threshold=DEFAULT_SCORE_THRESHOLD,
                       match_iou=DEFAULT_MATCH_IOU,
                       nms_iou=DEFAULT_NMS_IOU_THRESHOLD,
                       max_samples=None):
    """在验证目录上评估检测指标，返回总体 P/R/F1 和逐图统计。

    坐标在还原后与原图标签比较，因此不同输入尺寸和补边方式可以直接对比。
    模型会切到 eval 模式；调用方在训练循环中按需恢复 train 模式。
    """
    if not 0 < score_threshold < 1:
        raise ValueError("score_threshold must be in (0, 1)")
    if not 0 < match_iou <= 1:
        raise ValueError("match_iou must be in (0, 1]")
    if not 0 < nms_iou <= 1:
        raise ValueError("nms_iou must be in (0, 1]")
    model.eval()

    total_counts = np.zeros(3, dtype=np.int64)
    matched_ious = []
    per_image = []
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for index in range(n_samples):
        image_path, label_path = dataset.image_paths[index], dataset.label_paths[index]
        image = load_rgb_image(image_path)
        if label_path is None:
            quads = np.empty((0, 4, 2), dtype=np.float32)
            ignored = np.zeros(0, dtype=bool)
        else:
            quads, ignored = read_quads(label_path, image.size, dataset.label_format)
        boxes = quads_to_aabb(quads)

        predictions_boxes, scores, _ = predict_boxes(
            model, image, device, input_size, score_threshold, nms_iou)
        predictions = [{"box": predictions_boxes[i].tolist(), "score": float(scores[i])}
                       for i in range(len(predictions_boxes))]
        status, _, counts = match_box_status(predictions, boxes, ignored, match_iou)
        total_counts += counts

        truth = boxes[~ignored]
        tp_indices = np.where(status == 1)[0]
        if len(tp_indices):
            overlap = compute_iou_matrix(predictions_boxes[tp_indices], truth)
            matched_ious.extend(overlap.max(axis=1).tolist())

        per_image.append({
            "image_path": str(image_path),
            "width": image.size[0],
            "height": image.size[1],
            "tp": int(counts[0]),
            "fp": int(counts[1]),
            "fn": int(counts[2]),
            "n_pred": len(predictions_boxes),
            "n_gt": int(len(truth)),
        })

    metrics = summarize_counts(total_counts)
    metrics["avg_iou"] = float(np.mean(matched_ious)) if matched_ious else 0.0
    metrics.update({
        "total_tp": int(total_counts[0]),
        "total_fp": int(total_counts[1]),
        "total_fn": int(total_counts[2]),
        "n_images": n_samples,
        "per_image": per_image,
    })
    return metrics
