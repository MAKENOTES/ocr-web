"""小图检测的独立评估：连通域解码、匈牙利匹配、Precision/Recall/F1。

批量入口为 evaluate_detection，单图入口为 evaluate_single_image。
本模块使用八连通域和固定小区域阈值，与根目录矩形推理不完全一致。"""

import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment


#  1. 本模块独立解码预测框

# 默认参数 (可在调用时覆盖)
DEFAULT_SCORE_THRESHOLD = 0.7  # 置信度严格超过 0.7 的网格参与连通域。
DEFAULT_MIN_AREA = 2  # 最小连通域面积，单位为特征图像素数。
DEFAULT_SMALL_REGION_AREA = 15  # 小区域面积分界，单位为特征图像素数；当前不随输入尺寸调整。
DEFAULT_SMALL_MAX_ASPECT = 5.0  # 仅小区域使用的长边/短边比例上限，无单位。
DEFAULT_STRIDE = 4.0  # 特征图距离还原为输入图片像素的倍率。


def _box_from_small_region(ys, xs, geo_map_np, stride):
    """小连通域出框：全像素中位数"""
    d_top    = np.median(geo_map_np[0, ys, xs])
    d_right  = np.median(geo_map_np[1, ys, xs])
    d_bottom = np.median(geo_map_np[2, ys, xs])
    d_left   = np.median(geo_map_np[3, ys, xs])

    cy = np.median(ys)
    cx = np.median(xs)

    return [
        (cx + 0.5 - d_left)   * stride,
        (cy + 0.5 - d_top)    * stride,
        (cx + 0.5 + d_right)  * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]


def decode_predictions(score_map_np, geo_map_np,
                       score_threshold=DEFAULT_SCORE_THRESHOLD,
                       min_area=DEFAULT_MIN_AREA,
                       small_region_area=DEFAULT_SMALL_REGION_AREA,
                       small_max_aspect=DEFAULT_SMALL_MAX_ASPECT):
    """
    从 score_map 和 geo_map (numpy) 解码出预测框列表。
    geo_map_np 应已 expm1 还原为特征图像素距离，输出框再乘步长。

    返回: np.ndarray, shape (N, 4), 每行 [x_min, y_min, x_max, y_max]
          坐标在 input_size 空间 (例如 640x640)
    """
    stride = DEFAULT_STRIDE
    boxes = []

    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == label_id)

        # 小连通域
        if area < small_region_area:
            w_cc = stats[label_id, cv2.CC_STAT_WIDTH]
            h_cc = stats[label_id, cv2.CC_STAT_HEIGHT]
            if max(w_cc, h_cc) / (min(w_cc, h_cc) + 1e-6) > small_max_aspect:
                continue
            box = _box_from_small_region(ys, xs, geo_map_np, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        # 大连通域：边缘采样
        x_min_r = xs.min(); x_max_r = xs.max()
        y_min_r = ys.min(); y_max_r = ys.max()
        rw = x_max_r - x_min_r + 1
        rh = y_max_r - y_min_r + 1

        edge_w = max(1, min(5, int(rw * 0.1)))
        edge_h = max(1, min(3, int(rh * 0.1)))

        lm = (xs >= x_min_r) & (xs < x_min_r + edge_w)
        x_min_final = np.median(xs[lm] + 0.5 - geo_map_np[3, ys[lm], xs[lm]]) if lm.sum() > 0 else float(x_min_r)

        rm = (xs > x_max_r - edge_w) & (xs <= x_max_r)
        x_max_final = np.median(xs[rm] + 0.5 + geo_map_np[1, ys[rm], xs[rm]]) if rm.sum() > 0 else float(x_max_r + 1)

        tm = (ys >= y_min_r) & (ys < y_min_r + edge_h)
        y_min_final = np.median(ys[tm] + 0.5 - geo_map_np[0, ys[tm], xs[tm]]) if tm.sum() > 0 else float(y_min_r)

        bm = (ys > y_max_r - edge_h) & (ys <= y_max_r)
        y_max_final = np.median(ys[bm] + 0.5 + geo_map_np[2, ys[bm], xs[bm]]) if bm.sum() > 0 else float(y_max_r + 1)

        box = [x_min_final * stride, y_min_final * stride,
               x_max_final * stride, y_max_final * stride]
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    return np.array(boxes) if boxes else np.array([]).reshape(0, 4)


#  2. IoU 计算 + 匹配

def compute_iou_matrix(boxes_a, boxes_b):
    """
    计算两组 AABB 框之间的 IoU 矩阵。
    boxes_a: (M, 4), boxes_b: (N, 4), 格式 [x1, y1, x2, y2]
    返回: (M, N) IoU 矩阵
    """
    M = len(boxes_a)
    N = len(boxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    a = boxes_a[:, np.newaxis, :]  # (M, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, N, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union_area = area_a + area_b - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    用匈牙利算法匹配预测框和真值框。

    返回:
      tp: int  — 匹配成功的数量
      fp: int  — 预测框多余的数量 (无匹配的 pred)
      fn: int  — 真值框漏检的数量 (无匹配的 gt)
      matched_ious: list[float] — 每对匹配的 IoU 值
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0, []
    if n_pred == 0:
        return 0, 0, n_gt, []
    if n_gt == 0:
        return 0, n_pred, 0, []

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # 匈牙利算法 (最大化 IoU → 最小化 cost)
    cost_matrix = 1.0 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    tp = 0
    matched_ious = []
    matched_pred = set()
    matched_gt = set()

    for pi, gi in zip(pred_indices, gt_indices):
        if iou_matrix[pi, gi] >= iou_threshold:
            tp += 1
            matched_ious.append(float(iou_matrix[pi, gi]))
            matched_pred.add(pi)
            matched_gt.add(gi)

    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)

    return tp, fp, fn, matched_ious


#  3. 从 GT quads 提取 AABB 框 (用于与预测框对比)

def quads_to_aabb(quads, scale, pad_x, pad_y):
    """
    把原始 quads (经过 ResizeAndPad 变换后) 转换为 AABB 格式。
    quads: np.ndarray, shape (N, 4, 2) — 在 input_size 空间的坐标
    返回: np.ndarray, shape (N, 4) — [x_min, y_min, x_max, y_max]
    """
    if quads.size == 0:
        return np.array([]).reshape(0, 4)

    if len(quads.shape) != 3:
        return np.array([]).reshape(0, 4)

    x_min = quads[:, :, 0].min(axis=1)
    y_min = quads[:, :, 1].min(axis=1)
    x_max = quads[:, :, 0].max(axis=1)
    y_max = quads[:, :, 1].max(axis=1)

    return np.stack([x_min, y_min, x_max, y_max], axis=1)


#  4. 单张图评估

def evaluate_single_image(score_map_np, geo_map_np, gt_boxes,
                          score_threshold=DEFAULT_SCORE_THRESHOLD,
                          iou_threshold=0.5):
    """
    对单张图计算检测指标。

    参数:
      score_map_np: (H, W) — 模型输出的 score map (sigmoid 后)
      geo_map_np:   (4, H, W) — 模型输出的 geo map (已 expm1 还原)
      gt_boxes:     (N, 4) — 真值框 [x_min, y_min, x_max, y_max] 在 input_size 空间
      score_threshold: float
      iou_threshold: float — IoU 匹配阈值

    返回: dict with keys: tp, fp, fn, precision, recall, f1, avg_iou
    """
    pred_boxes = decode_predictions(score_map_np, geo_map_np,
                                    score_threshold=score_threshold)

    tp, fp, fn, matched_ious = match_boxes(pred_boxes, gt_boxes,
                                           iou_threshold=iou_threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou   = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'avg_iou': avg_iou,
        'n_pred': len(pred_boxes), 'n_gt': len(gt_boxes),
    }


#  5. 整个验证集评估 (集成到训练循环)

def evaluate_detection(model, val_dataset, device,
                       input_size=640,
                       score_threshold=DEFAULT_SCORE_THRESHOLD,
                       iou_threshold=0.5,
                       max_samples=None):
    """
    在整个验证集上评估检测指标。

    与 validate() 不同，这里不走 DataLoader 的 collate，
    而是逐张图处理，因为我们需要原始 quads 来构建 GT 框。

    参数:
      model: EASTModel
      val_dataset: OCRDataset (使用 val transform，但我们需要中间结果)
      device: torch.device
      input_size: int
      score_threshold: float
      iou_threshold: float
      max_samples: int or None — 限制评估样本数 (加速调试)

    返回: dict — 全局 Precision / Recall / F1 + 逐图指标列表
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    per_image_results = []

    n_samples = len(val_dataset)
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    # 我们需要一个不带 GenerateGT 和 ToTensor 的 transform
    # 来获取 ResizeAndPad 后的 quads。
    # 方案：直接用 val_dataset 的原始数据，手动走 transform 拿中间结果。
    from components.transforms import ResizeAndPad
    from torchvision.transforms import Normalize
    import torchvision.transforms.functional as TF

    resize_pad = ResizeAndPad(input_size)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for idx in range(n_samples):
            # 获取原始样本 (不经过 dataset 的 transform)
            image_path = val_dataset.image_paths[idx]
            label_path = val_dataset.label_paths[idx]

            from PIL import Image as PILImage
            image = PILImage.open(image_path).convert('RGB')

            # 解析 quads
            quads = _parse_label_file(label_path)

            # 应用 ResizeAndPad
            sample = {'image': image, 'quads': quads, 'transcriptions': []}
            sample = resize_pad(sample)
            padded_quads = sample['quads']  # 在 input_size 空间的 quads

            # 准备输入张量
            img_tensor = TF.to_tensor(sample['image'])
            img_tensor = normalize(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # 模型推理
            score_map, geo_map = model(img_tensor)

            score_map_np = score_map[0, 0].cpu().numpy()
            geo_map_np = geo_map[0, :4].cpu().numpy()

            # 从 log 空间还原
            geo_map_np = np.expm1(geo_map_np)
            geo_map_np = np.maximum(geo_map_np, 0)

            # GT 框
            gt_boxes = quads_to_aabb(padded_quads, scale=1.0, pad_x=0, pad_y=0)

            # 单图评估
            result = evaluate_single_image(
                score_map_np, geo_map_np, gt_boxes,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
            )
            result['image_path'] = image_path

            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']
            if result['avg_iou'] > 0:
                all_ious.extend([result['avg_iou']] * result['tp'])
            per_image_results.append(result)

    # 全局指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou   = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'n_images': n_samples,
        'per_image': per_image_results,
    }


def _parse_label_file(label_path):
    """历史 bbox4 解析：只读空白分隔的前四项并取整；缺文件或坏行返回空/跳过。
    与训练加载器的浮点、逗号和 quad8 支持不同，不适合作为严格数据校验器。"""
    import os
    quads = []

    if not os.path.exists(label_path):
        return np.array([], dtype=np.float32)

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\ufeff').strip()
            if not line:
                continue

            parts = line.split()
            try:
                if len(parts) < 4:
                    continue
                points = [int(float(p)) for p in parts[:4]]
                x_min, y_min, x_max, y_max = points

                quad = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.float32)
                quads.append(quad)
            except (ValueError, IndexError):
                continue

    return np.array(quads, dtype=np.float32) if quads else np.array([], dtype=np.float32)


#  6. 多阈值评估 (可选，用于分析最佳阈值)

def evaluate_multi_threshold(model, val_dataset, device,
                             input_size=640,
                             score_thresholds=None,
                             iou_threshold=0.5,
                             max_samples=None):
    """
    在多个 score_threshold 下评估，帮助选择最佳阈值。

    返回: list[dict] — 每个 threshold 对应一组指标
    """
    if score_thresholds is None:
        score_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for thr in score_thresholds:
        r = evaluate_detection(
            model, val_dataset, device,
            input_size=input_size,
            score_threshold=thr,
            iou_threshold=iou_threshold,
            max_samples=max_samples,
        )
        r['score_threshold'] = thr
        results.append(r)
        print(f"  Threshold={thr:.1f}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} AvgIoU={r['avg_iou']:.4f} "
              f"(TP={r['total_tp']} FP={r['total_fp']} FN={r['total_fn']})")

    return results