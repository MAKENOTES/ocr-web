"""小图检测的推理后处理：四连通域解码、NMS、框扩边和裁片命名。

训练期指标、报告和 ONNX/OpenVINO 部署共用本模块；部署 runtime 由
tools/sync_detection_small_runtime.py 生成本模块的副本，保证仓库与部署代码一致。
本模块不依赖任何推理后端。
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .preprocessing import STRIDE


MIN_AREA = 2  # 连通域最小面积，单位为置信度图像素数；小于该值的噪点不出框。
SMALL_REGION_BASE_AREA = 15  # 1024×1024 输入对应的小区域面积分界，单位为特征图像素数。
DEFAULT_SMALL_MAX_ASPECT = 5.0  # 小区域长边/短边比例上限，仅用于过滤细线噪点。
DEFAULT_SCORE_THRESHOLD = 0.7  # 默认文字置信度阈值，取值范围 (0, 1)。
DEFAULT_NMS_IOU_THRESHOLD = 0.3  # 默认重复框抑制交并比，取值范围 (0, 1]。
FEATURE_GEOMETRY_CLIP = 12.0  # expm1 前的上界，防止异常预测产生溢出距离。
CROP_INDEX_DIGITS = 6  # 裁片文件名序号位数，序号从 1 开始并左侧补零。


def normalized_size(input_size):
    """把 int 或 (高, 宽) 统一为二元整数元组。"""
    if isinstance(input_size, int):
        return input_size, input_size
    height, width = input_size
    return int(height), int(width)


def scaled_small_region_area(input_size):
    """按输入总像素面积缩放小区域阈值，1024×1024 时基准为 15。"""
    height, width = normalized_size(input_size)
    scaled_area = round(SMALL_REGION_BASE_AREA * (height * width) / (1024 * 1024))
    return max(MIN_AREA + 1, scaled_area)


def _box_from_small_region(ys, xs, geo_map_np, stride):
    """小连通域出框：取全像素几何中位数，避免边缘采样受噪点影响。"""
    d_top = np.median(geo_map_np[0, ys, xs])
    d_right = np.median(geo_map_np[1, ys, xs])
    d_bottom = np.median(geo_map_np[2, ys, xs])
    d_left = np.median(geo_map_np[3, ys, xs])

    cy = np.median(ys)
    cx = np.median(xs)
    return [
        (cx + 0.5 - d_left) * stride,
        (cy + 0.5 - d_top) * stride,
        (cx + 0.5 + d_right) * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]


def _box_from_large_region(ys, xs, geo_map_np, stride):
    """大连通域出框：在矩形四边取窄带像素的几何中位数。"""
    x_min_region, x_max_region = xs.min(), xs.max()
    y_min_region, y_max_region = ys.min(), ys.max()
    region_width = x_max_region - x_min_region + 1
    region_height = y_max_region - y_min_region + 1

    edge_w = max(1, min(5, int(region_width * 0.1)))
    edge_h = max(1, min(3, int(region_height * 0.1)))

    left = (xs >= x_min_region) & (xs < x_min_region + edge_w)
    x_min_final = (np.median(xs[left] + 0.5 - geo_map_np[3, ys[left], xs[left]])
                   if left.sum() > 0 else float(x_min_region))

    right = (xs > x_max_region - edge_w) & (xs <= x_max_region)
    x_max_final = (np.median(xs[right] + 0.5 + geo_map_np[1, ys[right], xs[right]])
                   if right.sum() > 0 else float(x_max_region + 1))

    top = (ys >= y_min_region) & (ys < y_min_region + edge_h)
    y_min_final = (np.median(ys[top] + 0.5 - geo_map_np[0, ys[top], xs[top]])
                   if top.sum() > 0 else float(y_min_region))

    bottom = (ys > y_max_region - edge_h) & (ys <= y_max_region)
    y_max_final = (np.median(ys[bottom] + 0.5 + geo_map_np[2, ys[bottom], xs[bottom]])
                   if bottom.sum() > 0 else float(y_max_region + 1))

    return [x_min_final * stride, y_min_final * stride,
            x_max_final * stride, y_max_final * stride]


def decode_predictions(score_map_np, geo_map_np,
                       score_threshold=DEFAULT_SCORE_THRESHOLD,
                       min_area=MIN_AREA,
                       small_region_area=None,
                       small_max_aspect=DEFAULT_SMALL_MAX_ASPECT):
    """由 score/geo 图解码输入画布坐标框，返回按置信度降序排列的框和分数。

    geo_map_np 必须已经 `expm1` 还原为特征图像素距离。返回 `(N,4)` 的
    `[x_min,y_min,x_max,y_max]` 框和 `(N,)` 置信度；没有预测时返回空数组。
    """
    score_map_np = np.asarray(score_map_np, dtype=np.float32)
    geo_map_np = np.asarray(geo_map_np, dtype=np.float32)
    if small_region_area is None:
        small_region_area = scaled_small_region_area(
            tuple(value * STRIDE for value in score_map_np.shape))

    boxes, scores = [], []
    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == label_id)
        if area < small_region_area:
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            if max(width, height) / (min(width, height) + 1e-6) > small_max_aspect:
                continue
            box = _box_from_small_region(ys, xs, geo_map_np, STRIDE)
            if not (box[2] - box[0] > 1 and box[3] - box[1] > 1):
                continue
        else:
            box = _box_from_large_region(ys, xs, geo_map_np, STRIDE)
            if not (box[2] - box[0] > 2 and box[3] - box[1] > 2):
                continue

        # 置信度取框内 score map 最大值；排序和一对一匹配都依赖它。
        x0, y0 = np.maximum(np.floor(np.asarray(box[:2]) / STRIDE), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(np.asarray(box[2:]) / STRIDE),
                            score_map_np.shape[::-1]).astype(int)
        region = score_map_np[y0:y1, x0:x1]
        boxes.append(box)
        scores.append(float(region.max()) if region.size else 0.0)

    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    return boxes[order], scores[order]


def decode_maps(score, geometry, threshold=DEFAULT_SCORE_THRESHOLD):
    """由模型输出解码输入画布坐标框，自动处理 squeeze、expm1 和非有限值。

    score 允许 `(1,H,W)` 或 `(H,W)`，geometry 为 `(4,H,W)` 的 log1p 距离。
    返回 `(N,4)` 框和 `(N,)` 置信度，不执行 NMS。
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    score = np.asarray(score, dtype=np.float32).squeeze(0)
    geometry = np.asarray(geometry, dtype=np.float32)
    finite = np.isfinite(score) & np.isfinite(geometry).all(axis=0)
    score = np.where(finite, score, 0)
    geometry = np.where(finite, geometry, 0)
    return decode_predictions(
        score, np.expm1(np.clip(geometry, 0, FEATURE_GEOMETRY_CLIP)),
        score_threshold=threshold)


def suppress_boxes(boxes, scores, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD):
    """按置信度抑制重叠的 xyxy 框，返回保留框索引；空输入返回空索引。"""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    return nms(torch.as_tensor(boxes, dtype=torch.float32),
               torch.as_tensor(scores, dtype=torch.float32), iou_threshold).numpy()


def crop_path(crop_dir, image_stem, index):
    """按 <原图 stem>-<序号>.png 生成平铺裁片路径；序号从 1 开始。"""
    return Path(crop_dir) / f"{image_stem}-{index + 1:0{CROP_INDEX_DIGITS}d}.png"


def remove_stale_crops(crop_dir, image_stem, keep_count):
    """删除该原图序号超出本次框数的旧裁片，保留其他原图的结果和无关文件。"""
    prefix = f"{image_stem}-"
    for path in Path(crop_dir).iterdir():
        if not path.is_file() or not path.name.startswith(prefix) or path.suffix.lower() != ".png":
            continue
        number = path.stem[len(prefix):]
        if number.isascii() and number.isdigit() and int(number) > keep_count:
            path.unlink()


def expand_predictions(predictions, image_size, ratio_w, ratio_h):
    """按框高向左右和上下扩边，裁片、标签和预览一致；坐标限制在原图内。"""
    if not all(np.isfinite(ratio) and ratio >= 0 for ratio in (ratio_w, ratio_h)):
        raise ValueError("Expansion ratios must be finite and nonnegative")
    width, height = image_size
    result = []
    for prediction in predictions:
        x0, y0, x1, y1 = prediction["box"]
        box_height = y1 - y0
        result.append({**prediction, "box": [
            max(0, x0 - box_height * ratio_w), max(0, y0 - box_height * ratio_h),
            min(width, x1 + box_height * ratio_w), min(height, y1 + box_height * ratio_h)]})
    return result
