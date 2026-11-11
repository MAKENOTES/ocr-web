"""统一检测的推理后处理：连通域解码、词/短语拆分、NMS、框扩边和裁片命名。

PyTorch 推理入口和评估/输出辅助共用本模块；ONNX/OpenVINO 部署由
tools/sync_detection_runtime.py 生成本模块的 runtime 副本，保证部署代码与仓库一致。
本模块不依赖任何推理后端。
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from detection.components.preprocessing import STRIDE


MIN_AREA = 1  # 连通域最小面积，单位为置信度图像素数；保留缩小后仅占一个高置信度像素的小字。
WORD_SPLIT_MIN_HEIGHT = 8  # 低于该高度的连通域不拆分，避免把单字符和符号切碎，单位为特征图像素。
WORD_SPLIT_LOW_COLUMN_RATIO = 0.08  # 列占用低于连通域高度的该比例时视为潜在词间空隙。
WORD_SPLIT_MIN_GAP_PIXELS = 2  # 潜在空隙的最小宽度，单位为特征图像素；过滤字符内部的小孔。
WORD_SPLIT_GAP_RATIO = 0.18  # 潜在空隙宽度至少为连通域高度的该比例才拆分，区分词间空格与字内空隙。
DEFAULT_SCORE_THRESHOLD = 0.7  # 默认文字置信度阈值，取值范围 (0, 1)。
DEFAULT_NMS_IOU_THRESHOLD = 0.3  # 默认重复框抑制交并比，取值范围 (0, 1]。
CROP_INDEX_DIGITS = 5  # 裁片文件名序号位数，序号从 1 开始并左侧补零。


def scaled_small_region_area(input_size):
    """按输入像素面积缩放小连通域阈值，1024×1024 对应 15 个置信度图像素。"""
    if isinstance(input_size, int):
        size = (input_size, input_size)
    else:
        size = tuple(input_size)
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError(f"input_size must be positive height/width: {input_size!r}")
    height, width = map(int, size)
    return max(MIN_AREA + 1, round(15 * height * width / (1024 * 1024)))


def box_from_small_region(ys, xs, geo_map_np, stride):
    """小连通域出框：取区域内全部像素的几何中位数，适配单字符和符号。

    geo_map_np 已经 expm1 还原，单位为特征图像素；乘 stride 后为输入图片像素。
    """
    d_top, d_right, d_bottom, d_left = (np.median(geo_map_np[index, ys, xs])
                                       for index in range(4))
    cy, cx = np.median(ys), np.median(xs)
    return [
        (cx + 0.5 - d_left) * stride,
        (cy + 0.5 - d_top) * stride,
        (cx + 0.5 + d_right) * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]


def box_from_large_region(ys, xs, geo_map_np, stride):
    """大连通域出框：左右上下边缘带分别取几何中位数。"""
    x_min_r, x_max_r = xs.min(), xs.max()
    y_min_r, y_max_r = ys.min(), ys.max()
    edge_w = max(1, min(5, int((x_max_r - x_min_r + 1) * 0.1)))
    edge_h = max(1, min(3, int((y_max_r - y_min_r + 1) * 0.1)))

    left = (xs >= x_min_r) & (xs < x_min_r + edge_w)
    x_min_final = (np.median(xs[left] + 0.5 - geo_map_np[3, ys[left], xs[left]])
                   if left.sum() else float(x_min_r))
    right = (xs > x_max_r - edge_w) & (xs <= x_max_r)
    x_max_final = (np.median(xs[right] + 0.5 + geo_map_np[1, ys[right], xs[right]])
                   if right.sum() else float(x_max_r + 1))
    top = (ys >= y_min_r) & (ys < y_min_r + edge_h)
    y_min_final = (np.median(ys[top] + 0.5 - geo_map_np[0, ys[top], xs[top]])
                   if top.sum() else float(y_min_r))
    bottom = (ys > y_max_r - edge_h) & (ys <= y_max_r)
    y_max_final = (np.median(ys[bottom] + 0.5 + geo_map_np[2, ys[bottom], xs[bottom]])
                   if bottom.sum() else float(y_max_r + 1))

    return [x_min_final * stride, y_min_final * stride,
            x_max_final * stride, y_max_final * stride]


def split_component(ys, xs):
    """按低占用列把粘连的文字连通域拆成词/短语片段。

    只对高度足够的连通域生效；低占用列连续宽度必须同时满足绝对最小宽度
    和相对高度的比例，避免把字符内部的空隙拆成多个框。返回 (ys, xs) 列表。
    """
    height = int(ys.max() - ys.min() + 1) if len(ys) else 0
    if height < WORD_SPLIT_MIN_HEIGHT or len(xs) < 2:
        return [(ys, xs)]

    x_min, x_max = int(xs.min()), int(xs.max())
    width = x_max - x_min + 1
    counts = np.bincount(xs - x_min, minlength=width)
    low_limit = max(1, int(WORD_SPLIT_LOW_COLUMN_RATIO * height))
    low = counts <= low_limit

    cuts = []
    start = None
    for index in range(width + 1):
        is_low = index < width and bool(low[index])
        if is_low and start is None:
            start = index
        elif not is_low and start is not None:
            gap_start, gap_end = start, index - 1
            gap_width = gap_end - gap_start + 1
            # 贴边的低列属于框边界，不作为词间空隙。
            if (gap_start > 0 and gap_end < width - 1
                    and gap_width >= max(WORD_SPLIT_MIN_GAP_PIXELS,
                                         int(WORD_SPLIT_GAP_RATIO * height))):
                cuts.append((gap_start + x_min, gap_end + x_min))
            start = None

    if not cuts:
        return [(ys, xs)]

    segments = []
    current = x_min
    for gap_start, gap_end in cuts:
        mask = (xs >= current) & (xs < gap_start)
        if mask.any():
            segments.append((ys[mask], xs[mask]))
        current = gap_end + 1
    mask = xs >= current
    if mask.any():
        segments.append((ys[mask], xs[mask]))
    return segments or [(ys, xs)]


def boxes_from_connected_components(score_map_np, geo_map_np, score_threshold,
                                     min_area, small_region_area, split_word_gaps=True):
    """单轮四连通域检测，返回输入图像素坐标的 xyxy 框。

    大连通域用边缘像素中位数采样，小连通域用全像素中位数；不按连通域
    长宽比过滤。默认按低占用列把粘连的整行框拆成词/短语框；设置
    split_word_gaps=False 可保留旧的整段输出。
    """
    stride = float(STRIDE)
    boxes = []
    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ys, xs = np.where(labels == label_id)

        if area < small_region_area:
            box = box_from_small_region(ys, xs, geo_map_np, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        segments = split_component(ys, xs) if split_word_gaps else [(ys, xs)]
        for segment_ys, segment_xs in segments:
            if len(segment_xs) < min_area:
                continue
            if len(segment_xs) < small_region_area:
                box = box_from_small_region(segment_ys, segment_xs, geo_map_np, stride)
                if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                    boxes.append(box)
            else:
                box = box_from_large_region(segment_ys, segment_xs, geo_map_np, stride)
                if box[2] - box[0] > 2 and box[3] - box[1] > 2:
                    boxes.append(box)

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def suppress_boxes(boxes, scores, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD):
    """按置信度抑制重叠的 xyxy 框，返回保留框索引；空输入返回空索引。"""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    return nms(torch.as_tensor(boxes, dtype=torch.float32),
               torch.as_tensor(scores, dtype=torch.float32), iou_threshold).numpy()


def decode_maps(score, geometry, threshold=DEFAULT_SCORE_THRESHOLD, split_word_gaps=True):
    """由置信度连通域解码输入图像素坐标框，按置信度降序返回。

    score 形状为 (1, H, W)，geometry 为 (4, H, W) 的 log1p 特征图距离。
    返回 (N, 4) 的 xyxy 框及 (N,) 置信度；非有限预测不参与出框。
    split_word_gaps=True 时按低占用列把整行粘连框拆成词/短语粒度。
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    score = np.asarray(score).squeeze(0)
    geometry = np.asarray(geometry)
    score = np.where(np.isfinite(score) & np.isfinite(geometry).all(axis=0), score, 0)

    boxes = boxes_from_connected_components(
        score, np.expm1(np.clip(geometry, 0, 12)), threshold, min_area=MIN_AREA,
        small_region_area=scaled_small_region_area(tuple(value * STRIDE for value in score.shape)),
        split_word_gaps=split_word_gaps,
    ).astype(np.float32).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(axis=1)]

    confidence = []
    for box in boxes:
        x0, y0 = np.maximum(np.floor(box[:2] / STRIDE), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(box[2:] / STRIDE), score.shape[::-1]).astype(int)
        region = score[y0:y1, x0:x1]
        confidence.append(float(region.max()) if region.size else 0.0)
    confidence = np.asarray(confidence, dtype=np.float32)

    order = np.argsort(-confidence, kind="stable")
    return boxes[order], confidence[order]


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
