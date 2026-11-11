"""统一检测包内推理共享的后处理与输出辅助：解码、NMS、框扩边和裁片命名。

GPU 推理入口和评估/输出辅助共用本模块，保证包内后端的输出粒度、坐标还原、
重复框抑制和裁片命名一致；根目录单文件 ONNX 入口内联同一套逻辑以便独立部署。
本模块不依赖任何推理后端。
"""
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

from detection.utils.postprocess import (
    MIN_AREA,
    boxes_from_connected_components,
    scaled_small_region_area,
)
from detection.components.transforms import STRIDE


DEFAULT_SCORE_THRESHOLD = 0.7  # 默认文字置信度阈值，取值范围 (0, 1)。
DEFAULT_NMS_IOU_THRESHOLD = 0.3  # 默认重复框抑制交并比，取值范围 (0, 1]。
CROP_INDEX_DIGITS = 5  # 裁片文件名序号位数，序号从 1 开始并左侧补零。


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
