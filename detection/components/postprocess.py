"""统一检测的连通域解码：小区域全像素中位数，大区域边缘统计。"""
import cv2
import numpy as np

from detection.components.transforms import STRIDE


MIN_AREA = 1  # 连通域最小面积，单位为置信度图像素数；保留缩小后仅占一个高置信度像素的小字。


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


def boxes_from_connected_components(score_map_np, geo_map_np, score_threshold,
                                     min_area, small_region_area):
    """单轮四连通域检测，返回输入图像素坐标的 xyxy 框。

    大连通域用边缘像素中位数采样，小连通域用全像素中位数；不按连通域
    长宽比过滤，避免正文长行和缩放后的编号、日期被误删。
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

        box = [x_min_final * stride, y_min_final * stride,
               x_max_final * stride, y_max_final * stride]
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
