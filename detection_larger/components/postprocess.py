"""文档检测独立使用连通域聚类与边缘几何统计解码文字框。"""
import cv2
import numpy as np

from detection_larger.components.transforms import STRIDE

MIN_AREA = 1  # 连通域最小面积，单位为置信度图像素数；保留缩小后仅占一个高置信度像素的小字。


def scaled_small_region_area(input_size):
    """按整图像素面积缩放小连通域阈值，1024×1024 对应 15 个置信度图像素。"""
    size = (input_size, input_size) if isinstance(input_size, int) else input_size
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError(f"input_size must be positive height/width: {input_size!r}")
    height, width = map(int, size)
    return max(MIN_AREA + 1, round(15 * height * width / (1024 * 1024)))


def box_from_small_region(ys, xs, geo_map_np, stride):
    """
    针对小连通域（√ 等单字符/符号）的出框策略。
    取所有像素 geo 预测的中位数，比边缘采样更稳定。
    geo_map_np 已经 expm1 还原，单位为特征图像素；乘 stride 后为输入图片像素。
    """
    d_top    = np.median(geo_map_np[0, ys, xs])
    d_right  = np.median(geo_map_np[1, ys, xs])
    d_bottom = np.median(geo_map_np[2, ys, xs])
    d_left   = np.median(geo_map_np[3, ys, xs])

    cy = np.median(ys)
    cx = np.median(xs)

    box = [
        (cx + 0.5 - d_left)   * stride,
        (cy + 0.5 - d_top)    * stride,
        (cx + 0.5 + d_right)  * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]
    return box


def boxes_from_connected_components(score_map_np, geo_map_np, score_threshold,
                                     min_area, small_region_area):
    """
    单轮连通域检测：

      - 大连通域（面积 >= small_region_area）：边缘像素中位数采样
        ★ 不做宽高比过滤，正文长文本行天然是扁长条
      - 小连通域（面积 < small_region_area）：全像素中位数出框
        文档缩放后的编号、日期可能只有一行特征像素，不能按连通域长宽比过滤。
    """
    stride = float(STRIDE)
    boxes = []

    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=4
    )

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == label_id)

        # 小连通域（单符号）
        if area < small_region_area:
            box = box_from_small_region(ys, xs, geo_map_np, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        # 大连通域（文本行）：边缘像素采样出框
        # ★ 不加宽高比过滤，正文长文本行宽高比本来就大
        x_min_r, x_max_r = xs.min(), xs.max()
        y_min_r, y_max_r = ys.min(), ys.max()
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

    return np.array(boxes) if boxes else np.array([])
