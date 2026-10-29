"""生成收缩置信度图、log1p 四方向距离及 ### 忽略区域掩码。"""
import cv2
import numpy as np


def shrink_polygon(poly, r=0.4):
    """按边法向收缩四边形，偏移不超过最短边的 45%，防止细长框翻转。"""
    area = cv2.contourArea(poly)
    perimeter = cv2.arcLength(poly, True)
    if perimeter == 0:
        return poly

    # 按面积和周长估计偏移距离，坐标单位沿用输入多边形。
    offset = area * (1 - r * r) / perimeter
    edge_lengths = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
    min_side = min(edge_lengths)
    # 上限随最短边缩放，框小于 1px 时不做额外收缩。
    offset = min(offset, max(0.0, min_side * 0.45))

    lines = []
    for i in range(4):
        p1, p2 = poly[i], poly[(i + 1) % 4]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        normal = np.array([0.0, 0.0]) if length < 1e-6 else np.array([-edge[1], edge[0]]) / length
        lines.append((p1 + offset * normal, edge))

    new_poly = []
    for i in range(4):
        p1_prev, v1_prev = lines[(i - 1) % 4]
        p1_curr, v1_curr = lines[i]
        det = v1_prev[0] * v1_curr[1] - v1_prev[1] * v1_curr[0]
        if abs(det) < 1e-6:
            new_poly.append(poly[i])
            continue
        diff = p1_curr - p1_prev
        t = (diff[0] * v1_curr[1] - diff[1] * v1_curr[0]) / det
        new_poly.append(p1_prev + t * v1_prev)

    return np.array(new_poly, dtype=np.float32)


def generate_ground_truth(image_size, quads, score_map_scale=0.25):
    """将输入图像素四边形转换为置信度图和四通道 log1p 特征图距离。

    image_size 是模型输入的高、宽；score_map_scale=0.25 对应网络步长 4。
    几何顺序为上、右、下、左，单位为特征图像素。重叠区域保留面积更小的
    文字框，避免相邻文字的目标互相覆盖。
    """
    img_h, img_w = image_size
    score_map_h, score_map_w = int(img_h * score_map_scale), int(img_w * score_map_scale)
    score_map = np.zeros((score_map_h, score_map_w), dtype=np.float32)
    geo_map = np.zeros((4, score_map_h, score_map_w), dtype=np.float32)
    if quads.size == 0:
        return score_map, geo_map

    for quad in quads:
        scaled_quad = quad * score_map_scale
        shrunk_quad = shrink_polygon(scaled_quad, r=0.35)

        mask = np.zeros((score_map_h, score_map_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shrunk_quad.astype(np.int32)], 1)
        score_map[mask == 1] = 1

        x_min, y_min = np.min(scaled_quad, axis=0)
        x_max, y_max = np.max(scaled_quad, axis=0)
        for y, x in np.argwhere(mask == 1):
            pixel_center_x, pixel_center_y = x + 0.5, y + 0.5
            distances = np.array([
                max(0, pixel_center_y - y_min),
                max(0, x_max - pixel_center_x),
                max(0, y_max - pixel_center_y),
                max(0, pixel_center_x - x_min),
            ], dtype=np.float32)

            new_area = (distances[0] + distances[2]) * (distances[1] + distances[3])
            old_area = (geo_map[0, y, x] + geo_map[2, y, x]) * (geo_map[1, y, x] + geo_map[3, y, x])
            if geo_map[0, y, x] == 0 or new_area < old_area:
                geo_map[:, y, x] = distances

    return score_map, np.log1p(geo_map)


def make_training_mask(image_size, ignored_quads, score_map_scale=0.25):
    """忽略与 ### 相交的输出网格，不改变普通文字框的收缩和几何距离。"""
    height, width = (int(value * score_map_scale) for value in image_size)
    mask = np.ones((height, width), dtype=np.float32)
    for quad in ignored_quads:
        minimum = np.floor(quad.min(axis=0) * score_map_scale).astype(int)
        maximum = np.ceil(quad.max(axis=0) * score_map_scale).astype(int)
        mask[max(0, minimum[1]):min(height, maximum[1]),
             max(0, minimum[0]):min(width, maximum[0])] = 0
    return mask
