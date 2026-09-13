"""独立生成文档检测的收缩真值、log1p 几何距离及忽略区域掩码。"""
import numpy as np

import cv2


def shrink_polygon(poly, r=0.4):
    """按边法向收缩四边形，偏移不超过最短边的 45%，防止细长框翻转。"""
    area = cv2.contourArea(poly)
    perimeter = cv2.arcLength(poly, True)

    if perimeter == 0:
        return poly

    # 按面积和周长估计偏移距离，坐标单位沿用输入多边形。
    d = area * (1 - r * r) / perimeter

    edge_lengths = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
    min_side = min(edge_lengths)

    # 上限随最短边缩放，不保证取整后必定保留一个正像素。
    max_d = min_side * 0.45

    # 如果 max_d <= 0，说明框太小了，甚至不足1px，此时强行保留极小厚度或不收缩
    if max_d < 0:
        max_d = 0  # 不收缩，或者极小收缩

    d = min(d, max_d)

    pco = []
    lines = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-6:
            normal = np.array([0, 0])
        else:
            # 顺时针顺序，向内法向量
            normal = np.array([-edge[1], edge[0]]) / length

        # 偏移后的直线点和方向
        start_pt = p1 + d * normal
        lines.append((start_pt, edge))  # 存方向向量即可，不需要归一化

    new_poly = []
    for i in range(4):
        # 相邻偏移边的交点对应原顶点，保持顶点顺序。

        # Line 1: 上一条边 (i-1)
        p1_prev, v1_prev = lines[(i - 1) % 4]
        # Line 2: 当前边 (i)
        p1_curr, v1_curr = lines[i]

        # 计算交点
        det = v1_prev[0] * v1_curr[1] - v1_prev[1] * v1_curr[0]

        if abs(det) < 1e-6:
            new_poly.append(poly[i])
            continue

        diff = p1_curr - p1_prev
        t = (diff[0] * v1_curr[1] - diff[1] * v1_curr[0]) / det
        intersection = p1_prev + t * v1_prev
        new_poly.append(intersection)

    return np.array(new_poly, dtype=np.float32)


def generate_ground_truth(image_size, quads, score_map_scale=0.25):
    """将像素四边形转换为收缩置信度图和四通道 log1p 特征图距离。"""
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

        positive_pixels = np.argwhere(mask == 1)

        for y, x in positive_pixels:
            pixel_center_x = x + 0.5
            pixel_center_y = y + 0.5
            dist_top = max(0, pixel_center_y - y_min)
            dist_right = max(0, x_max - pixel_center_x)
            dist_bottom = max(0, y_max - pixel_center_y)
            dist_left = max(0, pixel_center_x - x_min)

            new_area = (dist_top + dist_bottom) * (dist_left + dist_right)
            old_area = (geo_map[0, y, x] + geo_map[2, y, x]) * (geo_map[1, y, x] + geo_map[3, y, x])

            if geo_map[0, y, x] == 0 or new_area < old_area:
                geo_map[0, y, x] = dist_top
                geo_map[1, y, x] = dist_right
                geo_map[2, y, x] = dist_bottom
                geo_map[3, y, x] = dist_left

    # 对 geo_map 做 log(1+x) 变换，压缩大距离值的范围
    geo_map = np.log1p(geo_map)  # log(1+x)，x=0时结果为0，安全

    return score_map, geo_map


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
