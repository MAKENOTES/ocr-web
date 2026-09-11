"""复用原 EAST 真值算法，仅为 ### 区域增加训练掩码。"""
import numpy as np

from detection.components.ground_truth import generate_ground_truth, shrink_polygon


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
