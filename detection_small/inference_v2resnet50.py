"""
inference_v2.py — 改进版推理脚本
=======================================
核心改动：用「连通域聚类 + 边界统计」替代「逐像素独立出框 + NMS」

原版问题根因分析：
  EAST 让每个正样本像素独立预测到4个边界的距离，然后每个像素产出一个完整的框。
  对于长文本行（score map 上宽 200px），中心像素需要预测 dist_left=100px。
  即使 log 空间误差仅 0.1，expm1 还原后真实误差达 11px（原图44px，约2个字）。
  边缘像素更惨：左边缘像素需预测 dist_right≈200px，误差被指数放大到 20+px。

  原版用 NMS 从这些有大量噪声的框中选一个"最佳"框，但选谁都不准。

新方案：
  1. 用 score map 做连通域分析，找到每个文本区域
  2. 对每个区域内的所有正样本像素的 geo 预测取统计值（而非单像素决定）
  3. 用「最左边缘像素的 dist_left」和「最右边缘像素的 dist_right」确定左右边界
     （边缘像素只需预测很短的距离，精度高得多！）
  4. 上下边界同理
"""

import torch
import cv2
import numpy as np
import os
from PIL import Image
import sys
from torchvision.transforms import Normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.east_model import EASTModel
from components.transforms import ResizeAndPad

# --- 配置参数 ---
MODEL_PATH = "detection_small/checkpoints/best_detection_epoch_208_loss_0.01717_resnet50.pth"
IMAGE_PATH = "detection_small/data/test/1"
OUTPUT_DIR = "detection_small/data/test_results/resnet50"

SAVE_CROPS =  False
CROP_DIR_NAME = "detection_small/data/crops/20260131_v2"

EXPAND_BOX = True
EXPAND_RATIO_W = 0.30   # 左右扩充 = 字高 * 0.3
EXPAND_RATIO_H = 0.10   # 上下扩充 = 字高 * 0.1

INPUT_SIZE = 1024
SCORE_THRESHOLD = 0.7    # 可以比原来低一些，因为连通域方法更鲁棒
MIN_AREA = 5             # 最小连通域面积（score map 像素数），过滤噪点


def load_model(model_path, device):
    model = EASTModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法读取图像: {image_path}")
        return None, None, None

    original_w, original_h = image.size
    transform = ResizeAndPad(input_size)
    sample = {'image': image, 'quads': np.array([])}
    sample = transform(sample)
    resized_image = sample['image']

    img_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float() / 255.0
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, resized_image, (original_w, original_h)


def boxes_from_connected_components(score_map_np, geo_map_np, score_threshold, min_area):
    """
    核心改进：用连通域聚类 + 边界像素统计来确定框的边界

    思路：
    - score map 二值化后做连通域分析，每个连通域是一个文本区域
    - 对于每个连通域，不是让每个像素独立出框，而是：
      * 左边界：取最左侧 N 列像素的 dist_left 的中位数
      * 右边界：取最右侧 N 列像素的 dist_right 的中位数
      * 上下边界同理
    - 中位数比平均值更鲁棒（抗异常值）
    """
    # 二值化
    binary = (score_map_np > score_threshold).astype(np.uint8)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    stride = 4.0

    for label_id in range(1, num_labels):  # 跳过背景 (label=0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # 获取该连通域的所有像素坐标
        ys, xs = np.where(labels == label_id)

        # --- 用边缘像素确定边界（核心改进）---

        # 找到该连通域的边界列/行
        x_min_region = xs.min()
        x_max_region = xs.max()
        y_min_region = ys.min()
        y_max_region = ys.max()

        region_width = x_max_region - x_min_region + 1
        region_height = y_max_region - y_min_region + 1

        # 边缘采样宽度：取连通域宽度的 10%，但至少 1 像素，至多 5 像素
        edge_w = max(1, min(5, int(region_width * 0.1)))
        edge_h = max(1, min(3, int(region_height * 0.1)))

        # --- 左边界：取最左侧 edge_w 列像素的 dist_left ---
        left_mask = (xs >= x_min_region) & (xs < x_min_region + edge_w)
        if left_mask.sum() > 0:
            left_dists = geo_map_np[3, ys[left_mask], xs[left_mask]]  # dist_left
            left_pixels_x = xs[left_mask] + 0.5
            # 左边界 = 像素位置 - dist_left，取中位数
            left_boundaries = left_pixels_x - left_dists
            x_min_final = np.median(left_boundaries)
        else:
            x_min_final = x_min_region

        # --- 右边界：取最右侧 edge_w 列像素的 dist_right ---
        right_mask = (xs > x_max_region - edge_w) & (xs <= x_max_region)
        if right_mask.sum() > 0:
            right_dists = geo_map_np[1, ys[right_mask], xs[right_mask]]  # dist_right
            right_pixels_x = xs[right_mask] + 0.5
            # 右边界 = 像素位置 + dist_right，取中位数
            right_boundaries = right_pixels_x + right_dists
            x_max_final = np.median(right_boundaries)
        else:
            x_max_final = x_max_region + 1

        # --- 上边界：取最上方 edge_h 行像素的 dist_top ---
        top_mask = (ys >= y_min_region) & (ys < y_min_region + edge_h)
        if top_mask.sum() > 0:
            top_dists = geo_map_np[0, ys[top_mask], xs[top_mask]]  # dist_top
            top_pixels_y = ys[top_mask] + 0.5
            top_boundaries = top_pixels_y - top_dists
            y_min_final = np.median(top_boundaries)
        else:
            y_min_final = y_min_region

        # --- 下边界：取最下方 edge_h 行像素的 dist_bottom ---
        bottom_mask = (ys > y_max_region - edge_h) & (ys <= y_max_region)
        if bottom_mask.sum() > 0:
            bottom_dists = geo_map_np[2, ys[bottom_mask], xs[bottom_mask]]  # dist_bottom
            bottom_pixels_y = ys[bottom_mask] + 0.5
            bottom_boundaries = bottom_pixels_y + bottom_dists
            y_max_final = np.median(bottom_boundaries)
        else:
            y_max_final = y_max_region + 1

        # 转换到原图尺度 (score map → 输入图)
        box = [
            x_min_final * stride,
            y_min_final * stride,
            x_max_final * stride,
            y_max_final * stride
        ]

        # 基本合理性检查
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    return np.array(boxes) if boxes else np.array([])


def detect(model, image_path, output_dir, input_size=1024, score_threshold=0.7,
           save_crops=False, expand_box=False, ratio_w=0.0, ratio_h=0.0):
    device = next(model.parameters()).device

    img_tensor, resized_image, (orig_w, orig_h) = preprocess_image(image_path, input_size)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score_map, geo_map = model(img_tensor)

    score_map_np = score_map[0, 0].cpu().numpy()
    geo_map_np = geo_map[0, :4, :, :].cpu().numpy()

    # 从 log 空间还原
    geo_map_np = np.expm1(geo_map_np)
    geo_map_np = np.maximum(geo_map_np, 0)

    # ★ 核心改进：连通域聚类 + 边界统计
    final_boxes = boxes_from_connected_components(
        score_map_np, geo_map_np, score_threshold, MIN_AREA
    )

    # 准备图像
    img_vis = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"原图读取失败: {image_path}")
        return

    real_h, real_w = img_original.shape[:2]
    filename_full = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename_full)[0]

    # 坐标映射参数
    scale = input_size / max(orig_h, orig_w)
    current_content_w = int(orig_w * scale)
    current_content_h = int(orig_h * scale)
    pad_x = (input_size - current_content_w) // 2
    pad_y = (input_size - current_content_h) // 2

    if len(final_boxes) > 0:
        sorted_indices = np.argsort(final_boxes[:, 1])
        sorted_boxes = final_boxes[sorted_indices]

        if save_crops and not os.path.exists(CROP_DIR_NAME):
            os.makedirs(CROP_DIR_NAME)

        for i, box in enumerate(sorted_boxes):
            x_min, y_min, x_max, y_max = box

            if expand_box:
                h_box = y_max - y_min
                dx = h_box * ratio_w
                dy = h_box * ratio_h
                x_min -= dx
                x_max += dx
                y_min -= dy
                y_max += dy

            # 映射回原图
            real_x_min = (x_min - pad_x) / scale
            real_x_max = (x_max - pad_x) / scale
            real_y_min = (y_min - pad_y) / scale
            real_y_max = (y_max - pad_y) / scale

            real_x_min = max(0, int(real_x_min))
            real_y_min = max(0, int(real_y_min))
            real_x_max = min(real_w, int(real_x_max))
            real_y_max = min(real_h, int(real_y_max))

            if real_x_max - real_x_min <= 0 or real_y_max - real_y_min <= 0:
                continue

            if save_crops:
                crop_img = img_original[real_y_min:real_y_max, real_x_min:real_x_max]
                crop_filename = f"{filename_no_ext}-{i + 1:06d}.jpg"
                crop_save_path = os.path.join(CROP_DIR_NAME, crop_filename)
                cv2.imwrite(crop_save_path, crop_img)

            draw_x_min = max(0, int(x_min))
            draw_y_min = max(0, int(y_min))
            draw_x_max = min(input_size, int(x_max))
            draw_y_max = min(input_size, int(y_max))
            cv2.rectangle(img_vis, (draw_x_min, draw_y_min), (draw_x_max, draw_y_max), (0, 255, 0), 1)

    save_path = os.path.join(output_dir, "res_" + filename_full)
    cv2.imwrite(save_path, img_vis)
    print(f"[{filename_full}] (原图: {real_w}x{real_h}) 检测到 {len(final_boxes)} 个文本框")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("-" * 30)
    print(f"模型路径: {MODEL_PATH}")
    print(f"推理模式: 连通域聚类 + 边界统计 (v2)")
    print(f"Score 阈值: {SCORE_THRESHOLD}")
    print(f"最小区域面积: {MIN_AREA} px")
    print("-" * 30)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在。")
        return

    model = load_model(MODEL_PATH, device)

    if os.path.isdir(IMAGE_PATH):
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))

        if not image_files:
            print(f"警告: 在目录中未找到图像。")

        print(f"共找到 {len(image_files)} 张图像，开始推理...")
        for img_file in image_files:
            detect(model, img_file, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD,
                   save_crops=SAVE_CROPS, expand_box=EXPAND_BOX,
                   ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H)

    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD,
               save_crops=SAVE_CROPS, expand_box=EXPAND_BOX,
               ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H)
    else:
        print(f"错误: 路径不存在 -> {IMAGE_PATH}")


if __name__ == '__main__':
    main()
