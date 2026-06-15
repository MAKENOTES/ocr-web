"""
inference_v2.py — 改进版推理脚本 v3
=======================================
核心改动：用「连通域聚类 + 边界统计」替代「逐像素独立出框 + NMS」

v3 修复说明（相对原始版本的改动）：
  1. MIN_AREA 从 5 降到 2，修复 √ 等小符号漏检
  2. 小连通域（面积 < SMALL_REGION_AREA）用全像素中位数出框，不用边缘采样
  3. 宽高比过滤【仅对小连通域生效】，大连通域（正文长文本行）绝对不过滤
     （正文行在 score map 上是扁长条，宽高比天然很大，不能用宽高比过滤）
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
MODEL_PATH = "detection/checkpoints/best_detection_epoch_384_loss_0.01133_resnet_640.pth"
IMAGE_PATH = "detection/data/test/2"
OUTPUT_DIR = "detection/data/test_results/resnet18"

SAVE_CROPS = True
CROP_DIR_NAME = "detection/data/crops/20260131_v2"

EXPAND_BOX = True
EXPAND_RATIO_W = 0.30   # 左右扩充 = 字高 * 0.3
EXPAND_RATIO_H = 0.10   # 上下扩充 = 字高 * 0.1

INPUT_SIZE = 1024
SCORE_THRESHOLD       = 0.7   # 文本行阈值
MIN_AREA              = 2     # 最小连通域面积（score map 像素数），从 5 降到 2
SMALL_REGION_AREA     = 15    # 面积 < 此值视为小符号，走不同出框策略
# 注意：宽高比过滤只用于小连通域噪点过滤，大连通域不受限制
SMALL_MAX_ASPECT      = 5.0   # 小连通域宽高比超过此值才过滤（正方形符号宽高比应接近1）


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


def box_from_small_region(ys, xs, geo_map_np, stride):
    """
    针对小连通域（√ 等单字符/符号）的出框策略。
    取所有像素 geo 预测的中位数，比边缘采样更稳定。
    geo_map_np 此时已经过 expm1 还原，是真实像素距离。
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
                                     min_area):
    """
    单轮连通域检测：

      - 大连通域（面积 >= SMALL_REGION_AREA）：边缘像素采样，精度高
        ★ 不做宽高比过滤，正文长文本行天然是扁长条
      - 小连通域（面积 < SMALL_REGION_AREA） ：全像素中位数出框
        ★ 对小连通域做宽高比过滤，去除细线噪点
    """
    stride = 4.0
    boxes = []

    binary_high = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_high, connectivity=8
    )

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == label_id)

        # ── 小连通域（单符号）──────────────────────────────
        if area < SMALL_REGION_AREA:
            # 小连通域才做宽高比过滤（去除细线噪点）
            # √ 等符号宽高比接近 1，细线噪点宽高比极大
            w_cc = stats[label_id, cv2.CC_STAT_WIDTH]
            h_cc = stats[label_id, cv2.CC_STAT_HEIGHT]
            if max(w_cc, h_cc) / (min(w_cc, h_cc) + 1e-6) > SMALL_MAX_ASPECT:
                continue
            box = box_from_small_region(ys, xs, geo_map_np, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        # ── 大连通域（文本行）：边缘像素采样出框 ────────────
        # ★ 不加宽高比过滤，正文长文本行宽高比本来就大
        x_min_r = xs.min();  x_max_r = xs.max()
        y_min_r = ys.min();  y_max_r = ys.max()
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


def detect(model, image_path, output_dir, input_size=640, score_threshold=0.7,
           save_crops=False, expand_box=False, ratio_w=0.0, ratio_h=0.0):
    device = next(model.parameters()).device

    img_tensor, resized_image, (orig_w, orig_h) = preprocess_image(image_path, input_size)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score_map, geo_map = model(img_tensor)

    score_map_np = score_map[0, 0].cpu().numpy()
    geo_map_np   = geo_map[0, :4, :, :].cpu().numpy()

    # 从 log 空间还原为真实像素距离
    geo_map_np = np.expm1(geo_map_np)
    geo_map_np = np.maximum(geo_map_np, 0)

    final_boxes = boxes_from_connected_components(
        score_map_np, geo_map_np,
        score_threshold=score_threshold,
        min_area=MIN_AREA,
    )

    # 准备图像
    img_vis      = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"原图读取失败: {image_path}")
        return

    real_h, real_w  = img_original.shape[:2]
    filename_full   = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename_full)[0]

    # 坐标映射参数（输入图 → 原图）
    scale             = input_size / max(orig_h, orig_w)
    current_content_w = int(orig_w * scale)
    current_content_h = int(orig_h * scale)
    pad_x             = (input_size - current_content_w) // 2
    pad_y             = (input_size - current_content_h) // 2

    if len(final_boxes) > 0:
        sorted_indices = np.argsort(final_boxes[:, 1])
        sorted_boxes   = final_boxes[sorted_indices]

        if save_crops and not os.path.exists(CROP_DIR_NAME):
            os.makedirs(CROP_DIR_NAME)

        for i, box in enumerate(sorted_boxes):
            x_min, y_min, x_max, y_max = box

            if expand_box:
                h_box  = y_max - y_min
                dx     = h_box * ratio_w
                dy     = h_box * ratio_h
                x_min -= dx
                x_max += dx
                y_min -= dy
                y_max += dy

            # 映射回原图坐标
            real_x_min = max(0,      int((x_min - pad_x) / scale))
            real_y_min = max(0,      int((y_min - pad_y) / scale))
            real_x_max = min(real_w, int((x_max - pad_x) / scale))
            real_y_max = min(real_h, int((y_max - pad_y) / scale))

            if real_x_max - real_x_min <= 0 or real_y_max - real_y_min <= 0:
                continue

            if save_crops:
                crop_img       = img_original[real_y_min:real_y_max, real_x_min:real_x_max]
                crop_filename  = f"{filename_no_ext}-{i + 1:06d}.jpg"
                crop_save_path = os.path.join(CROP_DIR_NAME, crop_filename)
                cv2.imwrite(crop_save_path, crop_img)

            draw_x_min = max(0,          int(x_min))
            draw_y_min = max(0,          int(y_min))
            draw_x_max = min(input_size, int(x_max))
            draw_y_max = min(input_size, int(y_max))
            cv2.rectangle(img_vis,
                          (draw_x_min, draw_y_min),
                          (draw_x_max, draw_y_max),
                          (0, 255, 0), 1)

    save_path = os.path.join(output_dir, "res_" + filename_full)
    cv2.imwrite(save_path, img_vis)
    print(f"[{filename_full}] (原图: {real_w}x{real_h}) 检测到 {len(final_boxes)} 个文本框")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("-" * 30)
    print(f"模型路径:       {MODEL_PATH}")
    print(f"推理模式:       连通域检测 (v3)")
    print(f"阈值:           {SCORE_THRESHOLD}")
    print(f"最小面积:       {MIN_AREA} px")
    print(f"小符号面积上限: {SMALL_REGION_AREA} px")
    print("-" * 30)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 -> {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH, device)

    if os.path.isdir(IMAGE_PATH):
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))

        if not image_files:
            print(f"警告: 在目录 {IMAGE_PATH} 中未找到图像。")
            return

        print(f"共找到 {len(image_files)} 张图像，开始推理...")
        for img_file in sorted(image_files):
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