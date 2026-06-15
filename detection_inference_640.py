"""
inference_optimized.py — 多后端优化推理脚本
=====================================================================
支持三种后端（自动按速度选择）：
  1. ONNX Runtime FP32   （无损，2-3x 提速）
  2. ONNX Runtime INT8   （极小精度损失，4-6x 提速）
  3. OpenVINO            （Intel CPU 专属，5-10x 提速）
  4. PyTorch（原始，作为 fallback）

推理逻辑完全复用 inference_v2.py 的连通域聚类方案（高精度框定位）。
只替换了「模型加载 + 前向推理」部分。

用法:
    # 自动选最优后端
    python inference_optimized.py \
        --model_path detection/checkpoints/east_model_int8_static.onnx \
        --image_path detection/data/test/1 \
        --output_dir detection/data/test_results/optimized

    # 指定 OpenVINO 后端
    python inference_optimized.py \
        --model_path detection/checkpoints/east_model.onnx \
        --backend openvino \
        --image_path your_image.jpg

依赖:
    pip install onnxruntime onnxsim
    # OpenVINO（可选，Intel CPU 专属）:
    pip install openvino
"""

import os
import sys
import cv2
import time
import glob
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 参数配置
# ============================================================

DEFAULT_MODEL_PATH  = "best_detection_int8_static.onnx"
DEFAULT_IMAGE_PATH  = "detection/data/test/1"
DEFAULT_OUTPUT_DIR  = "detection/data/test_results/optimized/1"
DEFAULT_CROP_DIR    = "detection/data/crops/optimized/1"

INPUT_SIZE          = 640
SCORE_THRESHOLD     = 0.7
MIN_AREA            = 2

SAVE_CROPS          = False
EXPAND_BOX          = True
EXPAND_RATIO_W      = 0.10
EXPAND_RATIO_H      = 0.10
SMALL_REGION_AREA     = 6    # 面积 < 此值视为小符号，走不同出框策略
# 注意：宽高比过滤只用于小连通域噪点过滤，大连通域不受限制
SMALL_MAX_ASPECT      = 5.0   # 小连通域宽高比超过此值才过滤（正方形符号宽高比应接近1）


# ============================================================
# 后端加载器
# ============================================================

class ONNXBackend:
    """ONNX Runtime 后端（支持 FP32 和 INT8 模型）"""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # 开启全部图优化（自动算子融合：Conv+BN+ReLU 合并等）
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 使用所有可用 CPU 核心
        opts.intra_op_num_threads = os.cpu_count()
        opts.inter_op_num_threads = max(1, os.cpu_count() // 2)
        # 启用内存优化
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse   = True

        self.session = ort.InferenceSession(
            model_path, opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self._warmup()
        print(f"  ✓ ONNX Runtime 后端加载完成")
        print(f"    线程数: {os.cpu_count()} / 推理线程: {opts.intra_op_num_threads}")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})

    def infer(self, img_np: np.ndarray):
        """
        img_np: (1, 3, H, W) float32 numpy array
        Returns: score_map (1,1,H,W), geo_map (1,4,H,W) numpy arrays
        """
        outputs = self.session.run(None, {self.input_name: img_np})
        return outputs[0], outputs[1]


class OpenVINOBackend:
    """
    OpenVINO 后端（Intel CPU 专属，比 ONNX Runtime 再快 2-3x）
    支持直接从 .onnx 文件加载，无需额外转换。
    """

    def __init__(self, model_path: str):
        try:
            # from openvino.runtime import Core
            from openvino import Core
        except ImportError:
            raise ImportError(
                "OpenVINO 未安装。请运行: pip install openvino\n"
                "或使用 --backend onnx 改用 ONNX Runtime"
            )

        ie = Core()
        # 直接从 ONNX 加载（OpenVINO 原生支持）
        model = ie.read_model(model=model_path)
        # 编译到 CPU，开启性能模式
        config = {
            "PERFORMANCE_HINT": "THROUGHPUT",          # 吞吐量优先
            "NUM_STREAMS": "AUTO",
        }
        self.compiled_model = ie.compile_model(model=model, device_name="CPU", config=config)
        self.infer_request = self.compiled_model.create_infer_request()
        self._warmup()
        print(f"  ✓ OpenVINO 后端加载完成（Intel CPU 优化）")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        for _ in range(n):
            self.infer_request.infer({"input": dummy})

    def infer(self, img_np: np.ndarray):
        result = self.infer_request.infer({"input": img_np})
        outputs = list(result.values())
        # OpenVINO 输出顺序可能不固定，按形状判断
        out_list = sorted(outputs, key=lambda x: x.shape[1])  # 通道数少的是 score
        return out_list[0], out_list[1]


class PyTorchBackend:
    """原始 PyTorch 后端（fallback，无优化）"""

    def __init__(self, model_path: str):
        import torch
        from detection.models.east_model import EASTModel

        self.model = EASTModel(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        print(f"  ✓ PyTorch 后端加载完成（无优化）")

    def infer(self, img_np: np.ndarray):
        import torch
        tensor = torch.from_numpy(img_np)
        with torch.no_grad():
            score, geo = self.model(tensor)
        return score.numpy(), geo.numpy()


def load_backend(model_path: str, backend: str):
    """
    根据指定后端或文件扩展名自动选择最优后端。
    自动选择逻辑：
      .onnx → 优先 ONNX Runtime
      .pth  → PyTorch
      backend='openvino' → OpenVINO（需要安装）
    """
    ext = os.path.splitext(model_path)[1].lower()

    if backend == "auto":
        if ext == ".onnx":
            backend = "onnx"
        elif ext == ".pth":
            backend = "pytorch"
        else:
            backend = "onnx"

    print(f"\n[后端] 使用: {backend.upper()}")

    if backend == "onnx":
        return ONNXBackend(model_path)
    elif backend == "openvino":
        return OpenVINOBackend(model_path)
    elif backend == "pytorch":
        return PyTorchBackend(model_path)
    else:
        raise ValueError(f"未知后端: {backend}")


# ============================================================
# 图像预处理（与训练保持完全一致）
# ============================================================

def preprocess(image_path: str, input_size: int):
    """
    加载 & 预处理图像。
    返回: (img_tensor_np, resized_pil_img, (orig_w, orig_h))
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ✗ 无法读取图像 {image_path}: {e}")
        return None, None, None

    orig_w, orig_h = image.size

    # ResizeAndPad：保持宽高比
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    padded = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    padded.paste(resized, ((input_size - new_w) // 2, (input_size - new_h) // 2))

    # HWC→CHW, float32, ImageNet normalize
    arr = np.array(padded).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)

    return arr, padded, (orig_w, orig_h)


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

    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=4
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

# ============================================================
# 单图推理
# ============================================================

def detect_one(backend, image_path: str, output_dir: str,
               input_size: int = INPUT_SIZE,
               score_threshold: float = SCORE_THRESHOLD,
               save_crops: bool = SAVE_CROPS,
               expand_box: bool = EXPAND_BOX,
               ratio_w: float = EXPAND_RATIO_W,
               ratio_h: float = EXPAND_RATIO_H,
               timing: bool = False):
    """
    对单张图像执行检测，保存可视化结果和（可选）裁剪图像。
    """
    img_np, resized_pil, (orig_w, orig_h) = preprocess(image_path, input_size)
    if img_np is None:
        return 0, 0.0

    # --- 前向推理 ---
    t0 = time.perf_counter()
    score_raw, geo_raw = backend.infer(img_np)
    infer_ms = (time.perf_counter() - t0) * 1000

    # score_map: (1,1,H,W) → (H,W)
    score_map_np = score_raw[0, 0]
    # geo_map: (1,4,H,W) → (4,H,W)，从 log 空间还原
    geo_map_np = np.expm1(geo_raw[0, :4])
    geo_map_np = np.maximum(geo_map_np, 0)

    # --- 连通域检测 ---
    final_boxes = boxes_from_connected_components(
        score_map_np, geo_map_np,
        score_threshold=score_threshold,
        min_area=MIN_AREA,
    )

    # --- 坐标映射参数 ---
    scale = input_size / max(orig_h, orig_w)
    content_w = int(orig_w * scale)
    content_h = int(orig_h * scale)
    pad_x = (input_size - content_w) // 2
    pad_y = (input_size - content_h) // 2

    # --- 绘图 & 保存 ---
    img_vis = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"  ✗ 原图读取失败: {image_path}")
        return 0, infer_ms

    real_h, real_w = img_orig.shape[:2]
    fname = os.path.basename(image_path)
    fname_noext = os.path.splitext(fname)[0]

    if len(final_boxes) > 0:
        sorted_boxes = final_boxes[np.argsort(final_boxes[:, 1])]

        if save_crops:
            os.makedirs(DEFAULT_CROP_DIR, exist_ok=True)

        for i, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = box.copy()

            if expand_box:
                dh = y2 - y1
                x1 -= dh * ratio_w;  x2 += dh * ratio_w
                y1 -= dh * ratio_h;  y2 += dh * ratio_h

            # 映射回原图坐标
            rx1 = max(0,      int((x1 - pad_x) / scale))
            ry1 = max(0,      int((y1 - pad_y) / scale))
            rx2 = min(real_w, int((x2 - pad_x) / scale))
            ry2 = min(real_h, int((y2 - pad_y) / scale))

            if rx2 - rx1 <= 0 or ry2 - ry1 <= 0:
                continue

            if save_crops:
                crop = img_orig[ry1:ry2, rx1:rx2]
                cv2.imwrite(
                    os.path.join(DEFAULT_CROP_DIR, f"{fname_noext}-{i+1:06d}.jpg"),
                    crop
                )

            cv2.rectangle(
                img_vis,
                (max(0, int(x1)), max(0, int(y1))),
                (min(input_size, int(x2)), min(input_size, int(y2))),
                (0, 255, 0), 1
            )

    save_path = os.path.join(output_dir, "res_" + fname)
    cv2.imwrite(save_path, img_vis)

    if timing:
        print(f"  [{fname}] {len(final_boxes)} 框  推理: {infer_ms:.1f}ms")

    return len(final_boxes), infer_ms


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EAST 优化推理（ONNX/INT8/OpenVINO）")
    parser.add_argument("--model_path",  type=str,
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path",  type=str,
                        default=DEFAULT_IMAGE_PATH,
                        help="推理需要的图像文件路径")
    parser.add_argument("--output_dir",  type=str,
                        default=DEFAULT_OUTPUT_DIR,
                        help="推理产生的图像文件路径")
    parser.add_argument("--backend",     type=str, choices=["auto", "onnx", "openvino", "pytorch"],
                        default="openvino",
                        help="推理后端（auto=自动选择）")
    parser.add_argument("--input_size",  type=int,
                        default=INPUT_SIZE)
    parser.add_argument("--score_thr",   type=float,
                        default=SCORE_THRESHOLD)
    parser.add_argument("--save_crops",  action="store_true",
                        default=SAVE_CROPS)
    parser.add_argument("--benchmark",   action="store_true",
                        help="显示每张图的推理耗时")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("EAST 优化推理")
    print("=" * 60)
    print(f"  模型:    {args.model_path}")
    print(f"  输入:    {args.image_path}")
    print(f"  输出:    {args.output_dir}")
    print(f"  后端:    {args.backend}")
    print(f"  CPU核心: {os.cpu_count()}")
    print()

    if not os.path.exists(args.model_path):
        print(f"✗ 模型文件不存在: {args.model_path}")
        return

    # 加载后端
    backend = load_backend(args.model_path, args.backend)

    # 收集图像文件
    if os.path.isdir(args.image_path):
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(glob.glob(os.path.join(args.image_path, ext)))
        images.sort()
    elif os.path.isfile(args.image_path):
        images = [args.image_path]
    else:
        print(f"✗ 路径不存在: {args.image_path}")
        return

    if not images:
        print("✗ 未找到图像文件")
        return

    print(f"\n共 {len(images)} 张图像，开始推理...\n")

    # 批量推理
    total_boxes = 0
    total_ms = 0.0
    t_wall_start = time.perf_counter()

    for img_path in images:
        n_boxes, ms = detect_one(
            backend, img_path, args.output_dir,
            input_size=args.input_size,
            score_threshold=args.score_thr,
            save_crops=args.save_crops,
            timing=args.benchmark
        )
        total_boxes += n_boxes
        total_ms += ms

    t_wall = time.perf_counter() - t_wall_start

    # 汇总统计
    print(f"\n{'='*60}")
    print(f"推理完成")
    print(f"  图像数:         {len(images)} 张")
    print(f"  检测框总数:     {total_boxes}")
    print(f"  纯推理均值:     {total_ms/len(images):.1f} ms/张")
    print(f"  总耗时(含后处理): {t_wall*1000:.0f} ms")
    print(f"  吞吐量:         {len(images)/t_wall:.2f} 张/秒")
    print(f"  结果保存在:     {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()