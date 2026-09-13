"""UI 小图多后端矩形检测：默认输入高 256、宽 768 像素。

支持 PyTorch、ONNX Runtime 和 OpenVINO；auto 按后缀与依赖选择，不运行速度竞赛。
保存输入尺寸预览与可选原图裁片，详情见 detection/inference.md。"""

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

# 参数配置

DEFAULT_MODEL_PATH  = "best_detection.onnx"  # 默认小图 ONNX 权重路径，不与大图模型通用。
DEFAULT_IMAGE_PATH  = "detection/data/test/1"  # 默认待推理图片目录。
DEFAULT_OUTPUT_DIR  = "detection/data/test_results/optimized/1"  # 默认画框预览目录，覆盖同名结果。
DEFAULT_CROP_DIR    = "detection/data/crops/optimized/1"  # 开启裁片时的固定目录，独立于 --output_dir。

INPUT_SIZE          = (256, 768)  # (height, width)，与训练配置一致
SCORE_THRESHOLD     = 0.7  # 置信度严格超过此值的网格参与连通域。
MIN_AREA            = 2  # 最小连通域面积，单位为特征图像素数。

SAVE_CROPS          = False  # 启用时从原图保存 PNG 裁片，默认关闭。
EXPAND_BOX          = False  # 启用时按框高向四周扩边，默认关闭。
EXPAND_RATIO_W      = 0.10  # 左右各扩展框高的 10%，仅扩框启用时生效。
EXPAND_RATIO_H      = 0.10  # 上下各扩展框高的 10%，仅扩框启用时生效。
# 注意：宽高比过滤只用于小连通域噪点过滤，大连通域不受限制
SMALL_MAX_ASPECT      = 5.0   # 小连通域宽高比超过此值才过滤（正方形符号宽高比应接近1）


def normalize_input_size(input_size):
    """统一返回 (高, 宽)，兼容 int、二元序列和 ``256x768`` 字符串。"""
    if isinstance(input_size, str):
        parts = input_size.lower().replace("×", "x").split("x")
        if len(parts) == 1:
            input_size = int(parts[0])
        elif len(parts) == 2:
            input_size = tuple(int(part) for part in parts)
        else:
            raise ValueError(f"无效输入尺寸: {input_size!r}")
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        input_h, input_w = int(input_size[0]), int(input_size[1])
        if input_h > 0 and input_w > 0:
            return input_h, input_w
    raise ValueError(f"input_size 必须是正整数或 (高, 宽)，实际为: {input_size!r}")


def scaled_small_region_area(input_size):
    """按输入总像素面积缩放小连通域阈值，1024×1024 时基准为 15。"""
    input_h, input_w = normalize_input_size(input_size)
    scaled_area = round(15 * (input_h * input_w) / (1024 * 1024))
    return max(MIN_AREA + 1, scaled_area)


SMALL_REGION_AREA = scaled_small_region_area(INPUT_SIZE)  # 默认输入尺寸对应的小区域面积分界，单位为特征图像素数。


# 后端加载器

class ONNXBackend:
    """ONNX Runtime 后端（支持 FP32 和 INT8 模型）"""

    def __init__(self, model_path: str, input_size=INPUT_SIZE):
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
        self.input_size = normalize_input_size(input_size)
        self._warmup()
        print(f"  ✓ ONNX Runtime 后端加载完成")
        print(f"    线程数: {os.cpu_count()} / 推理线程: {opts.intra_op_num_threads}")

    def _warmup(self, n: int = 3):
        input_h, input_w = self.input_size
        dummy = np.random.randn(1, 3, input_h, input_w).astype(np.float32)
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
    OpenVINO CPU 后端，使用 THROUGHPUT 提示和自动 stream 数；性能须实测。
    支持直接从 .onnx 文件加载，无需额外转换。
    """

    def __init__(self, model_path: str, input_size=INPUT_SIZE):
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
        self.input_size = normalize_input_size(input_size)
        self._warmup()
        print(f"  ✓ OpenVINO 后端加载完成（Intel CPU 优化）")

    def _warmup(self, n: int = 3):
        input_h, input_w = self.input_size
        dummy = np.random.randn(1, 3, input_h, input_w).astype(np.float32)
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

    def __init__(self, model_path: str, input_size=INPUT_SIZE):
        import torch
        from detection.models.east_model import EASTModel

        self.model = EASTModel(pretrained=False)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        print(f"  ✓ PyTorch 后端加载完成（无优化）")

    def infer(self, img_np: np.ndarray):
        import torch
        tensor = torch.from_numpy(img_np)
        with torch.no_grad():
            score, geo = self.model(tensor)
        return score.numpy(), geo.numpy()


def load_backend(model_path: str, backend: str, input_size=INPUT_SIZE):
    """
    使用显式后端或按扩展名选择；不比较速度。
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
        return ONNXBackend(model_path, input_size)
    elif backend == "openvino":
        return OpenVINOBackend(model_path, input_size)
    elif backend == "pytorch":
        return PyTorchBackend(model_path, input_size)
    else:
        raise ValueError(f"未知后端: {backend}")


# 图像预处理（沿用训练的确定性缩放、补边和归一化）

def preprocess(image_path: str, input_size=INPUT_SIZE):
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

    input_h, input_w = normalize_input_size(input_size)

    # ResizeAndPad：保持宽高比，并使用与训练相同的边缘中位数填充
    scale = min(input_w / orig_w, input_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    pixels = np.asarray(image)
    border = np.concatenate(
        [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0
    )
    padding_color = tuple(np.median(border, axis=0).astype(np.uint8).tolist())
    padded = Image.new("RGB", (input_w, input_h), padding_color)
    padded.paste(resized, ((input_w - new_w) // 2, (input_h - new_h) // 2))

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
    取该区域所有像素 geo 预测的中位数。
    geo_map_np 已经 expm1 还原，单位为特征图像素，乘 stride 后才是输入图像素。
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
      - 小连通域（面积 < small_region_area） ：全像素中位数出框
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

        # 小连通域（单符号）
        if area < small_region_area:
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

        # 大连通域（文本行）：边缘像素采样出框
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

# 单图推理

def detect_one(backend, image_path: str, output_dir: str,
               input_size=INPUT_SIZE,
               score_threshold: float = SCORE_THRESHOLD,
               save_crops: bool = SAVE_CROPS,
               expand_box: bool = EXPAND_BOX,
               ratio_w: float = EXPAND_RATIO_W,
               ratio_h: float = EXPAND_RATIO_H,
               timing: bool = False):
    """
    对单张图像执行检测，保存可视化结果和（可选）裁剪图像。
    """
    preprocessed = preprocess(image_path, input_size)
    if preprocessed[0] is None:
        return 0, 0.0
    img_np, resized_pil, (orig_w, orig_h) = preprocessed
    input_h, input_w = normalize_input_size(input_size)

    # 前向推理
    t0 = time.perf_counter()
    score_raw, geo_raw = backend.infer(img_np)
    infer_ms = (time.perf_counter() - t0) * 1000

    # score_map: (1,1,H,W) → (H,W)
    score_map_np = score_raw[0, 0]
    # geo_map: (1,4,H,W) → (4,H,W)，从 log 空间还原
    geo_map_np = np.expm1(geo_raw[0, :4])
    geo_map_np = np.maximum(geo_map_np, 0)

    # 连通域检测
    final_boxes = boxes_from_connected_components(
        score_map_np, geo_map_np,
        score_threshold=score_threshold,
        min_area=MIN_AREA,
        small_region_area=scaled_small_region_area(input_size),
    )

    # 坐标映射参数
    scale = min(input_w / orig_w, input_h / orig_h)
    content_w = int(orig_w * scale)
    content_h = int(orig_h * scale)
    pad_x = (input_w - content_w) // 2
    pad_y = (input_h - content_h) // 2

    # 绘图 & 保存
    img_vis = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"  ✗ 原图读取失败: {image_path}")
        return 0, infer_ms

    real_h, real_w = img_orig.shape[:2]
    fname = os.path.basename(image_path)
    fname_noext = os.path.splitext(fname)[0]

    if len(final_boxes) > 0:
        sorted_boxes = final_boxes[np.lexsort((final_boxes[:, 0], final_boxes[:, 1]))]

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
                    os.path.join(DEFAULT_CROP_DIR, f"{fname_noext}-{i+1:06d}.png"),
                    crop
                )

            cv2.rectangle(
                img_vis,
                (max(0, int(x1)), max(0, int(y1))),
                (min(input_w, int(x2)), min(input_h, int(y2))),
                (0, 255, 0), 1
            )

    save_path = os.path.join(output_dir, "res_" + fname)
    cv2.imwrite(save_path, img_vis)

    if timing:
        print(f"  [{fname}] {len(final_boxes)} 框  推理: {infer_ms:.1f}ms")

    return len(final_boxes), infer_ms


# 主函数

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
    parser.add_argument(
        "--input_size", type=normalize_input_size, default=INPUT_SIZE,
        help="模型输入尺寸：正方形可写 640，矩形按高x宽写 256x768",
    )
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
    print(f"  尺寸:    {args.input_size[1]}×{args.input_size[0]} (宽×高)")
    print(f"  CPU核心: {os.cpu_count()}")
    print()

    if not os.path.exists(args.model_path):
        print(f"✗ 模型文件不存在: {args.model_path}")
        return

    # 加载后端
    backend = load_backend(args.model_path, args.backend, args.input_size)

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
