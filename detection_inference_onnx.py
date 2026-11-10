"""统一检测 ONNX CPU 推理：单文件部署入口。

复制本文件和 .onnx 模型即可部署，不依赖 detection 包。依赖 numpy、Pillow、
OpenCV 和 onnxruntime；使用 --backend openvino 时另需 openvino 和 onnx。
预处理、解码、NMS 和输出契约与 detection 包保持一致，修改任一侧后应同步
另一边并通过 tests/test_detection_unified_onnx.py 的对比测试。
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageStat


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}  # 支持读取的图片扩展名，比较前统一转小写。
STRIDE = 4  # 网络输出相对输入的步长，每个特征图像素对应 4 个输入像素。
INPUT_ALIGN = 32  # ResNet 下采样总倍数；输入高宽必须是它的整数倍。
MIN_INPUT_SIDE = 64  # 单边最小输入像素，保证四级下采样后仍有特征图。
MIN_AREA = 1  # 连通域最小面积，单位为置信度图像素数；保留单个高置信度像素的小字。
WORD_SPLIT_MIN_HEIGHT = 8  # 低于该高度的连通域不拆分，避免把单字符和符号切碎，单位为特征图像素。
WORD_SPLIT_LOW_COLUMN_RATIO = 0.08  # 列占用低于连通域高度的该比例时视为潜在词间空隙。
WORD_SPLIT_MIN_GAP_PIXELS = 2  # 潜在空隙的最小宽度，单位为特征图像素；过滤字符内部的小孔。
WORD_SPLIT_GAP_RATIO = 0.18  # 潜在空隙宽度至少为连通域高度的该比例才拆分。
MODEL_FORMAT = "unified-east-resnet18-v1"  # ONNX 元数据中的模型契约版本。
ONNX_CONFIG_KEY = "ocrdet_config"  # ONNX 元数据中的统一检测配置字段名。
CROP_INDEX_DIGITS = 5  # 裁片文件名序号位数，序号从 1 开始并左侧补零。
CPU_THREADS = 4  # CPU 推理最多使用的工作线程数，避免大图推理占满整机。
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet RGB 归一化均值。
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet RGB 归一化标准差。

MODEL_PATH = Path("best_detection.onnx")  # 已导出的统一检测 ONNX；本入口只接受 .onnx。
BACKEND = "openvino"  # auto 固定使用 ONNX Runtime；openvino 需显式选择。
WARMUP_RUNS = 3  # 正式计时前的模型前向预热次数，0 表示关闭。
EXPAND_BOX = False  # 启用时扩展最终框；默认关闭，避免表格相邻单元格混入裁片。
EXPAND_RATIO_W = 0.10  # 左右各扩展框高的比例，仅 EXPAND_BOX 启用时使用。
EXPAND_RATIO_H = 0.10  # 上下各扩展框高的比例，仅 EXPAND_BOX 启用时使用。
IMAGE_PATH = Path("detection/data/test_images")  # 输入图片或目录，不直接读取 PDF。
OUTPUT_DIR = Path("detection/data/test_results/onnx")  # 画框图与标签目录，重复运行覆盖同名结果。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，范围 (0, 1)。
NMS_IOU_THRESHOLD = 0.3  # 重复框抑制的交并比阈值，范围 (0, 1]。
SAVE_CROPS = False  # 启用时按原图文字框保存 PNG 裁片。
CROP_DIR = Path("detection/data/crops/onnx")  # 裁片平铺目录，覆盖同名裁片并清理该图多余旧序号。


# 图片读取与预处理：与 detection/components/image.py、transforms.py 行为一致。

def load_rgb_image(image_path):
    """读取图片、应用 EXIF 方向并合成为 RGB；透明图按可见前景亮度选黑/白底。"""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        has_transparency = "A" in image.getbands() or "transparency" in source.info
        if not has_transparency:
            return image.convert("RGB").copy()

        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        if alpha_min == 255:
            return rgba.convert("RGB")

        if alpha_max == 0:
            background_value = 255
        else:
            visible_luminance = ImageStat.Stat(rgba.convert("L"), mask=alpha).mean[0]
            background_value = 0 if visible_luminance >= 128 else 255

        background = Image.new("RGBA", rgba.size,
                               (background_value, background_value, background_value, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def border_fill_color(image):
    """用四周边缘像素的通道中位数估计补边背景色，兼容灰度输入。"""
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        border = np.concatenate((pixels[0, :], pixels[-1, :],
                                 pixels[:, 0], pixels[:, -1]), axis=0)
        return int(np.median(border))

    channels = pixels.shape[2]
    border = np.concatenate((
        pixels[0, :, :].reshape(-1, channels),
        pixels[-1, :, :].reshape(-1, channels),
        pixels[:, 0, :].reshape(-1, channels),
        pixels[:, -1, :].reshape(-1, channels),
    ), axis=0)
    return tuple(int(value) for value in np.median(border, axis=0))


def normalize_input_size(input_size):
    """返回 (高, 宽) 像素尺寸；各边至少 64 且为 32 的倍数。"""
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if (not isinstance(input_size, (tuple, list)) or len(input_size) != 2
            or any(not isinstance(value, int) or value < MIN_INPUT_SIDE
                   or value % INPUT_ALIGN for value in input_size)):
        raise ValueError("input_size must contain height/width >= 64 divisible by 32")
    return tuple(input_size)


def _align_up(value, limit):
    """把缩放后的边长向上取整到 32 的倍数，并限制在最大尺寸内。"""
    aligned = int(np.ceil(value / INPUT_ALIGN)) * INPUT_ALIGN
    return min(limit, max(MIN_INPUT_SIDE, aligned))


def dynamic_input_size(image_size, max_size):
    """按原图宽高比计算推理输入尺寸，返回 (高, 宽)。"""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive: {image_size!r}")
    max_height, max_width = normalize_input_size(max_size)
    scale = min(max_width / width, max_height / height)
    return (_align_up(height * scale, max_height),
            _align_up(width * scale, max_width))


def resize_metadata(image_size, input_size):
    """返回 (缩放比例, (水平偏移, 垂直偏移))，坐标单位为输入图像素。"""
    target_height, target_width = normalize_input_size(input_size)
    original_width, original_height = image_size
    scale = min(target_width / original_width, target_height / original_height)
    offset = ((target_width - int(original_width * scale)) // 2,
              (target_height - int(original_height * scale)) // 2)
    return scale, offset


def prepare_image(image, input_size):
    """等比缩放、边缘背景色居中补边并做 ImageNet 归一化。

    返回 (3,H,W) float32 张量和坐标还原参数；调用方负责增加 batch 维。
    """
    target_height, target_width = normalize_input_size(input_size)
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    padded = Image.new("RGB", (target_width, target_height), border_fill_color(image))
    padded.paste(resized, ((target_width - resized_width) // 2,
                           (target_height - resized_height) // 2))

    pixels = np.asarray(padded, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return pixels.transpose(2, 0, 1), resize_metadata(image.size, (target_height, target_width))


def restore_boxes(boxes, metadata, image_size):
    """将输入图上的 xyxy 框减去居中偏移、除以缩放比例，并裁到原图边界。"""
    scale, (offset_x, offset_y) = metadata
    result = boxes.copy()
    result[:, [0, 2]] = np.clip((result[:, [0, 2]] - offset_x) / scale, 0, image_size[0])
    result[:, [1, 3]] = np.clip((result[:, [1, 3]] - offset_y) / scale, 0, image_size[1])
    return result


# 连通域解码：与 detection/components/postprocess.py 行为一致。

def scaled_small_region_area(input_size):
    """按输入像素面积缩放小连通域阈值，1024×1024 对应 15 个置信度图像素。"""
    height, width = normalize_input_size(input_size)
    return max(MIN_AREA + 1, round(15 * height * width / (1024 * 1024)))


def box_from_small_region(ys, xs, geo_map_np, stride):
    """小连通域出框：取区域内全部像素的几何中位数。"""
    d_top, d_right, d_bottom, d_left = (np.median(geo_map_np[index, ys, xs])
                                       for index in range(4))
    cy, cx = np.median(ys), np.median(xs)
    return [
        (cx + 0.5 - d_left) * stride,
        (cy + 0.5 - d_top) * stride,
        (cx + 0.5 + d_right) * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]


def box_from_large_region(ys, xs, geo_map_np, stride):
    """大连通域出框：左右上下边缘带分别取几何中位数。"""
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

    return [x_min_final * stride, y_min_final * stride,
            x_max_final * stride, y_max_final * stride]


def split_component(ys, xs):
    """按低占用列把粘连的文字连通域拆成词/短语片段。"""
    height = int(ys.max() - ys.min() + 1) if len(ys) else 0
    if height < WORD_SPLIT_MIN_HEIGHT or len(xs) < 2:
        return [(ys, xs)]

    x_min, x_max = int(xs.min()), int(xs.max())
    width = x_max - x_min + 1
    counts = np.bincount(xs - x_min, minlength=width)
    low_limit = max(1, int(WORD_SPLIT_LOW_COLUMN_RATIO * height))
    low = counts <= low_limit

    cuts = []
    start = None
    for index in range(width + 1):
        is_low = index < width and bool(low[index])
        if is_low and start is None:
            start = index
        elif not is_low and start is not None:
            gap_start, gap_end = start, index - 1
            gap_width = gap_end - gap_start + 1
            if (gap_start > 0 and gap_end < width - 1
                    and gap_width >= max(WORD_SPLIT_MIN_GAP_PIXELS,
                                         int(WORD_SPLIT_GAP_RATIO * height))):
                cuts.append((gap_start + x_min, gap_end + x_min))
            start = None

    if not cuts:
        return [(ys, xs)]

    segments = []
    current = x_min
    for gap_start, gap_end in cuts:
        mask = (xs >= current) & (xs < gap_start)
        if mask.any():
            segments.append((ys[mask], xs[mask]))
        current = gap_end + 1
    mask = xs >= current
    if mask.any():
        segments.append((ys[mask], xs[mask]))
    return segments or [(ys, xs)]


def boxes_from_connected_components(score_map_np, geo_map_np, score_threshold,
                                     min_area, small_region_area, split_word_gaps=True):
    """单轮四连通域检测，返回输入图像素坐标的 xyxy 框。"""
    boxes = []
    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ys, xs = np.where(labels == label_id)

        if area < small_region_area:
            box = box_from_small_region(ys, xs, geo_map_np, float(STRIDE))
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        segments = split_component(ys, xs) if split_word_gaps else [(ys, xs)]
        for segment_ys, segment_xs in segments:
            if len(segment_xs) < min_area:
                continue
            if len(segment_xs) < small_region_area:
                box = box_from_small_region(segment_ys, segment_xs, geo_map_np, float(STRIDE))
                if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                    boxes.append(box)
            else:
                box = box_from_large_region(segment_ys, segment_xs, geo_map_np, float(STRIDE))
                if box[2] - box[0] > 2 and box[3] - box[1] > 2:
                    boxes.append(box)

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def decode_maps(score, geometry, threshold=SCORE_THRESHOLD, split_word_gaps=True):
    """由置信度连通域解码输入图像素坐标框，按置信度降序返回。"""
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


def suppress_boxes(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD):
    """按置信度用 NumPy 执行 NMS，返回保留框索引；空输入返回空索引。"""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if not len(boxes):
        return np.empty(0, dtype=np.int64)

    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area = (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1])
        other = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / np.maximum(area + other - intersection, 1e-8)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


# 输出辅助：与 detection/components/inference_utils.py 行为一致。

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


# ONNX/OpenVINO 后端：输入尺寸和元数据契约与导出入口一致。

def validate_config(config):
    """拒绝格式不符或改过尺寸、预处理、几何契约的 ONNX 配置。"""
    if not isinstance(config, dict) or config.get("format") != MODEL_FORMAT:
        raise ValueError(f"Expected a unified EAST ONNX model: {config!r}")
    try:
        input_size = normalize_input_size(config["input_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid unified detection config: {config!r}: {error}") from error
    expected = {
        "format": MODEL_FORMAT,
        "input_size": list(input_size),
        "stride": STRIDE,
        "geometry": "log1p-trbl-feature-pixels",
        "preprocess": "rgb-imagenet-center-auto-v1",
        "dynamic_spatial": bool(config.get("dynamic_spatial", False)),
    }
    if config != expected:
        raise ValueError(f"Incompatible unified detection config: {config!r}; expected {expected}")
    return config


def validate_metadata(properties, model_path):
    """读取并校验 ONNX 元数据中的统一检测配置。"""
    try:
        config = json.loads(properties[ONNX_CONFIG_KEY])
        validate_config(config)
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid unified detection ONNX config: {model_path}: {error}") from error
    return config


def static_spatial_size(shape):
    """返回 ONNX 输入 shape 的静态 (高, 宽)；动态维用 None 表示。"""
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D ONNX input shape: {shape!r}")
    return tuple(value if isinstance(value, int) and value > 0 else None for value in shape[2:])


class OnnxRuntimeBackend:
    """加载统一检测 ONNX，校验元数据/输入输出并执行 CPU 前向。"""

    def __init__(self, model_path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = CPU_THREADS
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = True
        options.enable_mem_reuse = True
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.config = validate_metadata(
            self.session.get_modelmeta().custom_metadata_map, model_path)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "image" or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3
                or {value.name for value in outputs} != {"score", "geometry"}):
            raise ValueError(f"Incompatible unified detection ONNX inputs/outputs: {model_path}")
        static_size = static_spatial_size(inputs[0].shape)
        if None not in static_size and static_size != tuple(self.config["input_size"]):
            raise ValueError(f"ONNX input size {static_size} does not match config "
                             f"{self.config['input_size']}: {model_path}")
        self.static_input_size = None if None in static_size else tuple(static_size)

    def infer(self, batch):
        """执行一次 NCHW 前向，返回 score/geometry 的 numpy 数组。"""
        score, geometry = self.session.run(["score", "geometry"], {"image": batch})
        return score, geometry


class OpenVINOBackend:
    """从统一检测 ONNX 编译 CPU FP32 模型，按名称取出输出。"""

    def __init__(self, model_path):
        import onnx
        import openvino as ov

        graph = onnx.load(str(model_path), load_external_data=False)
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        self.config = validate_metadata(properties, model_path)

        inputs = graph.graph.input
        if (len(inputs) != 1 or inputs[0].name != "image"
                or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or len(inputs[0].type.tensor_type.shape.dim) != 4
                or inputs[0].type.tensor_type.shape.dim[1].dim_value != 3
                or {value.name for value in graph.graph.output} != {"score", "geometry"}):
            raise ValueError(f"Incompatible unified detection ONNX inputs/outputs: {model_path}")
        spatial = tuple(dimension.dim_value if dimension.dim_value > 0 else None
                        for dimension in inputs[0].type.tensor_type.shape.dim[2:])
        if None not in spatial and spatial != tuple(self.config["input_size"]):
            raise ValueError(f"ONNX input size {spatial} does not match config "
                             f"{self.config['input_size']}: {model_path}")
        self.static_input_size = None if None in spatial else spatial

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(model_path)), "CPU", {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": CPU_THREADS,
            "INFERENCE_PRECISION_HINT": "f32",
        })
        self.request = self.compiled.create_infer_request()

    def infer(self, batch):
        """返回独立的置信度与几何数组，避免下一次请求覆盖输出缓冲区。"""
        self.request.infer({"image": batch})
        return tuple(np.array(self.request.get_tensor(name).data, copy=True)
                     for name in ("score", "geometry"))


def load_backend(model_path, backend):
    """按显式后端加载 ONNX；auto 固定为 ONNX Runtime。"""
    if Path(model_path).suffix.lower() != ".onnx":
        raise ValueError(f"ONNX 推理入口只接受 .onnx 模型: {model_path}")
    if backend == "auto":
        backend = "onnx"
    if backend == "onnx":
        print("  ✓ 后端: ONNX Runtime")
        return OnnxRuntimeBackend(model_path)
    if backend == "openvino":
        print("  ✓ 后端: OpenVINO")
        return OpenVINOBackend(model_path)
    raise ValueError(f"未知后端: {backend}")


def exported_shape(backend):
    """返回 (是否动态高宽, 输入尺寸上限, 固定输入尺寸)。"""
    config = backend.config
    static_size = backend.static_input_size
    dynamic = static_size is None
    if bool(config["dynamic_spatial"]) != dynamic:
        raise ValueError(f"ONNX dynamic_spatial={config['dynamic_spatial']} "
                         f"does not match exported shape {static_size}")
    if dynamic:
        size = tuple(int(value) for value in config["input_size"])
        return True, size, None
    size = tuple(int(value) for value in static_size)
    return False, size, size


def warmup_backend(backend, dynamic, max_input_size, fixed_input_size, count):
    """按模型实际输入方式预热；动态模型额外预热一个宽高互换的形状。"""
    if count < 0:
        raise ValueError("warmup count must be nonnegative")
    sizes = {fixed_input_size if not dynamic else max_input_size}
    if dynamic:
        height, width = max_input_size
        sizes.add((height, max(64, (width // 2) // 32 * 32)))
    for height, width in sizes:
        sample = np.zeros((1, 3, height, width), dtype=np.float32)
        for _ in range(count):
            backend.infer(sample)


def predict_image(backend, image, threshold, iou_threshold, dynamic,
                  max_input_size, fixed_input_size):
    """检测单张 RGB 图片，返回原图坐标预测列表和前向毫秒数。"""
    target_size = dynamic_input_size(image.size, max_input_size) if dynamic else fixed_input_size
    tensor, metadata = prepare_image(image, target_size)
    started = time.perf_counter()
    score, geometry = backend.infer(tensor[None])
    infer_ms = (time.perf_counter() - started) * 1000

    boxes, scores = decode_maps(score[0], geometry[0], threshold)
    boxes = restore_boxes(boxes, metadata, image.size)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    keep = suppress_boxes(boxes, scores, iou_threshold)
    keep = sorted(keep, key=lambda index: (boxes[index, 1], boxes[index, 0]))
    return [{"box": boxes[index].tolist(), "score": float(scores[index])}
            for index in keep], infer_ms


def collect_images(image_path):
    """收集单张图片或目录内的支持图片；同名不同扩展名会覆盖标签，直接拒绝。"""
    image_path = Path(image_path)
    if image_path.is_dir():
        paths = sorted(path for path in image_path.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = [image_path]
    if not paths:
        raise ValueError(f"没有可推理的图片: {image_path}")
    stems = {}
    for path in paths:
        if path.stem in stems:
            raise ValueError(f"同名图片会覆盖标签: {stems[path.stem]} and {path}")
        stems[path.stem] = path
    return paths


def run_onnx_inference(backend, image_path, output_dir, *, dynamic, max_input_size,
                       fixed_input_size, threshold=SCORE_THRESHOLD,
                       iou_threshold=NMS_IOU_THRESHOLD, save_crops=False,
                       crop_dir=None, warmup_runs=0,
                       expand_ratio_w=0.0, expand_ratio_h=0.0):
    """批量保存画框图、bbox4 标签和可选裁片，并打印每图和汇总耗时。"""
    if save_crops and crop_dir is None:
        raise ValueError("crop_dir is required when save_crops=True")
    if warmup_runs:
        warmup_backend(backend, dynamic, max_input_size, fixed_input_size, warmup_runs)

    paths = collect_images(image_path)
    output_dir = Path(output_dir)
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    if save_crops:
        Path(crop_dir).mkdir(parents=True, exist_ok=True)

    total_boxes, total_infer_ms, total_detect_ms = 0, 0.0, 0.0
    wall_started = time.perf_counter()
    for path in paths:
        image = load_rgb_image(path)
        detect_started = time.perf_counter()
        predictions, infer_ms = predict_image(
            backend, image, threshold, iou_threshold, dynamic, max_input_size, fixed_input_size)
        detect_ms = (time.perf_counter() - detect_started) * 1000
        total_boxes += len(predictions)
        total_infer_ms += infer_ms
        total_detect_ms += detect_ms

        predictions = expand_predictions(predictions, image.size, expand_ratio_w, expand_ratio_h)
        visual = image.copy()
        draw = ImageDraw.Draw(visual)
        for index, prediction in enumerate(predictions):
            box = prediction["box"]
            draw.rectangle(box, outline="red", width=2)
            if save_crops:
                crop_box = (int(np.floor(box[0])), int(np.floor(box[1])),
                            int(np.ceil(box[2])), int(np.ceil(box[3])))
                image.crop(crop_box).save(crop_path(crop_dir, path.stem, index))
        if save_crops:
            remove_stale_crops(crop_dir, path.stem, len(predictions))

        visual.save(images_dir / path.name)
        # bbox4 使用原图像素坐标，不附置信度或伪造文字；无检测框时写空标签。
        label_text = "".join(" ".join(str(float(value)) for value in prediction["box"]) + "\n"
                             for prediction in predictions)
        (labels_dir / f"gt_{path.stem}.txt").write_text(label_text, encoding="utf-8")
        print(f"{path}: {len(predictions)} boxes, 前向 {infer_ms:.1f} ms, "
              f"检测 {detect_ms:.1f} ms", flush=True)

    wall_seconds = time.perf_counter() - wall_started
    summary = {"images": len(paths), "boxes": total_boxes,
               "mean_infer_ms": total_infer_ms / len(paths),
               "mean_detect_ms": total_detect_ms / len(paths),
               "wall_seconds": wall_seconds, "images_per_second": len(paths) / wall_seconds}
    print(f"完成: {len(paths)} 张, {total_boxes} 框; "
          f"前向均值 {summary['mean_infer_ms']:.1f} ms; "
          f"检测均值 {summary['mean_detect_ms']:.1f} ms; "
          f"总耗时 {wall_seconds:.2f} s; "
          f"吞吐量 {summary['images_per_second']:.2f} 张/s; 输出: {output_dir}")
    return summary


def build_parser():
    """构造命令行参数；默认值全部来自文件顶部常量。"""
    parser = argparse.ArgumentParser(
        description="统一检测 ONNX CPU 推理：ONNX Runtime 或 OpenVINO。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", type=Path, default=MODEL_PATH,
                        help="已导出的统一检测 .onnx 模型")
    parser.add_argument("--image_path", type=Path, default=IMAGE_PATH,
                        help="输入图片文件或目录")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help="画框图、labels 和裁片输出目录")
    parser.add_argument("--backend", choices=("auto", "onnx", "openvino"),
                        default=BACKEND, help="auto 等价 onnx；openvino 需显式选择")
    parser.add_argument("--score_thr", type=float, default=SCORE_THRESHOLD,
                        help="文字置信度阈值，范围 (0, 1)")
    parser.add_argument("--nms_iou", type=float, default=NMS_IOU_THRESHOLD,
                        help="重复框抑制交并比，范围 (0, 1]")
    parser.add_argument("--save_crops", action="store_true", default=SAVE_CROPS,
                        help="保存原图文字框裁片")
    parser.add_argument("--crop_dir", type=Path, default=CROP_DIR,
                        help="裁片输出目录；仅 --save_crops 时使用")
    parser.add_argument("--warmup_runs", type=int, default=WARMUP_RUNS,
                        help="正式计时前的预热次数，0 表示关闭")
    return parser


def validate_args(parser, args):
    """在加载模型前拒绝明显无效的路径和阈值，避免跑到一半才失败。"""
    if not 0 < args.score_thr < 1:
        parser.error("--score_thr 必须在 (0, 1) 内")
    if not 0 < args.nms_iou <= 1:
        parser.error("--nms_iou 必须在 (0, 1] 内")
    if args.warmup_runs < 0:
        parser.error("--warmup_runs 必须是非负整数")
    if not args.model_path.is_file():
        parser.error(f"模型文件不存在: {args.model_path}")
    if args.model_path.suffix.lower() != ".onnx":
        parser.error(f"ONNX 推理入口只接受 .onnx 模型: {args.model_path}")
    if not args.image_path.exists():
        parser.error(f"图片路径不存在: {args.image_path}")
    if args.image_path.is_file() and args.image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f"不支持的图片扩展名: {args.image_path}")
    if args.image_path.is_dir() and not any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in args.image_path.iterdir()):
        parser.error(f"目录中没有支持的图片: {args.image_path}")


def main(argv=None):
    """读取命令行执行 ONNX CPU 推理；输入尺寸来自模型元数据。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    print("=" * 60)
    print("统一检测 ONNX CPU 推理")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  输入: {args.image_path}")
    print(f"  输出: {args.output_dir}")
    print(f"  后端: {args.backend}")
    print(f"  阈值: score={args.score_thr}, nms_iou={args.nms_iou}")
    print(f"  裁片: {args.crop_dir if args.save_crops else '关闭'}")
    print(f"  预热: {args.warmup_runs} 次")
    print()

    try:
        backend = load_backend(args.model_path, args.backend)
        dynamic, max_input_size, fixed_input_size = exported_shape(backend)
    except (ImportError, ValueError) as error:
        parser.error(str(error))
        return

    print(f"  输入尺寸上限: {max_input_size[1]}×{max_input_size[0]} (宽×高)，"
          f"动态高宽: {dynamic}")
    print()
    run_onnx_inference(
        backend, args.image_path, args.output_dir,
        dynamic=dynamic, max_input_size=max_input_size,
        fixed_input_size=fixed_input_size, threshold=args.score_thr,
        iou_threshold=args.nms_iou, save_crops=args.save_crops,
        crop_dir=args.crop_dir, warmup_runs=args.warmup_runs,
        expand_ratio_w=EXPAND_RATIO_W if EXPAND_BOX else 0.0,
        expand_ratio_h=EXPAND_RATIO_H if EXPAND_BOX else 0.0)


if __name__ == "__main__":
    main()
