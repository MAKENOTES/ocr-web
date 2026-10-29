"""统一检测的 30 种按框安全数据增强与批次执行策略。

增强分为两个执行段，避免在大尺寸 PDF 页面上重复做昂贵的像素操作：

- 原图几何段：随机裁剪、随机补边、随机宽高补边、轻微各向异性缩放；
- 固定输入像素段：亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、
  色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、
  JPEG 压缩、低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色，
  以及避开文字框的随机擦除、随机表格线和随机斑点。

几何段在 `ResizeAndPad` 之前执行，像素段在缩放补边后的 640 输入上执行。
所有操作接收 RGB `uint8` 数组和 `(N,4,2)` 水平四边形；几何操作后统一裁剪
坐标并删除退化框，保证标签不越界。旋转、透视和剪切会破坏水平框契约，本文件
不提供这些操作。

批次执行方案由 `Augmenter` 和 `policy_weights_for_epoch` 共同实现：每个样本
先从当前阶段权重采样 light/standard/strong，再按该阶段的几何/像素操作数和概率
采样若干增强；`train.py` 在课程阶段切换时重建训练 DataLoader，使同一批次混合
不同强度，并随训练轮次从轻增强逐步过渡到强增强。验证集不调用本文件。
"""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image


# 增强阶段：按几何、像素两个执行段分别设置操作数，scale 控制概率和幅度。
# 几何段在原图上执行；像素段在缩放补边后的固定输入上执行，避免大图重复模糊/压缩。
POLICIES = {
    "light": {"geometry": (0, 1), "pixel": (2, 3), "scale": 0.55},  # 轻增强：几何 0-1 种、像素 2-3 种
    "standard": {"geometry": (0, 2), "pixel": (3, 4), "scale": 1.0},  # 标准：几何 0-2 种、像素 3-4 种
    "strong": {"geometry": (1, 2), "pixel": (4, 6), "scale": 1.3},  # 强增强：几何 1-2 种、像素 4-6 种
}

# 分类执行顺序：几何先改变坐标和画布，光度再改变外观，遮挡最后加入干扰。
CATEGORY_ORDER = {"geometry": 0, "photometric": 1, "occlusion": 2}

# 轮次课程：按训练进度（0~1）设置 light/standard/strong 的采样权重。
CURRICULUM = (
    (0.15, {"light": 0.55, "standard": 0.40, "strong": 0.05}),  # 前 15%：轻增强为主
    (0.70, {"light": 0.20, "standard": 0.60, "strong": 0.20}),  # 15%-70%：标准增强为主
    (1.01, {"light": 0.05, "standard": 0.45, "strong": 0.50}),  # 后 30%：强增强为主
)

# 默认阶段权重：直接调用 get_transforms(is_train=True) 时使用标准增强。
DEFAULT_STAGE_WEIGHTS = {"standard": 1.0}


def _rng():
    """从 torch 的工作进程随机状态派生 numpy 生成器，保证各 worker 不同。"""
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))


def _blend(original, target, alpha):
    """按 alpha 混合两张 uint8 RGB 图像，结果保持 uint8。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.clip(original.astype(np.float32) * (1 - alpha)
                   + target.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _border_color(image):
    """用四周边缘像素中位数估计背景色，供补边、擦除和表格线使用。"""
    height, width = image.shape[:2]
    border = np.concatenate((
        image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]), axis=0)
    return tuple(int(value) for value in np.median(border, axis=0))


def _odd(value):
    """返回不小于 value 的奇数，供 OpenCV 模糊核使用。"""
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _sanitize(quads, ignored, width, height):
    """把四边形裁到画布内并删除宽或高不超过 1 像素的退化框。"""
    quads = np.asarray(quads, dtype=np.float32)
    ignored = np.asarray(ignored, dtype=bool)
    if quads.size == 0:
        return np.empty((0, 4, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    clipped = quads.copy()
    clipped[..., 0] = np.clip(clipped[..., 0], 0, width)
    clipped[..., 1] = np.clip(clipped[..., 1], 0, height)
    box_width = clipped[:, :, 0].max(axis=1) - clipped[:, :, 0].min(axis=1)
    box_height = clipped[:, :, 1].max(axis=1) - clipped[:, :, 1].min(axis=1)
    valid = (box_width > 1) & (box_height > 1)
    return clipped[valid], ignored[valid]


# 光度增强：只改变像素，不改变四边形。


def _brightness(image, quads, ignored, rng, strength):
    factor = 1.0 + rng.uniform(-0.28, 0.28) * strength
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8), quads, ignored


def _contrast(image, quads, ignored, rng, strength):
    mean = float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
    factor = 1.0 + rng.uniform(-0.28, 0.28) * strength
    output = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(output, 0, 255).astype(np.uint8), quads, ignored


def _saturation(image, quads, ignored, rng, strength):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-0.35, 0.35) * strength), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), quads, ignored


def _hue(image, quads, ignored, rng, strength):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    shift = int(rng.uniform(-9, 9) * strength) % 180
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), quads, ignored


def _gamma(image, quads, ignored, rng, strength):
    gamma = float(np.exp(rng.uniform(-0.35, 0.35) * strength))
    lookup = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)],
                      dtype=np.uint8)
    return cv2.LUT(image, lookup), quads, ignored


def _autocontrast(image, quads, ignored, rng, strength):
    flat = image.reshape(-1, 3)
    low = np.percentile(flat, 1, axis=0)
    high = np.percentile(flat, 99, axis=0)
    if np.any(high - low < 1):
        return image, quads, ignored
    stretched = np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return _blend(image, stretched, rng.uniform(0.3, 0.9)), quads, ignored


def _equalize(image, quads, ignored, rng, strength):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    equalized = cv2.cvtColor(cv2.merge([cv2.equalizeHist(lightness), a_channel, b_channel]),
                            cv2.COLOR_LAB2RGB)
    return _blend(image, equalized, rng.uniform(0.3, 0.9)), quads, ignored


def _posterize(image, quads, ignored, rng, strength):
    levels = int(rng.integers(4, 10))
    step = 256 // levels
    return ((image // step) * step).astype(np.uint8), quads, ignored


def _solarize(image, quads, ignored, rng, strength):
    threshold = int(rng.integers(180, 250))
    return np.where(image >= threshold, 255 - image, image).astype(np.uint8), quads, ignored


def _sharpness(image, quads, ignored, rng, strength):
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    amount = rng.uniform(0.15, 0.7) * strength
    output = image.astype(np.float32) + amount * (image.astype(np.float32) - blurred)
    return np.clip(output, 0, 255).astype(np.uint8), quads, ignored


def _gaussian_blur(image, quads, ignored, rng, strength):
    sigma = rng.uniform(0.4, 1.6) * strength
    size = _odd(max(3, int(sigma * 4 + 1)))
    return cv2.GaussianBlur(image, (size, size), sigma), quads, ignored


def _motion_blur(image, quads, ignored, rng, strength):
    length = int(rng.integers(3, 12))
    angle = float(rng.uniform(0, 180))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = float(kernel.sum())
    if total <= 0:
        return image, quads, ignored
    return cv2.filter2D(image, -1, kernel / total), quads, ignored


def _median_blur(image, quads, ignored, rng, strength):
    return cv2.medianBlur(image, int(rng.choice([3, 5]))), quads, ignored


def _gaussian_noise(image, quads, ignored, rng, strength):
    sigma = rng.uniform(2, 12) * strength
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), quads, ignored


def _salt_pepper(image, quads, ignored, rng, strength):
    output = image.copy()
    mask = rng.random(image.shape[:2]) < rng.uniform(0.0005, 0.004) * strength
    values = rng.integers(0, 2, int(mask.sum())) * 255
    output[mask] = values[:, None]
    return output, quads, ignored


def _jpeg_compression(image, quads, ignored, rng, strength):
    quality = int(np.clip(rng.integers(35, 91) - 10 * (strength - 1), 25, 95))
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return image, quads, ignored
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return image, quads, ignored
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), quads, ignored


def _downscale(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    low = max(0.35, 0.9 - 0.4 * strength)
    scale = float(rng.uniform(low, 0.95))
    small = (max(1, int(width * scale)), max(1, int(height * scale)))
    reduced = cv2.resize(image, small, interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR), quads, ignored


def _clahe(image, quads, ignored, rng, strength):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    target = cv2.cvtColor(cv2.merge([enhanced, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    return _blend(image, target, rng.uniform(0.25, 0.8)), quads, ignored


def _channel_gain(image, quads, ignored, rng, strength):
    gains = 1 + rng.uniform(-0.15, 0.15, 3) * strength
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8), quads, ignored


def _shadow_gradient(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    dark = float(rng.uniform(0.12, 0.40) * strength)
    direction = int(rng.integers(0, 4))
    if direction == 0:
        ramp = np.linspace(1, 1 - dark, width)[None, :, None]
    elif direction == 1:
        ramp = np.linspace(1 - dark, 1, width)[None, :, None]
    elif direction == 2:
        ramp = np.linspace(1, 1 - dark, height)[:, None, None]
    else:
        ramp = np.linspace(1 - dark, 1, height)[:, None, None]
    return np.clip(image.astype(np.float32) * ramp, 0, 255).astype(np.uint8), quads, ignored


def _background_tint(image, quads, ignored, rng, strength):
    color = rng.integers(0, 256, 3).astype(np.float32)
    alpha = float(rng.uniform(0.02, 0.10) * strength)
    output = image.astype(np.float32) * (1 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8), quads, ignored


def _vignette(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    distance = np.sqrt(((rows - height / 2) / max(height, 1)) ** 2
                       + ((columns - width / 2) / max(width, 1)) ** 2)
    mask = np.clip(1 - rng.uniform(0.12, 0.30) * strength * distance, 0.4, 1.0)
    return np.clip(image.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8), quads, ignored


def _invert(image, quads, ignored, rng, strength):
    return 255 - image, quads, ignored


# 几何增强：改变画布或坐标，调用方在操作后统一做 _sanitize。


def _random_crop(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    if height < 16 or width < 16:
        return image, quads, ignored
    minimum = float(np.clip(1 - 0.35 * strength, 0.55, 0.9))
    crop_width = max(2, min(width, int(width * rng.uniform(minimum, 0.98))))
    crop_height = max(2, min(height, int(height * rng.uniform(minimum, 0.98))))
    left = int(rng.integers(0, width - crop_width + 1))
    top = int(rng.integers(0, height - crop_height + 1))
    output = image[top:top + crop_height, left:left + crop_width].copy()
    if quads.size:
        quads = quads.copy()
        quads[..., 0] -= left
        quads[..., 1] -= top
    return output, quads, ignored


def _random_pad(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    base = max(1, min(height, width))
    limit = int(base * 0.12 * strength)
    top, bottom, left, right = (int(rng.integers(0, limit + 1)) for _ in range(4))
    if not (top or bottom or left or right):
        return image, quads, ignored
    output = np.empty((height + top + bottom, width + left + right, 3), dtype=np.uint8)
    output[:, :] = _border_color(image)
    output[top:top + height, left:left + width] = image
    if quads.size:
        quads = quads.copy()
        quads[..., 0] += left
        quads[..., 1] += top
    return output, quads, ignored


def _random_aspect_pad(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    if rng.random() < 0.5:
        left = right = int(width * rng.uniform(0.03, 0.25 * strength))
        top = bottom = 0
    else:
        top = bottom = int(height * rng.uniform(0.03, 0.25 * strength))
        left = right = 0
    if not (top or bottom or left or right):
        return image, quads, ignored
    output = np.empty((height + top + bottom, width + left + right, 3), dtype=np.uint8)
    output[:, :] = _border_color(image)
    output[top:top + height, left:left + width] = image
    if quads.size:
        quads = quads.copy()
        quads[..., 0] += left
        quads[..., 1] += top
    return output, quads, ignored


def _random_anisotropic_scale(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    scale_x = 1 + rng.uniform(-0.08, 0.08) * strength
    scale_y = 1 + rng.uniform(-0.08, 0.08) * strength
    new_width = max(2, int(round(width * scale_x)))
    new_height = max(2, int(round(height * scale_y)))
    output = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    if quads.size:
        quads = quads.copy()
        quads[..., 0] *= new_width / width
        quads[..., 1] *= new_height / height
    return output, quads, ignored


# 遮挡增强：不改变坐标，只加入局部干扰。


def _random_erase(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    text_mask = np.zeros((height, width), dtype=np.uint8)
    for quad in quads:
        x0, x1 = int(np.floor(quad[:, 0].min())), int(np.ceil(quad[:, 0].max()))
        y0, y1 = int(np.floor(quad[:, 1].min())), int(np.ceil(quad[:, 1].max()))
        text_mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 1

    output = image.copy()
    color = _border_color(image)
    for _ in range(int(rng.integers(1, 4))):
        rect_width = max(2, int(width * rng.uniform(0.02, 0.08) * strength))
        rect_height = max(2, int(height * rng.uniform(0.02, 0.08) * strength))
        left = int(rng.integers(0, max(1, width - rect_width + 1)))
        top = int(rng.integers(0, max(1, height - rect_height + 1)))
        region = text_mask[top:top + rect_height, left:left + rect_width]
        # 只在未覆盖文字的区域擦除，避免把检测标签变成错误负样本。
        if region.size and region.mean() > 0.25:
            continue
        output[top:top + rect_height, left:left + rect_width] = color
    return output, quads, ignored


def _random_lines(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    output = image.copy()
    for _ in range(int(rng.integers(1, 5))):
        overlay = output.copy()
        if rng.random() < 0.5:
            position = int(rng.integers(0, height))
            start, end = (0, position), (max(0, width - 1), position)
        else:
            position = int(rng.integers(0, width))
            start, end = (position, 0), (position, max(0, height - 1))
        color = (_border_color(image) if rng.random() < 0.5
                 else (int(rng.integers(0, 256)),) * 3)
        cv2.line(overlay, start, end, color, int(rng.integers(1, 3)))
        output = _blend(output, overlay, rng.uniform(0.25, 0.75))
    return output, quads, ignored


def _random_speckles(image, quads, ignored, rng, strength):
    height, width = image.shape[:2]
    output = image.copy()
    for _ in range(int(rng.integers(8, 40) * strength)):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(output, center, int(rng.integers(1, 3)), int(rng.integers(0, 256)), -1)
    return output, quads, ignored


@dataclass(frozen=True)
class AugOp:
    """一个带类别和基础触发概率的增强操作。"""
    name: str
    category: str
    probability: float
    function: Callable


# 30 种增强的注册表；概率是 standard 阶段的单图触发概率，light/strong 会再乘 scale。
AUGMENTATIONS = (
    AugOp("brightness", "photometric", 0.35, _brightness),
    AugOp("contrast", "photometric", 0.35, _contrast),
    AugOp("saturation", "photometric", 0.25, _saturation),
    AugOp("hue", "photometric", 0.15, _hue),
    AugOp("gamma", "photometric", 0.20, _gamma),
    AugOp("autocontrast", "photometric", 0.15, _autocontrast),
    AugOp("equalize", "photometric", 0.10, _equalize),
    AugOp("posterize", "photometric", 0.08, _posterize),
    AugOp("solarize", "photometric", 0.05, _solarize),
    AugOp("sharpness", "photometric", 0.15, _sharpness),
    AugOp("gaussian_blur", "photometric", 0.12, _gaussian_blur),
    AugOp("motion_blur", "photometric", 0.06, _motion_blur),
    AugOp("median_blur", "photometric", 0.05, _median_blur),
    AugOp("gaussian_noise", "photometric", 0.12, _gaussian_noise),
    AugOp("salt_pepper", "photometric", 0.08, _salt_pepper),
    AugOp("jpeg_compression", "photometric", 0.10, _jpeg_compression),
    AugOp("downscale", "photometric", 0.10, _downscale),
    AugOp("clahe", "photometric", 0.08, _clahe),
    AugOp("channel_gain", "photometric", 0.15, _channel_gain),
    AugOp("shadow_gradient", "photometric", 0.10, _shadow_gradient),
    AugOp("background_tint", "photometric", 0.12, _background_tint),
    AugOp("vignette", "photometric", 0.08, _vignette),
    AugOp("invert", "photometric", 0.02, _invert),
    AugOp("random_crop", "geometry", 0.20, _random_crop),
    AugOp("random_pad", "geometry", 0.20, _random_pad),
    AugOp("random_aspect_pad", "geometry", 0.12, _random_aspect_pad),
    AugOp("random_anisotropic_scale", "geometry", 0.08, _random_anisotropic_scale),
    AugOp("random_erase", "occlusion", 0.10, _random_erase),
    AugOp("random_lines", "occlusion", 0.08, _random_lines),
    AugOp("random_speckles", "occlusion", 0.08, _random_speckles),
)


def normalize_stage_weights(stage_weights=None):
    """校验并归一化 light/standard/strong 采样权重。"""
    if stage_weights is None:
        stage_weights = DEFAULT_STAGE_WEIGHTS
    if not isinstance(stage_weights, dict) or not stage_weights:
        raise ValueError("stage_weights must be a non-empty dict")
    unknown = set(stage_weights) - set(POLICIES)
    if unknown:
        raise ValueError(f"Unknown augmentation stages: {sorted(unknown)}")
    values = np.asarray(list(stage_weights.values()), dtype=np.float64)
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("stage_weights must be nonnegative and contain a positive value")
    return {name: float(value / values.sum()) for name, value in stage_weights.items()}


def policy_weights_for_epoch(epoch, epochs):
    """按训练进度返回三个阶段的采样权重，实现从轻到强的课程。"""
    if epochs < 1 or epoch < 1:
        raise ValueError("epoch and epochs must be positive")
    progress = (min(epoch, epochs) - 1) / max(1, epochs - 1)
    for boundary, weights in CURRICULUM:
        if progress < boundary:
            return dict(weights)
    return dict(CURRICULUM[-1][1])


class Augmenter:
    """按阶段权重对单个样本采样并执行增强，返回 PIL 样本。

    categories 只能是 {"geometry"} 或 {"photometric","occlusion"}：
    前者在原图上改变画布和坐标，后者在缩放补边后的固定输入上改变外观。
    """

    def __init__(self, stage_weights=None, categories=("geometry",)):
        self.stage_weights = normalize_stage_weights(stage_weights)
        self.categories = set(categories)
        if self.categories == {"geometry"}:
            self.policy_key = "geometry"
        elif self.categories and self.categories <= {"photometric", "occlusion"}:
            self.policy_key = "pixel"
        else:
            raise ValueError("Augmenter categories must be {'geometry'} "
                             "or a subset of {'photometric', 'occlusion'}")
        self.operations = [operation for operation in AUGMENTATIONS
                           if operation.category in self.categories]

    def _select_ops(self, rng, min_ops, max_ops, scale):
        """按概率选择操作，保证数量在 min_ops 到 max_ops 之间。"""
        selected = [operation for operation in self.operations
                    if rng.random() < min(1.0, operation.probability * scale)]
        if len(selected) > max_ops:
            indices = rng.choice(len(selected), size=max_ops, replace=False)
            selected = [selected[int(index)] for index in indices]
        if len(selected) < min_ops:
            chosen = {operation.name for operation in selected}
            remaining = [operation for operation in self.operations
                         if operation.name not in chosen]
            for index in rng.permutation(len(remaining)):
                if len(selected) >= min_ops:
                    break
                selected.append(remaining[int(index)])
        selected.sort(key=lambda operation: CATEGORY_ORDER[operation.category])
        return selected

    def __call__(self, sample):
        rng = _rng()
        stage = sample.get("aug_stage")
        if stage is None:
            stages = list(self.stage_weights)
            probabilities = list(self.stage_weights.values())
            stage = stages[int(rng.choice(len(stages), p=probabilities))]
            # 同一张图片的几何段和像素段共用一次阶段采样。
            sample["aug_stage"] = stage
        policy = POLICIES[stage]
        min_ops, max_ops = policy[self.policy_key]

        operations = self._select_ops(rng, min_ops, max_ops, policy["scale"])
        image = np.asarray(sample["image"].convert("RGB"), dtype=np.uint8).copy()
        quads = np.asarray(sample["quads"], dtype=np.float32).copy()
        ignored = np.asarray(sample["ignored"], dtype=bool).copy()

        for operation in operations:
            image, quads, ignored = operation.function(image, quads, ignored, rng, policy["scale"])
            if operation.category == "geometry":
                quads, ignored = _sanitize(quads, ignored, image.shape[1], image.shape[0])

        sample["image"] = Image.fromarray(image, "RGB")
        sample["quads"] = quads
        sample["ignored"] = ignored
        if self.policy_key == "pixel":
            # 像素段是训练管线的最后一段，清理临时阶段标记。
            sample.pop("aug_stage", None)
        return sample


def get_geometry_augmenter(stage_weights=None):
    """构建原图阶段的几何增强器；训练时在 ResizeAndPad 之前调用。"""
    return Augmenter(stage_weights, {"geometry"})


def get_pixel_augmenter(stage_weights=None):
    """构建固定输入阶段的光度/遮挡增强器；训练时在 ResizeAndPad 之后调用。"""
    return Augmenter(stage_weights, {"photometric", "occlusion"})
