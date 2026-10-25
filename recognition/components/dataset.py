"""识别图片加载、局部去章、缩放、训练增强与动态补边。

TextRecognitionDataset 读取 Tab 标注；InferenceDataset 读取图片列表。
两者共享基础预处理，训练另加随机增强；collate 返回图片、标签或路径、真实宽度。"""
import os
import random
from collections.abc import Sequence
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps, ImageStat
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from recognition.components.geometry import RandomSourceGeometry, border_fill_color
from recognition.utils.annotation import split_annotation_fields

# 允许 Pillow 加载部分截断图片；这是进程级开关，不保证损坏图片完整可用。
ImageFile.LOAD_TRUNCATED_IMAGES = True

MIN_IMAGE_WIDTH = 48  # 缩放后窄图的背景画布宽度，单位 px；显式 max_width 更小时以上限为准。
# MobileFeatureExtractor 的横向感受野为 53 px。每张图右侧至少保留 32 px
# 背景，使有效 CNN 特征不会因 batch 中最长图片不同而看到不同的张量边界。
CNN_RIGHT_CONTEXT = 32
# 训练 DataLoader 的随机基准种子。它只决定各 worker 的随机增强序列，与模型初始化和
# 主进程中的分桶打乱无关；固定后同一份代码和配置可以复现每个 worker 的增强。
DATALOADER_SEED = 20260923

# 去章时按平均正文列密度缩放的正文支持下限。细字和末行密度偏低，0.4 会把正常叠印
# 误判为无正文支持，0.35 在保持"红字旁边没有黑字就不处理"的前提下放行这类样本。
MIN_DARK_COLUMN_SUPPORT = 0.35
# 完全落在正文行内的红色区域所需的正文列支持。同行红字通常位于字间空白，支持接近 0；
# 被黑字穿过的章字支持很高，因此该阈值明显高于行外的 0.15，只放行有充分正文证据的碎片。
IN_BAND_DARK_SUPPORT = 0.4
# 行内红色碎片判定为章字残段的最大高度，按正文行高计。完整红字的高度接近行高，
# 被正文行切开的章字残段则明显更矮，两者由此区分，避免把同行红字一并删掉。
IN_BAND_FRAGMENT_HEIGHT = 0.35


def resolve_image_path(image_path: str | os.PathLike) -> str:
    """返回实际存在的图片路径；标注中的文件名和扩展名必须准确。"""
    path = Path(image_path)
    if path.is_file():
        return str(path)

    raise FileNotFoundError(f"图片不存在: {path}")


def load_rgb_image(image_path: str | os.PathLike) -> Image.Image:
    """读取图片、应用 EXIF 方向并合成为 RGB。

    有透明通道时，按可见前景亮度选择黑或白背景，避免直接丢弃 alpha。
    完全透明图使用白底；混合明暗前景仍需人工核对可见性，不保证恢复被遮挡内容。"""
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

        background = Image.new(
            "RGBA",
            rgba.size,
            (background_value, background_value, background_value, 255),
        )
        return Image.alpha_composite(background, rgba).convert("RGB")


def _is_red_seal_arc(component: np.ndarray, text_height: float) -> bool:
    """识别 detection 紧框内的连续弯曲章边；直线、短笔画和杂乱红字不通过。"""
    for oriented in (component, component.T):
        columns = np.flatnonzero(oriented.any(axis=0))
        span = columns[-1] - columns[0] + 1
        if span < 2 * text_height or columns.size < 0.9 * span:
            continue
        thickness = oriented[:, columns].sum(axis=0)
        if np.quantile(thickness, 0.9) > 0.6 * text_height:
            continue
        centerline = np.array([
            np.median(np.flatnonzero(oriented[:, column])) for column in columns
        ])
        # 局部圆弧用二次曲线近似；要求明显曲率，排除红色下划线/表格边框。
        coefficients = np.polyfit(columns, centerline, 2)
        if abs(coefficients[0]) * (span - 1) ** 2 < 0.5 * text_height:
            continue
        residual = np.abs(np.polyval(coefficients, columns) - centerline)
        if np.quantile(residual, 0.9) <= max(1.5, 0.12 * text_height):
            return True
    return False


def _is_single_seal_component(
    component: np.ndarray, text_height: float, *, allow_narrow: bool = False,
) -> bool:
    """单个连通区域需有弧线、椭圆或复杂章形，避免把直线/简单边框当章。

    allow_narrow 只放宽宽度判定，供"比行高还窄的组"使用：这类组按定义宽度不足，
    但高度足够、且轮廓成弧的是被正文行切开的章边。"""
    height, width = component.shape
    if height < 0.8 * text_height or (not allow_narrow and width < 0.8 * text_height):
        return False
    if _is_red_seal_arc(component, text_height):
        return True
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 12:
        return False
    # 星形、连在一起的章字无需拟合成圆；矩形框和简单折线继续保留。
    polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(polygon) >= 9:
        return True
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    if (
        not np.isfinite([cx, cy, axis_a, axis_b, angle]).all()
        or min(axis_a, axis_b) < text_height
        or max(axis_a, axis_b) > 3 * min(axis_a, axis_b)
        or component.mean() > 0.45
    ):
        return False
    points = contour[:, 0, :].astype(np.float64) - (cx, cy)
    radians = np.deg2rad(angle)
    u = points[:, 0] * np.cos(radians) + points[:, 1] * np.sin(radians)
    v = -points[:, 0] * np.sin(radians) + points[:, 1] * np.cos(radians)
    radius = np.sqrt((2 * u / axis_a) ** 2 + (2 * v / axis_b) ** 2)
    return bool(np.quantile(np.abs(radius - 1), 0.9) <= 0.12)


def remove_red_seal(image: Image.Image) -> Image.Image:
    """去除浅色纸面上与深色正文交错叠印的红章，包括被紧框截断的章字。

    按局部红色区域检查正文穿插和上下错位，不再要求每个碎片都有完整章轮廓。
    普通同行红字所在列通常没有黑色正文，或与正文行高对齐，因此保留。
    区域通过后仍逐个检查连通区域的正文支持，避免连带删除附近合法红字。
    只替换候选区域内的红像素，不膨胀掩码，以保护相邻的深色笔画。
    输入必须是 load_rgb_image 已完成透明背景合成的 RGB 图片。
    """
    if image.mode != "RGB":
        raise ValueError(f"remove_red_seal requires RGB input, got {image.mode}")

    pixels = np.asarray(image)
    # 先转有符号整数再求色差，避免 uint8 减法溢出把蓝字误判成红色。
    rgb = pixels.astype(np.int16)
    red, green, blue = rgb.transpose(2, 0, 1)
    red_mask = (
        (red >= 150) & (red - green >= 25) & (red - blue >= 25)
        & (np.abs(green - blue) <= 40)
    )
    if not red_mask.any() or red_mask.mean() > 0.35:
        return image

    # 兼容浅灰、轻微偏色和带扫描颗粒的纸面；红底白字、暗底仍不处理。
    channel_range = rgb.max(axis=2) - rgb.min(axis=2)
    paper_mask = (rgb.min(axis=2) >= 190) & (channel_range <= 35)
    paper_border = np.concatenate((
        paper_mask[0], paper_mask[-1], paper_mask[:, 0], paper_mask[:, -1],
    ))
    if paper_mask.mean() < 0.4 or paper_border.mean() < 0.5:
        return image

    # 灰黑正文也能提供证据，但红色候选不能同时充当“黑字”，避免自证为章。
    dark_mask = (rgb.max(axis=2) <= 170) & (channel_range <= 35) & ~red_mask
    dark_y = np.nonzero(dark_mask)[0]
    # 大章/星形可能比正文墨迹更多，不再要求黑像素数量超过全部红像素。
    if dark_y.size < max(16, red_mask.sum() * 0.15):
        return image

    # 用正文墨迹的中间 96% 估计行高，减少孤立噪点影响。
    text_top, text_bottom = np.quantile(dark_y, [0.02, 0.98])
    text_height = text_bottom - text_top + 1
    if text_height < 6:
        return image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        red_mask.astype(np.uint8), connectivity=8,
    )
    erase_mask = np.zeros(red_mask.shape, dtype=bool)
    band = slice(int(text_top), int(text_bottom) + 1)
    dark_columns = dark_mask[band].sum(axis=0) >= max(2, 0.08 * text_height)
    # 细字体的有墨迹列本来就少，按正文列密度调整最低支持率。
    # 保留绝对下限，避免几粒黑噪点让整段红字被当作叠印。
    min_overlap = max(0.12, MIN_DARK_COLUMN_SUPPORT * dark_columns.mean())
    red_columns = np.flatnonzero(red_mask.any(axis=0))
    # 章边、章字、中央星形之间常隔着一两个字宽，需作为局部整体判断。
    # 这里只分组，绝不对待删除像素做闭运算/膨胀；组内仍逐个连通区域复核。
    groups = np.split(
        red_columns,
        np.flatnonzero(np.diff(red_columns) > max(2, 2 * text_height)) + 1,
    )
    for columns in groups:
        left, right = columns[0], columns[-1] + 1
        region = red_mask[:, left:right]
        if region.sum() < 16:
            continue
        active_columns = columns[red_mask[:, columns].sum(axis=0) >= 2]
        if active_columns.size == 0 or dark_columns[active_columns].mean() < min_overlap:
            continue
        in_band_ratio = region[band].sum() / region.sum()
        # 同行红字/描边红字与黑字通常对齐；章既进入正文行，又有越界部分。
        # 允许只越过上边或下边，适配只剩半个章的 detection 框。
        if not 0.15 <= in_band_ratio <= 0.92:
            continue
        component_ids = np.unique(labels[:, left:right][region])
        component_ids = [
            int(component_id) for component_id in component_ids
            if stats[component_id, cv2.CC_STAT_AREA] >= max(4, 0.15 * text_height)
        ]
        # 比一个行高还窄的组只放行本身像章的连通区域：穿过单个高字的章弧宽度
        # 常常小于行高，而真正的短红字没有弧线或椭圆轮廓。
        if right - left < text_height and not any(
            _is_single_seal_component(
                labels[
                    stats[component_id, cv2.CC_STAT_TOP]:
                    stats[component_id, cv2.CC_STAT_TOP] + stats[component_id, cv2.CC_STAT_HEIGHT],
                    stats[component_id, cv2.CC_STAT_LEFT]:
                    stats[component_id, cv2.CC_STAT_LEFT] + stats[component_id, cv2.CC_STAT_WIDTH],
                ] == component_id,
                text_height,
                allow_narrow=True,
            )
            for component_id in component_ids
        ):
            continue
        for component_id in component_ids:
            x, y, width, height, _ = stats[component_id]
            component = labels[y:y + height, x:x + width] == component_id
            # 区域内的小红字仍须独立检查，不能因旁边有章就整片抹红。
            component_columns = np.flatnonzero(component.any(axis=0)) + x
            support = dark_columns[component_columns].mean()
            if y >= text_top and y + height - 1 <= text_bottom:
                # 完整红字的高度接近行高；只有明显更矮、且被黑字穿过的碎片才是章字残段。
                if (
                    height >= IN_BAND_FRAGMENT_HEIGHT * text_height
                    or support < IN_BAND_DARK_SUPPORT
                ):
                    continue
            elif len(component_ids) == 1 and not _is_single_seal_component(component, text_height):
                continue
            if support < 0.15:
                continue
            erase_mask[y:y + height, x:x + width] |= component

    if not erase_mask.any():
        return image
    result = pixels.copy()
    # 用纸面中位色替换，避免浅灰底出现纯白斑；不改任何非候选像素。
    result[erase_mask] = np.median(pixels[paper_mask], axis=0).astype(np.uint8)
    return Image.fromarray(result)


def resize_detection_image(
    image: Image.Image,
    target_height: int,
    max_width: int | None = None,
    min_width: int = MIN_IMAGE_WIDTH,
) -> Image.Image:
    """固定高度缩放完整文字框，不裁剪正文。
    优先保持宽高比；超过 max_width 时横向压缩。窄于 min_width 时居中补背景。"""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"非法图片尺寸: {image.size}")
    if target_height <= 0:
        raise ValueError(f"target_height 必须大于 0，当前为 {target_height}")

    resized_width = max(1, int(round(width * target_height / height)))
    if max_width is not None:
        resized_width = min(resized_width, max_width)

    image = image.resize((resized_width, target_height), Image.Resampling.LANCZOS)

    # 单字/符号 detection 框可能很窄。只补背景，不拉伸字符形状。
    if resized_width < min_width:
        canvas_width = min_width if max_width is None else min(min_width, max_width)
        fill = border_fill_color(image)
        canvas = Image.new(image.mode, (canvas_width, target_height), fill)
        offset_x = (canvas_width - resized_width) // 2
        canvas.paste(image, (offset_x, 0))
        image = canvas

    return image


# 1. 按宽高比组批次及动态补边

class BucketBatchSampler(Sampler):
    """
    桶采样器：将长宽比相近的图片放在同一个 Batch 中。
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        print("正在预扫描数据集长宽比以优化 Padding (Bucket Sampling)...")
        self.aspect_ratios = []
        for path in dataset.image_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    h = max(1, h)
                    self.aspect_ratios.append(w / h)
            except Exception as e:
                # 预扫描失败仅影响分桶；实际读取仍由 _load_and_process_image 报错。
                self.aspect_ratios.append(1.0)

        self.aspect_ratios = np.array(self.aspect_ratios)
        self.num_samples = len(self.dataset)
        print(f"扫描完成，样本数: {self.num_samples}")

    def __iter__(self):
        indices = np.argsort(self.aspect_ratios)
        batches = []
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i: i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            batches.append(batch_indices.tolist())

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


def dynamic_padding_collate_fn(batch):
    """
    自定义 collate_fn，实现按批次动态 Padding。
    支持训练模式 (img, label) 和 推理模式 (img, path)。

    返回:
        padded_imgs: [B, C, H, W_max] 填充后的图像张量；每张图右侧至少
                     保留 CNN_RIGHT_CONTEXT 像素的背景保护区
        labels_or_paths: 标签或路径的元组
        img_widths: [B] 每张图片未包含保护区和 batch padding 的真实宽度（像素），
                    用于计算每个样本在 CNN 下采样后的真实时间步数，
                    使 CTC Loss 和解码只关注有效区域，忽略 padding。
    """
    images, labels_or_paths = zip(*batch)

    # 记录每张图片的真实宽度
    img_widths = torch.IntTensor([img.shape[2] for img in images])

    # 最宽图片也必须拥有固定的右侧背景保护区。否则它单独推理时 CNN 会
    # 看到张量边界，与它和更宽图片组成 batch 时提取到的右边缘特征不同。
    max_width = max(img_widths).item() + CNN_RIGHT_CONTEXT

    # 宽度补齐到 8 的倍数
    max_width = ((max_width + 7) // 8) * 8

    num_channels = images[0].shape[0]
    height = images[0].shape[1]
    batch_size = len(images)

    padded_imgs = torch.empty((batch_size, num_channels, height, max_width))

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img
        if w < max_width:
            # detection 框既可能是白底黑字，也可能是深色/彩色底白字。
            # 使用右侧边缘的通道中位数补齐，避免统一白边制造额外颜色边界。
            border_width = min(4, w)
            fill = img[:, :, w - border_width:w].reshape(num_channels, -1).median(dim=1).values
            padded_imgs[i, :, :, w:] = fill[:, None, None]

    return padded_imgs, labels_or_paths, img_widths


# 2. 基础数据集类 (封装通用逻辑)

class AddGaussianNoise(object):
    """模拟扫描颗粒和相机传感器轻噪声；截断到 [0, 1]，避免异常亮暗值。"""

    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return (tensor + noise).clamp(0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class BaseTextDataset(Dataset):
    """
    文本识别数据集基类。
    封装了核心的图片加载、预处理、增强逻辑。
    """

    def __init__(
        self,
        img_height,
        num_channels,
        is_train=False,
        max_width=None,
        source_augment=None,
    ):
        if num_channels not in (1, 3):
            raise ValueError(f"num_channels 仅支持 1 或 3，当前为 {num_channels}")

        self.img_height = img_height
        self.num_channels = num_channels
        self.max_width = max_width
        # 缩放前作用于源分辨率图片的几何扰动。训练默认启用；调用方可传入
        # 确定性扰动做鲁棒性评估，传 None 且 is_train=False 时完全关闭。
        if is_train and source_augment is None:
            source_augment = RandomSourceGeometry()
        self.source_augment = source_augment
        self.image_paths = []  # 子类填充此列表

        channel_transform = (
            # 兼容旧的单通道模型；RGB 模型保留颜色信息。这是通道适配，不是增强。
            [transforms.Grayscale(num_output_channels=1)]
            if num_channels == 1
            else []
        )
        normalize = transforms.Normalize(
            # 将 [0, 1] 映射到 [-1, 1]，统一训练和推理数值范围，不是随机增强。
            mean=[0.5] * num_channels,
            std=[0.5] * num_channels,
        )

        if is_train:
            # 几何扰动在缩放前的源分辨率完成（见 source_augment），这里只做
            # 缩放后的颜色、极性、清晰度和轻噪声变化。
            train_transforms = [
                # 颜色扰动（70%）：模拟曝光、纸面反差、印刷浓淡和轻微偏色。
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.2,  # 明暗变化：不同光照和扫描曝光。
                        contrast=0.25,  # 反差变化：浅字、褪色打印和灰底扫描。
                        saturation=0.35,  # 饱和度变化：彩字/背景的颜色浓淡。
                        hue=0.03,  # 小幅色相变化：拍照白平衡和设备偏色。
                    )],
                    p=0.7,
                ),
                # 自动对比度（20%）：覆盖扫描软件拉伸灰度、截图反差差异。
                transforms.RandomAutocontrast(p=0.2),
                # 反色（8%）：覆盖深色主题、黑底白字；也会反转残留章的颜色。
                transforms.RandomInvert(p=0.08),
                *channel_transform,
                # 转为 [C, H, W] 的 [0, 1] 浮点张量，供模糊/噪声/擦除使用。
                transforms.ToTensor(),
                # 轻度模糊（15%）：模拟轻微失焦、低分辨率截图和缩放损失。
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.15,
                ),
                # 高斯噪声（15%）：模拟扫描颗粒、拍照暗部噪点，不改变文字位置。
                transforms.RandomApply(
                    [AddGaussianNoise(0.0, 0.015)],
                    p=0.15,
                ),
                # 小块擦除（5%，面积 0.5%～3%）：模拟污点、磨损和局部遮挡。
                # 仍可能遮住小数点或短文本的关键笔画，因此保持低概率、小面积。
                transforms.RandomErasing(
                    p=0.05,
                    scale=(0.005, 0.03),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
                normalize,
            ]
            self.transform = transforms.Compose(train_transforms)
        else:
            self.transform = transforms.Compose([
                *channel_transform,
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_process_image(self, index: int) -> torch.Tensor:
        """
        通用处理：读取/透明通道合成 -> 保守去章 -> 源分辨率几何扰动 -> 等比缩放
        -> 增强/归一化。几何扰动必须在缩放前完成，否则长行旋转会超出画布边界。
        """
        image_path = self.image_paths[index]
        try:
            image = load_rgb_image(image_path)
            # 在缩放和随机颜色扰动前判断原始红色；训练、验证、推理共享。
            image = remove_red_seal(image)
            if self.source_augment is not None:
                image = self.source_augment(image)
            image = resize_detection_image(
                image,
                target_height=self.img_height,
                max_width=self.max_width,
            )
            return self.transform(image)
        except Exception as exc:
            # 训练时不能用空白 dummy 图配真实标签，否则会静默污染 CTC loss。
            raise RuntimeError(f"图片预处理失败: {image_path}") from exc


# 3. 具体实现类 (训练 vs 推理)

class TextRecognitionDataset(BaseTextDataset):
    """读取一份或多份 Tab 标注，返回 (图片张量, 文字)。
    路径相对于所属标注；文字 strip 后为空则跳过。img_width 是缩放宽度上限。"""

    def __init__(
        self,
        annotation_path: str | os.PathLike | Sequence[str | os.PathLike],
        img_height: int,
        img_width: int,
        num_channels: int,
        is_train=True,
    ):
        super().__init__(
            img_height,
            num_channels,
            is_train=is_train,
            max_width=img_width,
        )

        self.labels = []

        if isinstance(annotation_path, (str, os.PathLike)):
            annotation_paths = [os.fspath(annotation_path)]
        else:
            annotation_paths = [os.fspath(path) for path in annotation_path]

        if not annotation_paths:
            raise ValueError("标注文件列表不能为空")

        empty_label_count = 0

        for current_annotation_path in annotation_paths:
            annotation_dir = os.path.dirname(current_annotation_path)
            if not os.path.exists(current_annotation_path):
                raise FileNotFoundError(f"标注文件不存在: {current_annotation_path}")

            malformed_lines = []
            missing_images = []
            with open(current_annotation_path, "r", encoding="utf-8") as annotation_file:
                for line_number, line in enumerate(annotation_file, start=1):
                    try:
                        path_part, label_field = split_annotation_fields(
                            line, current_annotation_path, line_number
                        )
                    except ValueError:
                        malformed_lines.append(line_number)
                        continue

                    label = label_field.strip()
                    if not label:
                        empty_label_count += 1
                        continue

                    annotated_path = os.path.join(annotation_dir, path_part)
                    try:
                        resolved_path = resolve_image_path(annotated_path)
                    except FileNotFoundError:
                        missing_images.append((line_number, annotated_path))
                        continue

                    self.image_paths.append(resolved_path)
                    self.labels.append(label)

            if malformed_lines or missing_images:
                details = []
                if malformed_lines:
                    details.append(f"格式错误行: {malformed_lines[:5]}")
                if missing_images:
                    details.append(f"缺失图片: {missing_images[:5]}")
                raise ValueError(
                    f"标注数据校验失败 ({current_annotation_path}): "
                    f"{'; '.join(details)}"
                )

        if empty_label_count:
            print(f"已跳过 {empty_label_count} 条空标签样本")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        label = self.labels[index]
        return image, label


class InferenceDataset(BaseTextDataset):
    """读取图片列表，返回 (图片张量, 图片路径)；默认不限宽且不随机增强。"""

    def __init__(
        self,
        image_paths: list,
        img_height: int,
        num_channels: int,
        img_width: int | None = None,
        source_augment=None,
    ):
        super().__init__(
            img_height,
            num_channels,
            is_train=False,
            max_width=img_width,
            source_augment=source_augment,
        )
        self.image_paths = [resolve_image_path(path) for path in image_paths]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        # 推理时没有 Label，返回路径作为 ID，方便对应结果
        path = self.image_paths[index]
        return image, path


# 4. DataLoader 创建工厂

def _seed_worker(worker_id: int) -> None:
    """按 worker 的 torch 种子同步 random 和 NumPy。

    torchvision 的随机增强走 torch 自身的 worker 种子，几何增强使用 random；
    不对齐时同一份配置每次训练的几何扰动序列都不同。"""
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def create_dataloader(annotation_path: str | os.PathLike | Sequence[str | os.PathLike],
                      img_height: int, img_width: int, num_channels: int, batch_size: int,
                      num_workers: int, is_train=True) -> DataLoader:
    """
    创建训练/验证用的 DataLoader，支持一个或多个标注文件。
    """
    dataset = TextRecognitionDataset(
        annotation_path=annotation_path,
        img_height=img_height,
        img_width=img_width,
        num_channels=num_channels,
        is_train=is_train,
    )

    if is_train:
        sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dynamic_padding_collate_fn,
            # 固定基准种子只影响增强抽样；每个 epoch 新建迭代器时基准种子继续前进，
            # 因此每个 epoch 的增强不同，同一份配置的运行之间可复现。
            generator=torch.Generator().manual_seed(DATALOADER_SEED),
            worker_init_fn=_seed_worker,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=dynamic_padding_collate_fn
        )

    return dataloader
