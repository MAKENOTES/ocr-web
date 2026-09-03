"""
用于文本识别的自定义数据集和数据加载器 (通用封装版)。
包含:
1. BucketBatchSampler: 桶采样 (训练用)
2. dynamic_padding_collate_fn: 动态 Padding (训练/推理通用)
3. TextRecognitionDataset: 带标签的数据集 (训练用)
4. InferenceDataset: 纯图片数据集 (推理用)
"""
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFile, ImageOps, ImageStat
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# 防止截断报错 (针对超大图)
ImageFile.LOAD_TRUNCATED_IMAGES = True

MIN_IMAGE_WIDTH = 48
# MobileFeatureExtractor 的横向感受野为 53 px。每张图右侧至少保留 32 px
# 背景，使有效 CNN 特征不会因 batch 中最长图片不同而看到不同的张量边界。
CNN_RIGHT_CONTEXT = 32


def resolve_image_path(image_path: str | os.PathLike) -> str:
    """返回实际存在的图片路径；标注中的文件名和扩展名必须准确。"""
    path = Path(image_path)
    if path.is_file():
        return str(path)

    raise FileNotFoundError(f"图片不存在: {path}")


def load_rgb_image(image_path: str | os.PathLike) -> Image.Image:
    """
    安全读取图片为 RGB。

    PNG 可能带透明通道；直接 ``convert('RGB')`` 会把透明区域变成黑色。
    这里根据可见内容亮度选择黑/白底进行合成，确保黑字和白字都不会消失。
    """
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


def _border_fill_color(image: Image.Image):
    """估计 detection 框的背景色，用于给窄图补边。"""
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        border = np.concatenate(
            (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
        )
        return int(np.median(border))

    channels = pixels.shape[2]
    border = np.concatenate(
        (
            pixels[0, :, :].reshape(-1, channels),
            pixels[-1, :, :].reshape(-1, channels),
            pixels[:, 0, :].reshape(-1, channels),
            pixels[:, -1, :].reshape(-1, channels),
        ),
        axis=0,
    )
    return tuple(int(value) for value in np.median(border, axis=0))


def resize_detection_image(
    image: Image.Image,
    target_height: int,
    max_width: int | None = None,
    min_width: int = MIN_IMAGE_WIDTH,
) -> Image.Image:
    """保持宽高比缩放 detection 结果；不再做二次内容裁剪。"""
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
        fill = _border_fill_color(image)
        canvas = Image.new(image.mode, (canvas_width, target_height), fill)
        offset_x = (canvas_width - resized_width) // 2
        canvas.paste(image, (offset_x, 0))
        image = canvas

    return image


# ==========================================
# 1. 采样与整理工具 (无需修改，完全通用)
# ==========================================

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


# ==========================================
# 2. 基础数据集类 (封装通用逻辑)
# ==========================================

class AddGaussianNoise(object):
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
    ):
        if num_channels not in (1, 3):
            raise ValueError(f"num_channels 仅支持 1 或 3，当前为 {num_channels}")

        self.img_height = img_height
        self.num_channels = num_channels
        self.max_width = max_width
        self.image_paths = []  # 子类填充此列表

        channel_transform = (
            [transforms.Grayscale(num_output_channels=1)]
            if num_channels == 1
            else []
        )
        normalize = transforms.Normalize(
            mean=[0.5] * num_channels,
            std=[0.5] * num_channels,
        )

        if is_train:
            # detection 已经给出文字区域，不再旋转、透视或二次裁剪，避免紧框文字
            # 被切掉。增强集中在颜色、极性、清晰度与轻噪声上。
            train_transforms = [
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.25,
                        saturation=0.35,
                        hue=0.03,
                    )],
                    p=0.7,
                ),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomInvert(p=0.08),
                *channel_transform,
                transforms.ToTensor(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.15,
                ),
                transforms.RandomApply(
                    [AddGaussianNoise(0.0, 0.015)],
                    p=0.15,
                ),
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
        通用处理：读取/透明通道合成 -> 等比缩放 -> 颜色增强/归一化。
        """
        image_path = self.image_paths[index]
        try:
            image = load_rgb_image(image_path)
            image = resize_detection_image(
                image,
                target_height=self.img_height,
                max_width=self.max_width,
            )
            return self.transform(image)
        except Exception as exc:
            # 训练时不能用空白 dummy 图配真实标签，否则会静默污染 CTC loss。
            raise RuntimeError(f"图片预处理失败: {image_path}") from exc


# ==========================================
# 3. 具体实现类 (训练 vs 推理)
# ==========================================

class TextRecognitionDataset(BaseTextDataset):
    """
    [训练用] 读取标注文件，返回 (Image, Label)
    """

    def __init__(
        self,
        annotation_path: str,
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

        annotation_dir = os.path.dirname(annotation_path)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

        empty_label_count = 0
        malformed_lines = []
        missing_images = []

        with open(annotation_path, "r", encoding="utf-8") as annotation_file:
            for line_number, line in enumerate(annotation_file, start=1):
                parts = line.rstrip("\r\n").split(",", 1)
                if len(parts) != 2:
                    malformed_lines.append(line_number)
                    continue

                path_part = parts[0].strip()
                label = parts[1].strip().replace('"', "")
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
            raise ValueError(f"标注数据校验失败 ({annotation_path}): {'; '.join(details)}")

        if empty_label_count:
            print(f"已跳过 {empty_label_count} 条空标签样本")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        label = self.labels[index]
        return image, label


class InferenceDataset(BaseTextDataset):
    """
    [推理用] 直接读取图片列表，返回 (Image, ImagePath)
    """

    def __init__(
        self,
        image_paths: list,
        img_height: int,
        num_channels: int,
        img_width: int | None = None,
    ):
        super().__init__(
            img_height,
            num_channels,
            is_train=False,
            max_width=img_width,
        )
        self.image_paths = [resolve_image_path(path) for path in image_paths]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        # 推理时没有 Label，返回路径作为 ID，方便对应结果
        path = self.image_paths[index]
        return image, path


# ==========================================
# 4. DataLoader 创建工厂
# ==========================================

def create_dataloader(annotation_path: str, img_height: int, img_width: int, num_channels: int, batch_size: int,
                      num_workers: int, is_train=True) -> DataLoader:
    """
    创建训练/验证用的 DataLoader
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
            collate_fn=dynamic_padding_collate_fn
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
