"""推理数据集、共享图像预处理管线与动态补边。

确定性步骤（RGB 合成、去章、定高缩放）由 preprocessing.py 提供；本模块只负责
推理数据集、验证/推理张量化以及 batch 补边。训练随机增强仍在 augmentation.py，
训练标注数据集在 dataset.py。runtime 中的 dataset.py 由本文件生成。
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from recognition.components.preprocessing import (
    load_rgb_image,
    remove_red_seal,
    resize_detection_image,
    resolve_image_path,
)

# MobileFeatureExtractor 的横向感受野为 53 px。每张图右侧至少保留 32 px
# 背景，使有效 CNN 特征不会因 batch 中最长图片不同而看到不同的张量边界。
CNN_RIGHT_CONTEXT = 32


def build_eval_transform(num_channels: int) -> transforms.Compose:
    """构建验证/推理的确定性变换：通道适配、张量化和 [-1, 1] 归一化。"""
    if num_channels not in (1, 3):
        raise ValueError(f"num_channels 仅支持 1 或 3，当前为 {num_channels}")
    channel_transform = (
        # 兼容旧的单通道模型；RGB 模型保留颜色信息。这是通道适配，不是增强。
        [transforms.Grayscale(num_output_channels=1)]
        if num_channels == 1
        else []
    )
    return transforms.Compose([
        *channel_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
    ])


def load_and_preprocess_image(image_path, img_height: int, max_width: int | None = None,
                              source_augment=None) -> Image.Image:
    """读取、去章、可选源分辨率扰动、定高缩放；不做张量化。

    训练、验证和推理共用该顺序，避免几何扰动发生在缩放后而裁掉字形。
    """
    image = load_rgb_image(image_path)
    image = remove_red_seal(image)
    if source_augment is not None:
        image = source_augment(image)
    return resize_detection_image(
        image,
        target_height=img_height,
        max_width=max_width,
    )


def dynamic_padding_collate_fn(batch):
    """按批次动态 Padding，返回图片、路径或标签、每张图的真实宽度。

    padded_imgs: [B, C, H, W_max]，每张图右侧至少保留 CNN_RIGHT_CONTEXT 像素背景。
    img_widths: [B]，不含右侧保护区和 batch padding 的真实宽度（像素），
    用于计算 CNN 下采样后的有效时间步，使 CTC 解码忽略 padding。
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


class InferenceDataset(Dataset):
    """读取完整 detection 文字框，返回 (图片张量, 图片路径)。"""

    def __init__(
        self,
        image_paths: list,
        img_height: int,
        num_channels: int,
        img_width: int | None = None,
        source_augment=None,
    ):
        self.image_paths = [resolve_image_path(path) for path in image_paths]
        self.img_height = img_height
        self.num_channels = num_channels
        self.max_width = img_width
        # 默认关闭扰动；鲁棒性评估可传入确定性扰动，训练随机增强不在这里。
        self.source_augment = source_augment
        self.transform = build_eval_transform(num_channels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        try:
            image = load_and_preprocess_image(
                image_path,
                self.img_height,
                self.max_width,
                self.source_augment,
            )
            return self.transform(image), image_path
        except Exception as exc:
            # 推理也不能用空白 dummy 图替代，否则错误只会在结果里表现为乱码。
            raise RuntimeError(f"图片预处理失败: {image_path}") from exc
