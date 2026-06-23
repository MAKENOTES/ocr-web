"""
用于文本识别的自定义数据集和数据加载器 (通用封装版)。
包含:
1. BucketBatchSampler: 桶采样 (训练用)
2. dynamic_padding_collate_fn: 动态 Padding (训练/推理通用)
3. TextRecognitionDataset: 带标签的数据集 (训练用)
4. InferenceDataset: 纯图片数据集 (推理用)
"""
import os
import math
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# 防止截断报错 (针对超大图)
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        padded_imgs: [B, C, H, W_max] 填充后的图像张量
        labels_or_paths: 标签或路径的元组
        img_widths: [B] 每张图片 padding 前的真实宽度（像素），
                    用于计算每个样本在 CNN 下采样后的真实时间步数，
                    使 CTC Loss 和解码只关注有效区域，忽略 padding。
    """
    images, labels_or_paths = zip(*batch)

    # 记录每张图片的真实宽度
    img_widths = torch.IntTensor([img.shape[2] for img in images])

    # 找到当前 Batch 中最宽的宽度
    max_width = max(img_widths).item()

    # 宽度补齐到 8 的倍数
    max_width = ((max_width + 7) // 8) * 8

    num_channels = images[0].shape[0]
    height = images[0].shape[1]
    batch_size = len(images)

    # 填充背景 (Normalize mean=0.5 -> 1.0 为白色)
    padded_imgs = torch.full((batch_size, num_channels, height, max_width), 1.0)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img

    return padded_imgs, labels_or_paths, img_widths


# ==========================================
# 2. 基础数据集类 (封装通用逻辑)
# ==========================================

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def remove_red_seal(image: Image.Image,
                    sensitivity: float = 0.7,
                    min_saturation: int = 50,
                    min_value: int = 80,
                    keep_ratio_thresh: float = 0.5) -> Image.Image:
    """
    从RGB图像中去除红色印章区域，用白色替换（带"内容护栏"）。

    适用场景：黑字上盖了红章 —— 去掉红色，保留黑字。

    护栏 (keep_ratio_thresh)：
        本数据集里存在大量"红色本身就是内容"的样本 ——
        例如红色文字(删除/导入说明)、红底白字按钮、红色角标数字(49/27/61)。
        对这些图无脑抹红会把内容整片涂白，造成不可恢复的信息丢失。
        因此抹白前先估算"去红后还剩多少深色内容"：
          - 若剩余内容 < keep_ratio_thresh，判定红色=内容 → 原样返回，不去红。
          - 否则才执行抹白（真正的"黑字+红章"场景）。
        关掉护栏可设 keep_ratio_thresh=0.0。

    Args:
        image: PIL RGB图像
        sensitivity: 红色检测灵敏度 (0.0-1.0)，越高越激进
                     0.5 = 保守 (只去除非常红的区域)
                     0.7 = 推荐值
                     0.9 = 激进 (可能误伤部分棕色/橙色文字)
        min_saturation: 最小饱和度阈值 (0-255)，低于此值不认为是红色
        min_value: 最小亮度阈值 (0-255)，低于此值不认为是红色
        keep_ratio_thresh: 去红后深色内容的最低留存比例，低于此值则放弃去红

    Returns:
        去除红章后的PIL RGB图像（若触发护栏则返回原图）
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)

    # ========== 转HSV色彩空间 ==========
    # PIL的HSV转换比较慢，这里手动用numpy实现
    # 或者用简单的方式：用PIL转换
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image)

    h = hsv_array[:, :, 0]  # Hue: 0-255 (PIL的HSV, 0=0°, 255=360°)
    s = hsv_array[:, :, 1]  # Saturation: 0-255
    v = hsv_array[:, :, 2]  # Value: 0-255

    # ========== 红色检测 ==========
    # PIL的H值范围是 0-255，对应 0°-360°
    # 红色在HSV中分两段:
    #   H ≈ 0°-20° (对应PIL: 0-14)
    #   H ≈ 340°-360° (对应PIL: 241-255)

    # 根据sensitivity调整范围
    # sensitivity=0.7 -> hue_range_low=0~18, hue_range_high=237~255
    hue_expand = int(sensitivity * 6)  # 基准±扩展

    low_hue_max = 12 + hue_expand  # 默认约18
    high_hue_min = 243 - hue_expand  # 默认约237

    # 调整饱和度阈值
    sat_thresh = max(10, int(min_saturation * (1.2 - sensitivity * 0.4)))

    # 红色mask: (低色相段 OR 高色相段) AND 足够饱和 AND 足够亮
    red_mask = (
            ((h <= low_hue_max) | (h >= high_hue_min)) &  # 色相在红色范围
            (s >= sat_thresh) &  # 饱和度够高（排除灰白色）
            (v >= min_value)  # 亮度够高（排除深黑色）
    )

    # 没有检测到红色，直接返回，省去后续计算
    if not red_mask.any():
        return image

    # ========== 内容护栏：判断"红色是否就是内容" ==========
    # 用灰度下的深色像素近似"内容/墨迹"。比较去红前后的深色内容量：
    # 若去红后深色内容留存比例过低，说明红色承载了主要内容，放弃去红。
    # 注意：必须用真实亮度 L（红色 L≈76 属于"深色"），不能用 HSV 的 V 通道
    #       （红色 V=255 会被误判为亮色，导致护栏失效）。
    gray = np.array(image.convert('L'))
    dark_mask = gray < 150
    ink_before = int(dark_mask.sum())
    if ink_before > 0:
        # 红色文字/红底/红角标本身就是"深色内容"，抹白会让信息坍塌；
        # 这里同时考虑：红色像素占比过大时也视为"红色即内容"。
        ink_after = int((dark_mask & ~red_mask).sum())
        keep_ratio = ink_after / ink_before
        red_frac = float(red_mask.mean())
        if keep_ratio < keep_ratio_thresh or red_frac > 0.18:
            return image  # 红色=内容，保留原图

    # ========== 形态学优化 (可选但推荐) ==========
    # 轻微膨胀，填补印章内部的小空洞，让去除更干净。
    # iterations 由 2 降为 1：减少对红章邻近黑色笔画的啃噬。
    try:
        from scipy import ndimage
        # 膨胀1个像素，连接碎片
        red_mask = ndimage.binary_dilation(red_mask, iterations=1)
        # 闭运算，填充小洞
        red_mask = ndimage.binary_closing(red_mask, iterations=1)
    except ImportError:
        # 没有scipy也能用，只是效果稍差
        pass

    # ========== 用白色替换红色区域 ==========
    result = img_array.copy()
    result[red_mask] = [255, 255, 255]

    return Image.fromarray(result)

class BaseTextDataset(Dataset):
    """
    文本识别数据集基类。
    封装了核心的图片加载、预处理、增强逻辑。
    """

    def __init__(self, img_height, num_channels, is_train=False):
        self.img_height = img_height
        self.num_channels = num_channels
        self.image_paths = []  # 子类填充此列表

        # 定义 Transform
        if is_train:
            self.transform = transforms.Compose([
                # 训练集：
                # 1️⃣ 轻微旋转（保持文本可读）
                transforms.RandomRotation(
                    degrees=2,
                    fill=255  # 白底图像很重要
                ),

                # 2️⃣ 轻微透视扰动
                transforms.RandomPerspective(
                    distortion_scale=0.1,
                    p=0.3
                ),

                # 3️⃣ 颜色扰动
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1
                ),

                # 4️⃣ 转 tensor
                transforms.ToTensor(),

                # 5️⃣ 高斯模糊（模拟拍照虚焦）
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.3
                ),

                # 6️⃣ 添加轻微高斯噪声
                transforms.RandomApply(
                    [AddGaussianNoise(0., 0.02)],
                    p=0.3
                ),

                # 7️⃣ 随机擦除（必须在 ToTensor 之后）
                transforms.RandomErasing(
                    p=0.18,
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                    value='random'
                ),

                # 8️⃣ 归一化
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )
            ])
        else:
            self.transform = transforms.Compose([
                # 验证/推理集：仅归一化
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_process_image(self, index: int) -> torch.Tensor:
        """
        通用的单张图片处理逻辑：读取 -> 转通道 -> Resize -> Transform
        """
        image_path = self.image_paths[index]
        try:
            # 1. 统一转为 RGB 读取
            image = Image.open(image_path).convert('RGB')
            image = remove_red_seal(image)  # 在RGB阶段去红章
            # 2. 通道处理 (保留红章逻辑)
            if self.num_channels == 1:
                image = image.convert('L')

            # 3. Resize (固定高度，宽度按比例缩放，含两层保护)
            w, h = image.size
            target_h = self.img_height
            scale = target_h / h
            new_w = int(w * scale)

            # ---- 保护 A：高图宽度保护 ----
            # 原图很高时（如 h=93），按高度缩放会把宽度压缩得太狠，
            # 导致 CTC 时间步不足。此处保证缩放后宽度不低于原始宽度的 50%。
            min_w_from_aspect = int(w * 0.5)
            if new_w < min_w_from_aspect:
                new_w = min(min_w_from_aspect, 800)  # 上限 800 防止显存爆炸

            # 先 Resize 到目标尺寸
            image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)

            # ---- 保护 B：最小宽度保护 ----
            # 小图（单字符、细线符号等）Resize 后宽度太窄，CNN 下采样后
            # 特征几乎为零。将过窄的图片居中放置在白色画布上，不拉伸变形。
            MIN_WIDTH = 48  # CNN 4x 下采样后至少 12 个时间步
            if new_w < MIN_WIDTH:
                if self.num_channels == 1:
                    canvas = Image.new('L', (MIN_WIDTH, target_h), 255)
                else:
                    canvas = Image.new('RGB', (MIN_WIDTH, target_h), (255, 255, 255))
                offset_x = (MIN_WIDTH - new_w) // 2
                canvas.paste(image, (offset_x, 0))
                image = canvas

            # 4. Apply Transforms
            image_tensor = self.transform(image)
            return image_tensor

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回 dummy tensor
            return torch.ones((self.num_channels, self.img_height, 8))


# ==========================================
# 3. 具体实现类 (训练 vs 推理)
# ==========================================

class TextRecognitionDataset(BaseTextDataset):
    """
    [训练用] 读取标注文件，返回 (Image, Label)
    """

    def __init__(self, annotation_path: str, img_height: int, img_width: int, num_channels: int, is_train=True):
        # 初始化基类
        super().__init__(img_height, num_channels, is_train)

        # img_width 参数仅作为兼容保留，实际由 dynamic_padding 控制
        self.labels = []

        annotation_dir = os.path.dirname(annotation_path)
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        path_part = parts[0].strip()
                        label_part = ','.join(parts[1:]).strip().replace('"', '')

                        full_path = os.path.join(annotation_dir, path_part)
                        self.image_paths.append(full_path)
                        self.labels.append(label_part)
        else:
            print(f"Warning: Annotation file not found at {annotation_path}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        label = self.labels[index]
        return image, label


class InferenceDataset(BaseTextDataset):
    """
    [推理用] 直接读取图片列表，返回 (Image, ImagePath)
    """

    def __init__(self, image_paths: list, img_height: int, num_channels: int):
        # 推理模式 is_train=False
        super().__init__(img_height, num_channels, is_train=False)
        self.image_paths = image_paths

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        # 推理时没有 Label，返回路径作为 ID，方便对应结果
        path = self.image_paths[index]
        return image, path


# ==========================================
# 4. DataLoader 创建工厂 (保持不变，仅适配新类)
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
        is_train=is_train
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