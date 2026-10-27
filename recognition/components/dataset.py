"""训练数据集、分桶与 DataLoader。

推理数据集和共享图像预处理管线位于 inference_data.py；本文件保留
InferenceDataset 与动态补边的兼容导出，训练和推理使用同一份实现。
随机增强见 augmentation.py，标注解析见 recognition.utils.annotation。"""
import os
import random
from collections.abc import Sequence
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from recognition.components.augmentation import RandomSourceGeometry, build_transform
from recognition.components.inference_data import (
    CNN_RIGHT_CONTEXT,  # noqa: F401 - 兼容旧导入，训练补边常量
    InferenceDataset,  # noqa: F401 - 兼容旧导入，实现位于 inference_data.py
    dynamic_padding_collate_fn,
    load_and_preprocess_image,
)
from recognition.components.preprocessing import resolve_image_path
from recognition.utils.annotation import split_annotation_fields

# 训练 DataLoader 的随机基准种子。它只决定各 worker 的随机增强序列，与模型初始化和
# 主进程中的分桶打乱无关；固定后同一份代码和配置可以复现每个 worker 的增强。
DATALOADER_SEED = 20260923


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


# 2. 基础数据集类 (封装通用逻辑)


class BaseTextDataset(Dataset):
    """
    文本识别数据集基类。
    封装图片加载、预处理和增强调用。
    """

    def __init__(
        self,
        img_height,
        num_channels,
        is_train=False,
        max_width=None,
        source_augment=None,
    ):
        self.img_height = img_height
        self.num_channels = num_channels
        self.max_width = max_width
        # 缩放前作用于源分辨率图片的几何扰动。训练默认启用；调用方可传入
        # 确定性扰动做鲁棒性评估，传 None 且 is_train=False 时完全关闭。
        if is_train and source_augment is None:
            source_augment = RandomSourceGeometry()
        self.source_augment = source_augment
        self.image_paths = []  # 子类填充此列表

        # 缩放后的随机颜色、极性、清晰度和轻噪声变化；验证/推理只做基础张量化。
        self.transform = build_transform(is_train=is_train, num_channels=num_channels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_process_image(self, index: int) -> torch.Tensor:
        """
        通用处理：读取/透明通道合成 -> 保守去章 -> 源分辨率几何扰动 -> 等比缩放
        -> 增强/归一化。几何扰动必须在缩放前完成，否则长行旋转会超出画布边界。
        """
        image_path = self.image_paths[index]
        try:
            image = load_and_preprocess_image(
                image_path,
                self.img_height,
                self.max_width,
                self.source_augment,
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
