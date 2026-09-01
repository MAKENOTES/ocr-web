import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

class OCRDataset(Dataset):
    """
    用于OCR文本检测任务的自定义数据集。
    该数据集负责加载图像和对应的ICDAR格式的标签文件。
    1、把label 的两个点转成四个点
    2、调用 transform 得到 张量
    """
    def __init__(self, image_dir, label_dir, transform=None,
                 image_extensions=DEFAULT_IMAGE_EXTENSIONS,
                 require_labels=False):
        """
        初始化数据集。
        :param image_dir: string, 存放所有图像的目录。
        :param label_dir: string, 存放所有标签文件的目录。
        :param transform: callable, optional, 应用于样本的可选转换。
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # 同时支持多种扩展名，也允许训练入口显式限定为 PNG，避免旧 JPG
        # 留在目录中时被意外混入新数据。
        extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in image_extensions
        }
        image_root = Path(image_dir)
        self.image_paths = sorted(
            str(path)
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ) if image_root.exists() else []

        # 创建对应的标签文件路径
        self.label_paths = []
        for p in self.image_paths:
            basename = os.path.basename(p)
            name, _ = os.path.splitext(basename)
            gt_label_path = os.path.join(self.label_dir, f"gt_{name}.txt")
            plain_label_path = os.path.join(self.label_dir, f"{name}.txt")
            self.label_paths.append(
                gt_label_path if os.path.exists(gt_label_path) else plain_label_path
            )

        if require_labels:
            missing_labels = [
                label_path
                for label_path in self.label_paths
                if not os.path.exists(label_path)
            ]
            if missing_labels:
                examples = "\n  ".join(missing_labels[:5])
                raise FileNotFoundError(
                    f"{len(missing_labels)} 张图片缺少对应标签。示例:\n  {examples}\n"
                    "标签名应为 gt_<图片名>.txt 或 <图片名>.txt。"
                )

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取一个样本。
        :param idx: int, 样本的索引。
        :return: dict, 包含图像和标签信息的字典。如果应用了转换，则返回转换后的样本。
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 加载标签
        quads = []
        transcriptions = []
        
        if not os.path.exists(label_path):
            sample = {'image': image, 'quads': np.array([], dtype=np.float32), 'transcriptions': []}
            if self.transform:
                sample = self.transform(sample)
            return sample

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\ufeff').strip()
                if not line:
                    continue
                
                # 同时兼容空格分隔 bbox4 和逗号/空格分隔的 ICDAR 四点格式。
                parts = line.replace(',', ' ').split()
                
                try:
                    numeric_values = []
                    for part in parts:
                        try:
                            numeric_values.append(float(part))
                        except ValueError:
                            break

                    if len(numeric_values) < 4:
                        print(f"警告: 跳过格式错误的行于 {label_path}: {line}")
                        continue

                    if len(numeric_values) >= 8:
                        # ICDAR: x1 y1 x2 y2 x3 y3 x4 y4 [transcription]
                        quad = np.asarray(numeric_values[:8], dtype=np.float32).reshape(4, 2)
                        transcription_start = 8
                    else:
                        # bbox4: x_min y_min x_max y_max [transcription]
                        x_min, y_min, x_max, y_max = numeric_values[:4]
                        quad = np.array([
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max]
                        ], dtype=np.float32)
                        transcription_start = 4

                    # 小图中 1px 误差占比很大，因此保留浮点坐标，不再强制取整。
                    cv_width = float(quad[:, 0].max() - quad[:, 0].min())
                    cv_height = float(quad[:, 1].max() - quad[:, 1].min())
                    if cv_width <= 0 or cv_height <= 0:
                        print(f"警告: 跳过退化框于 {label_path}: {line}")
                        continue
                    quads.append(quad)

                    # 处理可选的文本内容
                    transcription = " ".join(parts[transcription_start:])
                    if transcription != "###": # 忽略 "don't care" 区域
                        transcriptions.append(transcription)

                except (ValueError, IndexError):
                    print(f"警告: 无法解析行于 {label_path}: {line}")
                    continue

        #把一个普通的 Python 列表（list）转换成一个 NumPy 数组（array）。
        quads = np.array(quads, dtype=np.float32) if quads else np.array([], dtype=np.float32)

        sample = {'image': image, 'quads': quads, 'transcriptions': transcriptions}

        if self.transform:
            sample = self.transform(sample)

        return sample
