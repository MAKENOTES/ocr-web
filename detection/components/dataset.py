import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    """
    用于OCR文本检测任务的自定义数据集。
    该数据集负责加载图像和对应的ICDAR格式的标签文件。
    1、把label 的两个点转成四个点
    2、调用 transform 得到 张量
    """
    def __init__(self, image_dir, label_dir, transform=None):
        """
        初始化数据集。
        :param image_dir: string, 存放所有图像的目录。
        :param label_dir: string, 存放所有标签文件的目录。
        :param transform: callable, optional, 应用于样本的可选转换。
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # 查找 .jpg 和 .png 格式的图像
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

        # 创建对应的标签文件路径
        self.label_paths = []
        for p in self.image_paths:
            basename = os.path.basename(p)
            name, _ = os.path.splitext(basename)
            label_name = "gt_" + name + ".txt"
            self.label_paths.append(os.path.join(self.label_dir, label_name))

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
                
                # 修正后的逻辑：处理以空格分隔的值
                parts = line.split() # 使用 split() 来按空格分割
                
                try:
                    if len(parts) < 4:
                        print(f"警告: 跳过格式错误的行于 {label_path}: {line}")
                        continue
                    
                    # 假设格式为: x_min y_min x_max y_max
                    points = [int(float(p)) for p in parts[:4]]
                    x_min, y_min, x_max, y_max = points
                    
                    # 将矩形转换为 4x2 的四边形数组
                    quad = np.array([
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ], dtype=np.float32)
                    quads.append(quad)
                    
                    # 处理可选的文本内容
                    transcription = parts[4] if len(parts) > 4 else ""
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
