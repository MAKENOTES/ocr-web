import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from .ground_truth import generate_ground_truth
from torchvision.transforms import Normalize

class ResizeAndPad:
    """
    一个自定义的transform，用于在保持宽高比的同时缩放和填充图像。
    它会将图像缩放到一个目标尺寸内，然后用图像边缘背景色填充剩余部分，
    使得最终图像尺寸固定。同时，它也会相应地调整边界框的坐标。
    """
    def __init__(self, output_size, padding_color="auto"):
        """
        初始化ResizeAndPad。
        :param output_size: int或tuple, 目标输出尺寸。如果是int，则宽高相同。
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.padding_color = padding_color

    def _resolve_padding_color(self, image):
        if self.padding_color != "auto":
            return self.padding_color

        # UI 裁剪通常是白底、灰底或彩色按钮。使用边缘像素中位数填充，
        # 避免固定灰色在小图四周制造一圈训练时并不存在的强边界。
        pixels = np.asarray(image)
        border = np.concatenate([
            pixels[0, :, :], pixels[-1, :, :],
            pixels[:, 0, :], pixels[:, -1, :]
        ], axis=0)
        return tuple(np.median(border, axis=0).astype(np.uint8).tolist())

    def __call__(self, sample):
        """
        对输入的样本（包含图像和四边形）进行变换。
        :param sample: dict, 包含'image'和'quads'的字典。
        :return: dict, 变换后的样本。
        """
        image, quads = sample['image'], sample['quads']
        
        w, h = image.size
        new_h, new_w = self.output_size
        
        # 计算缩放比例和新的尺寸
        scale = min(new_w / w, new_h / h)
        resized_w, resized_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        resized_image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        
        # 使用与原图边缘一致的背景色填充。
        new_image = Image.new('RGB', (new_w, new_h), self._resolve_padding_color(image))
        new_image.paste(resized_image, ((new_w - resized_w) // 2, (new_h - resized_h) // 2))
        
        # 调整四边形坐标
        if quads.size > 0:
            quads *= scale
            quads[:, :, 0] += (new_w - resized_w) // 2
            quads[:, :, 1] += (new_h - resized_h) // 2
            
        sample['image'] = new_image
        sample['quads'] = quads
        
        return sample

class GenerateGT:
    """
    一个自定义的transform，用于生成EAST模型的真值（ground truth）图。
    它调用 `generate_ground_truth` 函数来创建分数图和几何图。
    """
    def __init__(self, input_size=512, score_map_scale=0.25):
        """
        初始化GenerateGT。
        :param input_size: int, 模型的输入图像尺寸。
        :param score_map_scale: float, 分数图相对于输入图像的缩放比例。
        """
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.score_map_scale = score_map_scale

    def __call__(self, sample):
        """
        对输入的样本进行变换，添加'score_map'和'geo_map'。
        :param sample: dict, 包含'quads'的字典。
        :return: dict, 添加了真值图的样本。
        """
        quads = sample['quads']
        
        score_map, geo_map = generate_ground_truth(
            image_size=self.input_size,
            quads=quads,
            score_map_scale=self.score_map_scale
        )
        
        sample['score_map'] = score_map
        sample['geo_map'] = geo_map
        
        return sample


class ColorJitterTransform:
    def __init__(self, brightness=0.12, contrast=0.15, saturation=0.10, hue=0.02):
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, sample):
        # 仅对图像进行颜色抖动，不改变 quads
        # 需要在 ToTensor 和 ResizeAndPad 之前执行，或者确保数据格式兼容 PIL
        sample['image'] = self.jitter(sample['image'])
        return sample


class RandomScale:
    """随机缩放图像，模拟不同距离拍摄的文档"""

    def __init__(self, scale_range=(0.6, 1.4)):
        self.scale_range = scale_range

    def __call__(self, sample):
        image, quads = sample['image'], sample['quads']
        if quads.size == 0:
            return sample

        s = np.random.uniform(*self.scale_range)
        w, h = image.size
        new_w, new_h = int(w * s), int(h * s)

        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        if quads.size > 0:
            quads = quads * s

        sample['image'] = image
        sample['quads'] = quads
        return sample

class RandomHorizontalCrop:
    """随机水平裁剪，迫使模型学习不完整文本行的边界"""

    def __init__(self, crop_ratio_range=(0.7, 1.0)):
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        image, quads = sample['image'], sample['quads']

        if np.random.random() > 0.5:  # 50% 概率执行
            return sample

        if quads.size == 0 or len(quads.shape) != 3:
            return sample

        w, h = image.size
        ratio = np.random.uniform(*self.crop_ratio_range)
        new_w = int(w * ratio)
        x_offset = np.random.randint(0, max(1, w - new_w))

        cropped_image = image.crop((x_offset, 0, x_offset + new_w, h))

        new_quads = quads.copy()
        new_quads[:, :, 0] -= x_offset
        # 保留中心仍在裁剪区域内的框
        centers_x = new_quads[:, :, 0].mean(axis=1)
        keep = (centers_x > 0) & (centers_x < new_w)
        new_quads = new_quads[keep]

        # 裁剪后没有框了，放弃本次裁剪，返回原样本
        if new_quads.size == 0:
            return sample

        new_quads[:, :, 0] = np.clip(new_quads[:, :, 0], 0, new_w)

        sample['image'] = cropped_image
        sample['quads'] = new_quads
        return sample

class ToTensor:
    def __init__(self):
        # ImageNet 的标准均值和方差
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        """
        对输入的样本进行变换。
        :param sample: dict, 包含'image', 'score_map', 'geo_map'等键的字典。
        :return: dict, 只包含'image', 'score_map', 'geo_map'张量的字典。
        """
        # 将 image, score_map, 和 geo_map 转换为张量
        image = F.to_tensor(sample['image'])
        image = self.normalize(image)
        score_map = torch.from_numpy(sample['score_map']).float()
        geo_map = torch.from_numpy(sample['geo_map']).float()

        # 只返回训练所需的张量。
        # 这可以防止 collate 函数尝试堆叠 'quads' 和 'transcriptions'，
        # 因为它们的尺寸是可变的。
        return {
            'image': image,
            'score_map': score_map,
            'geo_map': geo_map
        }

def get_transforms(is_train=True, input_size=512):
    """
    获取一个组合好的数据转换流程 (pipeline)。
    :param is_train: bool, 是否为训练模式。如果是，未来可以加入数据增强。
    :param input_size: int, 模型的输入尺寸。
    :return: 一个 `torchvision.transforms.Compose` 对象。
    """
    # 为训练添加更多的数据增强
    if is_train:
        return transforms.Compose([
            ColorJitterTransform(),
            # 新数据已经是紧凑 UI 裁剪。RandomHorizontalCrop 会截断文字，
            # RandomScale 随后又被 ResizeAndPad 抵消，因此两者均不启用。
            ResizeAndPad(input_size, padding_color="auto"),
            GenerateGT(input_size),
            ToTensor()
        ])
    else:
        return transforms.Compose([
            ResizeAndPad(input_size, padding_color="auto"),
            GenerateGT(input_size),
            ToTensor()
        ])
