"""文档检测独立维护居中等比缩放、补边和 ImageNet 归一化。"""
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from PIL import Image
from detection_larger.components.ground_truth import generate_ground_truth, make_training_mask

STRIDE = 4  # 网络输出相对输入的步长，每个特征图像素对应 4 个输入像素。
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 按 RGB 通道应用 ImageNet 均值与标准差。


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
        """使用指定颜色或原图边缘像素中位数填充背景。"""
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


class ColorJitterTransform:
    def __init__(self, brightness=0.12, contrast=0.15, saturation=0.10, hue=0.02):
        """设置文档图像的轻量颜色扰动范围，保护浅色文字对比。"""
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, sample):
        """仅修改样本图片的颜色，不改变标注坐标。"""
        # 需要在 ToTensor 和 ResizeAndPad 之前执行，或者确保数据格式兼容 PIL
        sample['image'] = self.jitter(sample['image'])
        return sample


def normalize_input_size(input_size):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if len(input_size) != 2 or any(not isinstance(v, int) or v < 64 or v % 32 for v in input_size):
        raise ValueError("input_size must contain height/width >= 64 divisible by 32")
    return tuple(input_size)


def resize_metadata(image_size, input_size):
    height, width = normalize_input_size(input_size)
    original_width, original_height = image_size
    scale = min(width / original_width, height / original_height)
    # 与原 ResizeAndPad 的向下取整及居中偏移完全一致。
    offset = ((width - int(original_width * scale)) // 2,
              (height - int(original_height * scale)) // 2)
    return scale, offset


def prepare_image(image, input_size):
    """按调用方传入的尺寸预处理整图，返回张量及坐标还原参数；尺寸单位为像素。"""
    size = normalize_input_size(input_size)
    sample = ResizeAndPad(size, padding_color="auto")({
        "image": image, "quads": np.empty((0, 4, 2), dtype=np.float32),
    })
    return NORMALIZE(F.to_tensor(sample["image"])), resize_metadata(image.size, size)


def restore_boxes(boxes, metadata, image_size):
    scale, (offset_x, offset_y) = metadata
    result = boxes.copy()
    result[:, [0, 2]] = np.clip((result[:, [0, 2]] - offset_x) / scale, 0, image_size[0])
    result[:, [1, 3]] = np.clip((result[:, [1, 3]] - offset_y) / scale, 0, image_size[1])
    return result


class GenerateGT:
    def __init__(self, input_size):
        self.input_size = normalize_input_size(input_size)

    def __call__(self, sample):
        quads, ignored = sample["quads"], sample["ignored"]
        sample["score_map"], sample["geo_map"] = generate_ground_truth(self.input_size, quads[~ignored])
        sample["training_mask"] = make_training_mask(self.input_size, quads[ignored])
        return sample


class ToTensor:
    def __call__(self, sample):
        return {
            "image": NORMALIZE(F.to_tensor(sample["image"])),
            **{key: torch.from_numpy(sample[key]).float() for key in ("score_map", "geo_map", "training_mask")},
        }


def get_transforms(is_train, input_size):
    """按显式输入尺寸构建训练或验证变换；训练入口负责传入尺寸，单位为像素。"""
    size = normalize_input_size(input_size)
    # 相对 majorbio 降低颜色扰动，保留浅蓝/浅灰文字对比；不随机截断单元格。
    steps = [ColorJitterTransform()] if is_train else []
    return transforms.Compose(steps + [ResizeAndPad(size, padding_color="auto"), GenerateGT(size), ToTensor()])
