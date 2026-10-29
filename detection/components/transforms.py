"""统一检测的缩放补边、真值生成和 ImageNet 归一化，并接入训练数据增强。

训练使用固定输入尺寸保证 batch 张量可直接堆叠；推理可调用
`dynamic_input_size` 按原图宽高比选择 32 的倍数尺寸，减少补边计算。
"""
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

from detection.components.augmentation import get_geometry_augmenter, get_pixel_augmenter
from detection.components.ground_truth import generate_ground_truth, make_training_mask
from detection.components.image import border_fill_color


STRIDE = 4  # 网络输出相对输入的步长，每个特征图像素对应 4 个输入像素。
INPUT_ALIGN = 32  # ResNet 下采样总倍数；输入高宽必须是它的整数倍。
MIN_INPUT_SIDE = 64  # 单边最小输入像素，保证四级下采样后仍有特征图。
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 按 RGB 通道应用 ImageNet 均值与标准差。


def normalize_input_size(input_size):
    """返回 (高, 宽) 像素尺寸；各边至少 64 且为 32 的倍数。"""
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if (not isinstance(input_size, (tuple, list)) or len(input_size) != 2
            or any(not isinstance(value, int) or value < MIN_INPUT_SIDE or value % INPUT_ALIGN
                   for value in input_size)):
        raise ValueError("input_size must contain height/width >= 64 divisible by 32")
    return tuple(input_size)


def _align_up(value, limit):
    """把缩放后的边长向上取整到 32 的倍数，并限制在最大尺寸内。"""
    aligned = int(np.ceil(value / INPUT_ALIGN)) * INPUT_ALIGN
    return min(limit, max(MIN_INPUT_SIDE, aligned))


def dynamic_input_size(image_size, max_size):
    """按原图宽高比计算推理输入尺寸，返回 (高, 宽)。

    最长边不超过 max_size 的对应边长；每条边向上取整到 32 的倍数，
    最小 64。极端细长图仍会在较短的边上补背景，但不再固定补齐成正方形。
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive: {image_size!r}")
    max_height, max_width = normalize_input_size(max_size)
    scale = min(max_width / width, max_height / height)
    return (_align_up(height * scale, max_height),
            _align_up(width * scale, max_width))


class ResizeAndPad:
    """保持宽高比缩放到目标尺寸内，再用图像边缘背景色居中补边。"""

    def __init__(self, output_size, padding_color="auto"):
        """output_size 为 int 或 (高, 宽)；auto 使用边缘像素中位数作为背景色。"""
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
            raise ValueError(f"output_size must be an int or (height, width): {output_size!r}")
        self.output_size = tuple(int(value) for value in output_size)
        self.padding_color = padding_color

    def __call__(self, sample):
        """缩放并补边图片，同时把四边形坐标映射到目标尺寸。"""
        image, quads = sample["image"], sample["quads"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        target_height, target_width = self.output_size

        scale = min(target_width / width, target_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized_image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

        padding_color = (border_fill_color(image) if self.padding_color == "auto"
                         else self.padding_color)
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        new_image = Image.new("RGB", (target_width, target_height), padding_color)
        new_image.paste(resized_image, (offset_x, offset_y))

        if quads.size > 0:
            quads = quads * scale
            quads[:, :, 0] += offset_x
            quads[:, :, 1] += offset_y

        sample["image"] = new_image
        sample["quads"] = quads
        return sample


def resize_metadata(image_size, input_size):
    """返回 (缩放比例, (水平偏移, 垂直偏移))，坐标单位为输入图像素。"""
    target_height, target_width = normalize_input_size(input_size)
    original_width, original_height = image_size
    scale = min(target_width / original_width, target_height / original_height)
    # 与 ResizeAndPad 的向下取整及居中偏移完全一致。
    offset = ((target_width - int(original_width * scale)) // 2,
              (target_height - int(original_height * scale)) // 2)
    return scale, offset


def prepare_image(image, input_size, dynamic=False):
    """按固定尺寸或按宽高比动态选择尺寸预处理整图，返回张量和坐标还原参数。

    dynamic=False 时严格使用 input_size；dynamic=True 时把它作为最大尺寸。
    """
    size = dynamic_input_size(image.size, input_size) if dynamic else normalize_input_size(input_size)
    sample = ResizeAndPad(size, padding_color="auto")({
        "image": image, "quads": np.empty((0, 4, 2), dtype=np.float32),
    })
    return NORMALIZE(F.to_tensor(sample["image"])), resize_metadata(image.size, size)


def restore_boxes(boxes, metadata, image_size):
    """将输入图上的 xyxy 框减去居中偏移、除以缩放比例，并裁到原图边界。"""
    scale, (offset_x, offset_y) = metadata
    result = boxes.copy()
    result[:, [0, 2]] = np.clip((result[:, [0, 2]] - offset_x) / scale, 0, image_size[0])
    result[:, [1, 3]] = np.clip((result[:, [1, 3]] - offset_y) / scale, 0, image_size[1])
    return result


class GenerateGT:
    """按显式输入尺寸生成置信度、几何真值及忽略掩码。"""

    def __init__(self, input_size):
        self.input_size = normalize_input_size(input_size)

    def __call__(self, sample):
        quads = np.asarray(sample["quads"], dtype=np.float32)
        ignored = np.asarray(sample["ignored"], dtype=bool)
        sample["score_map"], sample["geo_map"] = generate_ground_truth(
            self.input_size, quads[~ignored])
        sample["training_mask"] = make_training_mask(self.input_size, quads[ignored])
        return sample


class ToTensor:
    """转换为归一化图片和训练所需的三张 float32 张量。"""

    def __call__(self, sample):
        return {
            "image": NORMALIZE(F.to_tensor(sample["image"])),
            **{key: torch.from_numpy(sample[key]).float()
               for key in ("score_map", "geo_map", "training_mask")},
        }


def get_transforms(is_train, input_size, stage_weights=None):
    """构建训练或验证变换。

    训练时几何增强在原图执行，光度/遮挡增强在缩放补边后的固定输入上执行；
    验证只做确定性缩放和归一化。
    """
    size = normalize_input_size(input_size)
    steps = [get_geometry_augmenter(stage_weights)] if is_train else []
    steps.append(ResizeAndPad(size, padding_color="auto"))
    if is_train:
        steps.append(get_pixel_augmenter(stage_weights))
    steps += [GenerateGT(size), ToTensor()]
    return transforms.Compose(steps)
