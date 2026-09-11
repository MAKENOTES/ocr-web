"""整图训练与推理复用原检测的居中等比缩放、补边和 ImageNet 归一化。"""
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from detection.components.transforms import ResizeAndPad, ColorJitterTransform
from detection_larger.components.ground_truth import generate_ground_truth, make_training_mask

STRIDE = 4
DEFAULT_INPUT_SIZE = 1024  # 默认整图输入边长，单位 px；与训练入口的初始配置一致
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


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


def prepare_image(image, input_size=DEFAULT_INPUT_SIZE):
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


def get_transforms(is_train=True, input_size=DEFAULT_INPUT_SIZE):
    size = normalize_input_size(input_size)
    # 相对 majorbio 降低颜色扰动，保留浅蓝/浅灰文字对比；不随机截断单元格。
    steps = [ColorJitterTransform()] if is_train else []
    return transforms.Compose(steps + [ResizeAndPad(size, padding_color="auto"), GenerateGT(size), ToTensor()])
