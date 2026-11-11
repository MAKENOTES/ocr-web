"""小图检测的训练数据增强接线和真值生成。

推理预处理（尺寸校验、缩放补边、归一化和坐标还原）在 preprocessing.py；
本模块只保留训练专用的增强、真值和张量化，并再导出 `ResizeAndPad` 以兼容
历史根目录推理入口的导入路径。
"""
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from .augmentation import get_geometry_augmenter, get_pixel_augmenter
from .ground_truth import generate_ground_truth, make_training_mask
from .preprocessing import (
    NORMALIZE,
    ResizeAndPad,
    normalize_input_size,
)

__all__ = ["GenerateGT", "ResizeAndPad", "ToTensor", "get_transforms"]


class GenerateGT:
    """按输入尺寸生成置信度、几何真值及忽略掩码。"""

    def __init__(self, input_size, score_map_scale=0.25):
        self.input_size = normalize_input_size(input_size)
        self.score_map_scale = score_map_scale

    def __call__(self, sample):
        """把四边形写入真值图；### 框只进入忽略掩码，不产生正样本。"""
        quads = np.asarray(sample["quads"], dtype=np.float32)
        ignored = np.asarray(sample.get("ignored", np.zeros(len(quads), dtype=bool)), dtype=bool)
        sample["score_map"], sample["geo_map"] = generate_ground_truth(
            self.input_size, quads[~ignored], self.score_map_scale)
        sample["training_mask"] = make_training_mask(
            self.input_size, quads[ignored], self.score_map_scale)
        return sample


class ToTensor:
    """转换为归一化图片和训练所需的三张 float32 张量。"""

    def __call__(self, sample):
        return {
            "image": NORMALIZE(F.to_tensor(sample["image"])),
            **{key: torch.from_numpy(sample[key]).float()
               for key in ("score_map", "geo_map", "training_mask")},
        }


def get_transforms(is_train=True, input_size=(256, 768), stage_weights=None):
    """构建训练或验证变换。

    训练时几何增强在原图执行，光度/遮挡增强在缩放补边后的固定输入上执行；
    验证只做确定性缩放和归一化。`stage_weights` 为空时使用标准增强。
    """
    size = normalize_input_size(input_size)
    steps = [get_geometry_augmenter(stage_weights)] if is_train else []
    steps.append(ResizeAndPad(size, padding_color="auto"))
    if is_train:
        steps.append(get_pixel_augmenter(stage_weights))
    steps += [GenerateGT(size), ToTensor()]
    return transforms.Compose(steps)
