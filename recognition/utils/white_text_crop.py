"""白色文字/有色背景裁剪工具。"""

from pathlib import Path

import numpy as np
from PIL import Image

from recognition.components.dataset import InferenceDataset


def find_white_text_bbox(
    image: Image.Image,
    *,
    min_fg_ratio: float = 0.001,
    max_fg_ratio: float = 0.45,
    white_value_thresh: int = 185,
    white_saturation_max: int = 90,
) -> tuple[int, int, int, int] | None:
    """
    检测“有色背景 + 白色文字”图中的白色像素 bbox。

    返回:
        (left, top, right, bottom)，其中 right/bottom 为 PIL crop 风格的开区间；
        如果白色像素占比不可信或没有检测到白字，则返回 None。
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w <= 1 or h <= 1:
        return None

    hsv = np.array(image.convert("HSV"))
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_mask = (value >= white_value_thresh) & (saturation <= white_saturation_max)
    fg_ratio = float(white_mask.mean())

    # 太少：不像文字；太多：可能是白底图或大面积高亮背景。
    if fg_ratio < min_fg_ratio or fg_ratio > max_fg_ratio:
        return None

    ys, xs = np.where(white_mask)
    if xs.size == 0 or ys.size == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def auto_crop_white_text_image(
    image: Image.Image,
    *,
    margin_ratio: float = 0.35,
    margin_x_ratio: float | None = 0.05,
    margin_y_ratio: float | None = 0.3,
    min_margin: int = 2,
    min_fg_ratio: float = 0.001,
    max_fg_ratio: float = 0.45,
    max_text_height_ratio: float = 0.55,
    white_value_thresh: int = 185,
    white_saturation_max: int = 90,
) -> Image.Image:
    """
    针对“有色背景 + 白色文字”的短期裁剪。

    逻辑：
      1. 用 HSV 找高亮、低饱和像素，近似白色文字；
      2. 根据白色像素 bbox 裁剪；
      3. bbox 置信不足或疑似整图白底时，返回原图，避免误裁。

    参数：
        image: 输入图片。建议传 RGB 图片；如果不是 RGB，会在函数内部转换为 RGB。
        margin_ratio: 兼容参数。当 margin_x_ratio 或 margin_y_ratio 为 None 时，
            使用该比例作为对应方向的 bbox 外扩比例。
        margin_x_ratio: 横向外扩比例。基于检测到的白字 bbox 宽度计算，
            默认 0.05，适合菜单项这类左右背景不宜保留太多的场景。
        margin_y_ratio: 纵向外扩比例。基于检测到的白字 bbox 高度计算，
            默认 0.3，用于保留适量上下背景，避免文字被裁得过紧。
        min_margin: 最小外扩像素数，防止 bbox 太小时外扩为 0。
        min_fg_ratio: 白色像素最小占比。低于该值认为白字区域不可靠，不裁剪。
        max_fg_ratio: 白色像素最大占比。高于该值通常是白底图、大面积高亮背景
            或非目标场景，不裁剪。
        max_text_height_ratio: 白字 bbox 高度占整图高度的最大比例。超过该值说明
            图片基本已经是目标区域，避免二次紧裁。
        white_value_thresh: HSV 中 V 通道阈值，越高越只接受更亮的白色像素。
        white_saturation_max: HSV 中 S 通道最大值，越低越只接受低饱和的白/灰像素。

    返回：
        裁剪后的 PIL Image；如果不满足裁剪条件，则返回原图对象或 RGB 转换后的图片。
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w <= 1 or h <= 1:
        return image

    bbox = find_white_text_bbox(
        image,
        min_fg_ratio=min_fg_ratio,
        max_fg_ratio=max_fg_ratio,
        white_value_thresh=white_value_thresh,
        white_saturation_max=white_saturation_max,
    )
    if bbox is None:
        return image

    left, top, right, bottom = bbox
    box_w = right - left
    box_h = bottom - top

    # 白色区域过小，通常是噪声或图标碎片。
    if box_w < 3 or box_h < 3:
        return image

    # 文字高度已经足够大时，说明图片基本已经是目标区域，避免二次紧裁。
    if (box_h / h) >= max_text_height_ratio:
        return image

    effective_margin_x_ratio = margin_ratio if margin_x_ratio is None else margin_x_ratio
    effective_margin_y_ratio = margin_ratio if margin_y_ratio is None else margin_y_ratio
    margin_x = max(min_margin, int(round(box_w * effective_margin_x_ratio)))
    margin_y = max(min_margin, int(round(box_h * effective_margin_y_ratio)))

    left = max(0, left - margin_x)
    top = max(0, top - margin_y)
    right = min(w, right + margin_x)
    bottom = min(h, bottom + margin_y)

    crop_w = right - left
    crop_h = bottom - top

    # 如果几乎没裁掉，保留原图，避免无意义处理。
    if crop_w >= int(w * 0.98) and crop_h >= int(h * 0.98):
        return image

    return image.crop((left, top, right, bottom))


def save_crop_debug_image(image: Image.Image, source_path: str, output_dir: str) -> Path:
    """保存裁剪后的 RGB 图，便于人工核查裁剪是否正确。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_stem = Path(source_path).stem
    saved_path = output_path / f"{source_stem}.png"
    image.save(saved_path)
    return saved_path


class WhiteTextCropInferenceDataset(InferenceDataset):
    """
    推理专用 Dataset：使用与训练/验证一致的白字裁剪预处理。
    """

    def __init__(
        self,
        image_paths: list,
        img_height: int,
        num_channels: int,
        crop_kwargs: dict | None = None,
        save_crops_dir: str | None = None,
    ):
        super().__init__(
            image_paths=image_paths,
            img_height=img_height,
            num_channels=num_channels,
            use_auto_crop=True,
            crop_kwargs=crop_kwargs,
            save_crops_dir=save_crops_dir,
        )
