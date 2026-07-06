"""白色文字/有色背景裁剪工具。"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from recognition.components.dataset import InferenceDataset, remove_red_seal

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
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w <= 1 or h <= 1:
        return image

    hsv = np.array(image.convert("HSV"))
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_mask = (value >= white_value_thresh) & (saturation <= white_saturation_max)
    fg_ratio = float(white_mask.mean())

    # 太少：不像文字；太多：可能是白底图或大面积高亮背景。
    if fg_ratio < min_fg_ratio or fg_ratio > max_fg_ratio:
        return image

    ys, xs = np.where(white_mask)
    if xs.size == 0 or ys.size == 0:
        return image

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
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
    推理专用 Dataset：读取 RGB 图后先做白字裁剪，再复用原 resize/normalize 规则。
    """

    def __init__(
        self,
        image_paths: list,
        img_height: int,
        num_channels: int,
        crop_kwargs: dict | None = None,
        save_crops_dir: str | None = None,
    ):
        super().__init__(image_paths=image_paths, img_height=img_height, num_channels=num_channels)
        self.crop_kwargs = crop_kwargs or {}
        self.save_crops_dir = save_crops_dir

    def _load_and_process_image(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert("RGB")
            image = auto_crop_white_text_image(image, **self.crop_kwargs)
            if self.save_crops_dir:
                save_crop_debug_image(image, image_path, self.save_crops_dir)
            image = remove_red_seal(image)

            if self.num_channels == 1:
                image = image.convert("L")

            w, h = image.size
            target_h = self.img_height
            scale = target_h / h
            new_w = int(w * scale)

            min_w_from_aspect = int(w * 0.5)
            if new_w < min_w_from_aspect:
                new_w = min(min_w_from_aspect, 800)

            image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)

            min_width = 48
            if new_w < min_width:
                if self.num_channels == 1:
                    canvas = Image.new("L", (min_width, target_h), 255)
                else:
                    canvas = Image.new("RGB", (min_width, target_h), (255, 255, 255))
                offset_x = (min_width - new_w) // 2
                canvas.paste(image, (offset_x, 0))
                image = canvas

            return self.transform(image)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.ones((self.num_channels, self.img_height, 8))

