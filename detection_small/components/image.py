"""小图检测的图片读取与边缘背景色估计，训练、评估和报告共用。"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")  # 小图检测默认支持的图片扩展名；训练入口只使用 PNG。


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """读取图片、应用 EXIF 方向并合成为 RGB。

    UI 截图可能带透明通道。直接 `convert("RGB")` 会丢弃 alpha，黑字或白字
    可能直接消失；这里按可见前景亮度选择黑或白背景，完全透明图统一使用白底。
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        has_transparency = "A" in image.getbands() or "transparency" in source.info
        if not has_transparency:
            return image.convert("RGB").copy()

        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        if alpha_min == 255:
            return rgba.convert("RGB")

        # 完全透明图没有可见前景，统一使用白底，避免得到全黑样本。
        if alpha_max == 0:
            background_value = 255
        else:
            visible_luminance = ImageStat.Stat(rgba.convert("L"), mask=alpha).mean[0]
            background_value = 0 if visible_luminance >= 128 else 255

        background = Image.new("RGBA", rgba.size,
                               (background_value, background_value, background_value, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def border_fill_color(image: Image.Image):
    """用四周边缘像素的通道中位数估计背景色，兼容灰度输入。"""
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        border = np.concatenate((pixels[0, :], pixels[-1, :],
                                 pixels[:, 0], pixels[:, -1]), axis=0)
        return int(np.median(border))

    channels = pixels.shape[2]
    border = np.concatenate((
        pixels[0, :, :].reshape(-1, channels),
        pixels[-1, :, :].reshape(-1, channels),
        pixels[:, 0, :].reshape(-1, channels),
        pixels[:, -1, :].reshape(-1, channels),
    ), axis=0)
    return tuple(int(value) for value in np.median(border, axis=0))
