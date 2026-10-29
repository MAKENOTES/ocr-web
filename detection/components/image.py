"""图片读取与边缘背景色估计，训练和推理共用。"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}  # 支持读取的图片扩展名，枚举前统一转小写。


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """读取图片、应用 EXIF 方向并合成为 RGB。

    有透明通道时按可见前景亮度选择黑或白背景，避免直接丢弃 alpha 后
    黑字或白字消失；完全透明图使用白底。EXIF 方向在合成前应用，
    确保训练张量和推理坐标使用同一套像素方向。
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

        # 完全透明图没有可见前景，统一使用白底，避免输出全黑样本。
        if alpha_max == 0:
            background_value = 255
        else:
            visible_luminance = ImageStat.Stat(rgba.convert("L"), mask=alpha).mean[0]
            background_value = 0 if visible_luminance >= 128 else 255

        background = Image.new("RGBA", rgba.size,
                               (background_value, background_value, background_value, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def border_fill_color(image: Image.Image):
    """用四周边缘像素的通道中位数估计补边背景色，兼容灰度输入。"""
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
