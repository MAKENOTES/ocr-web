"""识别数据预处理：路径校验、RGB 合成、去红章与定高缩放。

训练、验证和推理共享这些确定性步骤。随机增强见 augmentation.py，
推理数据集见 inference_data.py，训练数据集见 dataset.py。
"""

# 本文件由 tools/sync_recognition_runtime.py 从 recognition/components/preprocessing.py 生成。
# 请勿手改；修改源文件后重新运行同步脚本。
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps, ImageStat

# 允许 Pillow 加载部分截断图片；这是进程级开关，不保证损坏图片完整可用。
ImageFile.LOAD_TRUNCATED_IMAGES = True

MIN_IMAGE_WIDTH = 48  # 缩放后窄图的背景画布宽度，单位 px；显式 max_width 更小时以上限为准。

# 去章时按平均正文列密度缩放的正文支持下限。细字和末行密度偏低，0.4 会把正常叠印
# 误判为无正文支持，0.35 在保持"红字旁边没有黑字就不处理"的前提下放行这类样本。
MIN_DARK_COLUMN_SUPPORT = 0.35
# 完全落在正文行内的红色区域所需的正文列支持。同行红字通常位于字间空白，支持接近 0；
# 被黑字穿过的章字支持很高，因此该阈值明显高于行外的 0.15，只放行有充分正文证据的碎片。
IN_BAND_DARK_SUPPORT = 0.4
# 行内红色碎片判定为章字残段的最大高度，按正文行高计。完整红字的高度接近行高，
# 被正文行切开的章字残段则明显更矮，两者由此区分，避免把同行红字一并删掉。
IN_BAND_FRAGMENT_HEIGHT = 0.35


def resolve_image_path(image_path: str | os.PathLike) -> str:
    """返回实际存在的图片路径；标注中的文件名和扩展名必须准确。"""
    path = Path(image_path)
    if path.is_file():
        return str(path)

    raise FileNotFoundError(f"图片不存在: {path}")


def load_rgb_image(image_path: str | os.PathLike) -> Image.Image:
    """读取图片、应用 EXIF 方向并合成为 RGB。

    有透明通道时，按可见前景亮度选择黑或白背景，避免直接丢弃 alpha。
    完全透明图使用白底；混合明暗前景仍需人工核对可见性，不保证恢复被遮挡内容。"""
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

        if alpha_max == 0:
            background_value = 255
        else:
            visible_luminance = ImageStat.Stat(rgba.convert("L"), mask=alpha).mean[0]
            background_value = 0 if visible_luminance >= 128 else 255

        background = Image.new(
            "RGBA",
            rgba.size,
            (background_value, background_value, background_value, 255),
        )
        return Image.alpha_composite(background, rgba).convert("RGB")


def border_fill_color(image: Image.Image):
    """用四周边缘的通道中位数估计背景色，供窄图补边和几何增强填充。"""
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        border = np.concatenate(
            (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
        )
        return int(np.median(border))

    channels = pixels.shape[2]
    border = np.concatenate(
        (
            pixels[0, :, :].reshape(-1, channels),
            pixels[-1, :, :].reshape(-1, channels),
            pixels[:, 0, :].reshape(-1, channels),
            pixels[:, -1, :].reshape(-1, channels),
        ),
        axis=0,
    )
    return tuple(int(value) for value in np.median(border, axis=0))


def _is_red_seal_arc(component: np.ndarray, text_height: float) -> bool:
    """识别 detection 紧框内的连续弯曲章边；直线、短笔画和杂乱红字不通过。"""
    for oriented in (component, component.T):
        columns = np.flatnonzero(oriented.any(axis=0))
        span = columns[-1] - columns[0] + 1
        if span < 2 * text_height or columns.size < 0.9 * span:
            continue
        thickness = oriented[:, columns].sum(axis=0)
        if np.quantile(thickness, 0.9) > 0.6 * text_height:
            continue
        centerline = np.array([
            np.median(np.flatnonzero(oriented[:, column])) for column in columns
        ])
        # 局部圆弧用二次曲线近似；要求明显曲率，排除红色下划线/表格边框。
        coefficients = np.polyfit(columns, centerline, 2)
        if abs(coefficients[0]) * (span - 1) ** 2 < 0.5 * text_height:
            continue
        residual = np.abs(np.polyval(coefficients, columns) - centerline)
        if np.quantile(residual, 0.9) <= max(1.5, 0.12 * text_height):
            return True
    return False


def _is_single_seal_component(
    component: np.ndarray, text_height: float, *, allow_narrow: bool = False,
) -> bool:
    """单个连通区域需有弧线、椭圆或复杂章形，避免把直线/简单边框当章。

    allow_narrow 只放宽宽度判定，供"比行高还窄的组"使用：这类组按定义宽度不足，
    但高度足够、且轮廓成弧的是被正文行切开的章边。"""
    height, width = component.shape
    if height < 0.8 * text_height or (not allow_narrow and width < 0.8 * text_height):
        return False
    if _is_red_seal_arc(component, text_height):
        return True
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 12:
        return False
    # 星形、连在一起的章字无需拟合成圆；矩形框和简单折线继续保留。
    polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(polygon) >= 9:
        return True
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    if (
        not np.isfinite([cx, cy, axis_a, axis_b, angle]).all()
        or min(axis_a, axis_b) < text_height
        or max(axis_a, axis_b) > 3 * min(axis_a, axis_b)
        or component.mean() > 0.45
    ):
        return False
    points = contour[:, 0, :].astype(np.float64) - (cx, cy)
    radians = np.deg2rad(angle)
    u = points[:, 0] * np.cos(radians) + points[:, 1] * np.sin(radians)
    v = -points[:, 0] * np.sin(radians) + points[:, 1] * np.cos(radians)
    radius = np.sqrt((2 * u / axis_a) ** 2 + (2 * v / axis_b) ** 2)
    return bool(np.quantile(np.abs(radius - 1), 0.9) <= 0.12)


def remove_red_seal(image: Image.Image) -> Image.Image:
    """去除浅色纸面上与深色正文交错叠印的红章，包括被紧框截断的章字。

    按局部红色区域检查正文穿插和上下错位，不再要求每个碎片都有完整章轮廓。
    普通同行红字所在列通常没有黑色正文，或与正文行高对齐，因此保留。
    区域通过后仍逐个检查连通区域的正文支持，避免连带删除附近合法红字。
    只替换候选区域内的红像素，不膨胀掩码，以保护相邻的深色笔画。
    输入必须是 load_rgb_image 已完成透明背景合成的 RGB 图片。
    """
    if image.mode != "RGB":
        raise ValueError(f"remove_red_seal requires RGB input, got {image.mode}")

    pixels = np.asarray(image)
    # 先转有符号整数再求色差，避免 uint8 减法溢出把蓝字误判成红色。
    rgb = pixels.astype(np.int16)
    red, green, blue = rgb.transpose(2, 0, 1)
    red_mask = (
        (red >= 150) & (red - green >= 25) & (red - blue >= 25)
        & (np.abs(green - blue) <= 40)
    )
    if not red_mask.any() or red_mask.mean() > 0.35:
        return image

    # 兼容浅灰、轻微偏色和带扫描颗粒的纸面；红底白字、暗底仍不处理。
    channel_range = rgb.max(axis=2) - rgb.min(axis=2)
    paper_mask = (rgb.min(axis=2) >= 190) & (channel_range <= 35)
    paper_border = np.concatenate((
        paper_mask[0], paper_mask[-1], paper_mask[:, 0], paper_mask[:, -1],
    ))
    if paper_mask.mean() < 0.4 or paper_border.mean() < 0.5:
        return image

    # 灰黑正文也能提供证据，但红色候选不能同时充当“黑字”，避免自证为章。
    dark_mask = (rgb.max(axis=2) <= 170) & (channel_range <= 35) & ~red_mask
    dark_y = np.nonzero(dark_mask)[0]
    # 大章/星形可能比正文墨迹更多，不再要求黑像素数量超过全部红像素。
    if dark_y.size < max(16, red_mask.sum() * 0.15):
        return image

    # 用正文墨迹的中间 96% 估计行高，减少孤立噪点影响。
    text_top, text_bottom = np.quantile(dark_y, [0.02, 0.98])
    text_height = text_bottom - text_top + 1
    if text_height < 6:
        return image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        red_mask.astype(np.uint8), connectivity=8,
    )
    erase_mask = np.zeros(red_mask.shape, dtype=bool)
    band = slice(int(text_top), int(text_bottom) + 1)
    dark_columns = dark_mask[band].sum(axis=0) >= max(2, 0.08 * text_height)
    # 细字体的有墨迹列本来就少，按正文列密度调整最低支持率。
    # 保留绝对下限，避免几粒黑噪点让整段红字被当作叠印。
    min_overlap = max(0.12, MIN_DARK_COLUMN_SUPPORT * dark_columns.mean())
    red_columns = np.flatnonzero(red_mask.any(axis=0))
    # 章边、章字、中央星形之间常隔着一两个字宽，需作为局部整体判断。
    # 这里只分组，绝不对待删除像素做闭运算/膨胀；组内仍逐个连通区域复核。
    groups = np.split(
        red_columns,
        np.flatnonzero(np.diff(red_columns) > max(2, 2 * text_height)) + 1,
    )
    for columns in groups:
        left, right = columns[0], columns[-1] + 1
        region = red_mask[:, left:right]
        if region.sum() < 16:
            continue
        active_columns = columns[red_mask[:, columns].sum(axis=0) >= 2]
        if active_columns.size == 0 or dark_columns[active_columns].mean() < min_overlap:
            continue
        in_band_ratio = region[band].sum() / region.sum()
        # 同行红字/描边红字与黑字通常对齐；章既进入正文行，又有越界部分。
        # 允许只越过上边或下边，适配只剩半个章的 detection 框。
        if not 0.15 <= in_band_ratio <= 0.92:
            continue
        component_ids = np.unique(labels[:, left:right][region])
        component_ids = [
            int(component_id) for component_id in component_ids
            if stats[component_id, cv2.CC_STAT_AREA] >= max(4, 0.15 * text_height)
        ]
        # 比一个行高还窄的组只放行本身像章的连通区域：穿过单个高字的章弧宽度
        # 常常小于行高，而真正的短红字没有弧线或椭圆轮廓。
        if right - left < text_height and not any(
            _is_single_seal_component(
                labels[
                    stats[component_id, cv2.CC_STAT_TOP]:
                    stats[component_id, cv2.CC_STAT_TOP] + stats[component_id, cv2.CC_STAT_HEIGHT],
                    stats[component_id, cv2.CC_STAT_LEFT]:
                    stats[component_id, cv2.CC_STAT_LEFT] + stats[component_id, cv2.CC_STAT_WIDTH],
                ] == component_id,
                text_height,
                allow_narrow=True,
            )
            for component_id in component_ids
        ):
            continue
        for component_id in component_ids:
            x, y, width, height, _ = stats[component_id]
            component = labels[y:y + height, x:x + width] == component_id
            # 区域内的小红字仍须独立检查，不能因旁边有章就整片抹红。
            component_columns = np.flatnonzero(component.any(axis=0)) + x
            support = dark_columns[component_columns].mean()
            if y >= text_top and y + height - 1 <= text_bottom:
                # 完整红字的高度接近行高；只有明显更矮、且被黑字穿过的碎片才是章字残段。
                if (
                    height >= IN_BAND_FRAGMENT_HEIGHT * text_height
                    or support < IN_BAND_DARK_SUPPORT
                ):
                    continue
            elif len(component_ids) == 1 and not _is_single_seal_component(component, text_height):
                continue
            if support < 0.15:
                continue
            erase_mask[y:y + height, x:x + width] |= component

    if not erase_mask.any():
        return image
    result = pixels.copy()
    # 用纸面中位色替换，避免浅灰底出现纯白斑；不改任何非候选像素。
    result[erase_mask] = np.median(pixels[paper_mask], axis=0).astype(np.uint8)
    return Image.fromarray(result)


def resize_detection_image(
    image: Image.Image,
    target_height: int,
    max_width: int | None = None,
    min_width: int = MIN_IMAGE_WIDTH,
) -> Image.Image:
    """固定高度缩放完整文字框，不裁剪正文。
    优先保持宽高比；超过 max_width 时横向压缩。窄于 min_width 时居中补背景。"""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"非法图片尺寸: {image.size}")
    if target_height <= 0:
        raise ValueError(f"target_height 必须大于 0，当前为 {target_height}")

    resized_width = max(1, int(round(width * target_height / height)))
    if max_width is not None:
        resized_width = min(resized_width, max_width)

    image = image.resize((resized_width, target_height), Image.Resampling.LANCZOS)

    # 单字/符号 detection 框可能很窄。只补背景，不拉伸字符形状。
    if resized_width < min_width:
        canvas_width = min_width if max_width is None else min(min_width, max_width)
        fill = border_fill_color(image)
        canvas = Image.new(image.mode, (canvas_width, target_height), fill)
        offset_x = (canvas_width - resized_width) // 2
        canvas.paste(image, (offset_x, 0))
        image = canvas

    return image
