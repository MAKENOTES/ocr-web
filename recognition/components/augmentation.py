"""训练期数据增强：缩放前的源分辨率几何扰动与缩放后的光度扰动。

图片加载、去章和缩放见 preprocessing.py，数据集与补边见 dataset.py；
本模块只提供随机增强。RandomSourceGeometry 在缩放前改写检测框，
build_transform 返回缩放后的 torchvision 变换。验证和推理不启用随机增强。
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from recognition.components.inference_data import build_eval_transform
from recognition.components.preprocessing import border_fill_color

# ---- 源分辨率几何增强 ----

INK_THRESHOLD = 40  # 与背景中位色的最大通道差；超过该值视为墨迹，用于重裁外接框。
# 重裁后每边至少保留的背景宽度，单位 px。零留白框会让后续按四周边缘估计纸面色的
# 环节（窄图补边）把墨色当成背景色，因此几何增强始终保留 1 px 背景。
MIN_CROP_MARGIN = 1
# 旋转带来的垂直跨度上限，按文字高度的倍数计。窄而高的框可按构造角度旋转；
# 长行旋转同样的角度会让竖直跨度远超文字高度，重裁后文字被缩小数倍，因此按宽高比收紧。
MAX_ROTATED_SPAN = 0.3
# 水平斜切系数上限，x 方向位移 = 系数 × 行高。页面上 2° 倾斜在 40 px 高的行里
# 约等于 0.035，取 0.04 覆盖轻微倾斜，避免长行两端被拉出画布。
SHEAR_LIMIT = 0.04
# 透视四个角点的位移上限，按图片高度计（不按宽度），长行两端不会被拉变形。
PERSPECTIVE_LIMIT = 0.04
# 弹性扭曲的位移上限，单位 px。作用在源分辨率，矮框里相当于明显抖动，
# 长行则是缓慢起伏；不做逐像素随机位移，避免笔画被撕裂。
ELASTIC_ALPHA = 2.0
# 边缘轻裁切入正文时允许损失的墨迹比例上限；超过则退回只裁留白。
MAX_TRIM_INK_FRACTION = 0.08
# 缩放使用的高质量重采样核，覆盖不同软件/驱动产生的笔画粗细和抗锯齿差异；
# 不含 NEAREST，最近邻会把细笔画打成断点，超出正常渲染的失真范围。
RESAMPLE_FILTERS = (
    Image.Resampling.LANCZOS,
    Image.Resampling.BICUBIC,
    Image.Resampling.BILINEAR,
)
# PIL 的 Image.rotate 不接受 LANCZOS（1）；旋转时退到 BICUBIC，两者在轻微旋转下
# 的差异远小于重采样核本身的差异。
ROTATION_RESAMPLE = {
    Image.Resampling.LANCZOS: Image.Resampling.BICUBIC,
    Image.Resampling.BICUBIC: Image.Resampling.BICUBIC,
    Image.Resampling.BILINEAR: Image.Resampling.BILINEAR,
}

# ---- 缩放后光度增强 ----

COLOR_JITTER_PROBABILITY = 0.7  # 训练时触发颜色扰动的概率。
COLOR_JITTER_BRIGHTNESS = 0.2  # 亮度抖动幅度，相对 ToTensor 前的 [0, 1] 取值。
COLOR_JITTER_CONTRAST = 0.25  # 对比度抖动幅度，模拟浅字和灰底扫描。
COLOR_JITTER_SATURATION = 0.35  # 饱和度抖动幅度，模拟彩字与背景的浓淡变化。
COLOR_JITTER_HUE = 0.03  # 色相抖动幅度，覆盖轻微白平衡和设备偏色。
AUTOCONTRAST_PROBABILITY = 0.2  # 训练时触发自动对比度的概率。
INVERT_PROBABILITY = 0.08  # 训练时触发反色的概率，覆盖深色主题和黑底白字。
BLUR_PROBABILITY = 0.15  # 训练时触发高斯模糊的概率。
BLUR_KERNEL_SIZE = 3  # 高斯模糊核大小，单位 px，必须为奇数。
BLUR_SIGMA = (0.1, 0.8)  # 高斯模糊 sigma 的随机范围，单位 px。
NOISE_PROBABILITY = 0.15  # 训练时叠加轻噪声的概率。
NOISE_MEAN = 0.0  # 噪声均值，作用在 [0, 1] 浮点张量上。
NOISE_STD = 0.015  # 噪声标准差，作用在 [0, 1] 浮点张量上。
ERASING_PROBABILITY = 0.05  # 训练时触发小块擦除的概率。
ERASING_SCALE = (0.005, 0.03)  # 擦除面积占整图的比例范围。
ERASING_RATIO = (0.3, 3.3)  # 擦除矩形的宽高比范围。


def ink_extent(image: Image.Image, fill) -> tuple[int, int, int, int]:
    """返回非背景像素的外接框 (left, top, right, bottom)，右、下不含。

    fill 是该图的背景色，通常来自 border_fill_color；没有明显墨迹时返回整图。"""
    pixels = np.asarray(image).astype(np.int16)
    difference = np.abs(pixels - np.asarray(fill, dtype=np.int16)).max(axis=-1)
    ys, xs = np.nonzero(difference > INK_THRESHOLD)
    if ys.size == 0:
        return 0, 0, image.width, image.height
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def limited_rotation_degrees(width: int, height: int, degrees: float) -> float:
    """按宽高比收紧旋转角度：旋转造成的垂直跨度不超过文字高度的 MAX_ROTATED_SPAN。

    训练增强和鲁棒性评估必须使用同一规则，否则评估条件会超出训练覆盖范围。"""
    if width <= 0 or height <= 0 or degrees <= 0:
        return 0.0
    return min(degrees, math.degrees(math.atan(MAX_ROTATED_SPAN * height / width)))


class RandomSourceGeometry:
    """训练期在源分辨率扰动检测框的几何形状，保持标签文字完整。

    检测框的松紧决定文字占画布的高度比例，字形宽高比和页面倾斜决定笔画形态；
    这些维度在当前训练中没有其它增强覆盖。缩放作用于内容并重裁，因此会改变文字
    占画布的比例，而不是被后续的定高缩放抵消。

    旋转、斜切和透视同属页面倾斜，每次最多施加一种，避免叠加出正常文档不会出现的
    畸变；旋转在扩大的画布上完成，字形不会被画布边缘截断，角度按宽高比收紧。
    所有新增边缘用图片四周的中位色填充，兼容白底、深色底和彩色底。"""

    def __init__(
        self,
        scale=(0.85, 1.18),  # 内容等比缩放范围；改变文字占画布的比例和重采样密度。
        stretch=(0.90, 1.10),  # 只作用于宽度的缩放范围；改变字形宽高比。
        pad_px=3,  # 重裁后每边随机留白的上限，单位 px；0 表示不补边。
        degrees=2.0,  # 旋转角度上限，单位度；实际角度还受长宽比收紧。
        pad_p=0.5,  # 每次处理触发边缘补边的概率。
        scale_p=0.5,  # 每次处理触发等比缩放的概率。
        stretch_p=0.4,  # 每次处理触发横向拉伸的概率。
        rotate_p=0.5,  # 每次处理触发旋转的概率；与斜切、透视互斥。
        # 斜切、透视、弹性扭曲默认关闭：2026-10-24 那轮把它们和边缘轻裁全部打开后，
        # 同 epoch 训练 loss 是只开 A1-A4 时的 3 倍，验证集准确率和几何鲁棒性同时下降。
        # 单独开启时用这里的概率参数做消融，不要一次打开多项。
        shear_p=0.0,  # 每次处理触发水平斜切的概率；与旋转、透视互斥。
        perspective_p=0.0,  # 每次处理触发透视畸变的概率；与旋转、斜切互斥。
        elastic_p=0.0,  # 每次处理触发弹性扭曲的概率；可与倾斜操作叠加。
        crop_p=0.2,  # 每次处理触发单侧边缘轻裁的概率；只削 1 px 且限制墨迹损失，对字形改变最小。
    ):
        if pad_px < 0:
            raise ValueError(f"pad_px 不能为负，当前为 {pad_px}")
        if degrees < 0:
            raise ValueError(f"degrees 不能为负，当前为 {degrees}")
        self.scale = scale
        self.stretch = stretch
        self.pad_px = pad_px
        self.degrees = degrees
        self.pad_p = pad_p
        self.scale_p = scale_p
        self.stretch_p = stretch_p
        self.rotate_p = rotate_p
        self.shear_p = shear_p
        self.perspective_p = perspective_p
        self.elastic_p = elastic_p
        self.crop_p = crop_p

    def _aspect_limited_degrees(self, image: Image.Image) -> float:
        width, height = image.size
        return limited_rotation_degrees(width, height, self.degrees)

    @staticmethod
    def _warp(image: Image.Image, matrix, size, fill, resample) -> Image.Image:
        """按仿射矩阵重采样，画布外的区域用背景色填充。"""
        interpolation = {
            Image.Resampling.LANCZOS: cv2.INTER_LANCZOS4,
            Image.Resampling.BICUBIC: cv2.INTER_CUBIC,
            Image.Resampling.BILINEAR: cv2.INTER_LINEAR,
        }[resample]
        warped = cv2.warpAffine(
            np.asarray(image),
            matrix,
            size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        return Image.fromarray(warped)

    def _rotate(self, image: Image.Image, fill, resample) -> Image.Image:
        """扩边旋转：expand 保证内容不被裁掉，角度按长宽比收紧。"""
        limit = self._aspect_limited_degrees(image)
        if limit <= 0:
            return image
        return image.rotate(
            random.uniform(-limit, limit),
            resample=ROTATION_RESAMPLE[resample],
            expand=True,
            fillcolor=fill,
        )

    @staticmethod
    def _shear(image: Image.Image, fill, resample) -> Image.Image:
        """水平斜切：顶部相对底部水平位移，模拟行基线倾斜。"""
        coefficient = random.uniform(-SHEAR_LIMIT, SHEAR_LIMIT)
        width, height = image.size
        shift = max(0.0, -coefficient * height)
        canvas_width = width + int(math.ceil(abs(coefficient) * height))
        matrix = np.array(
            [[1.0, coefficient, shift], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        return RandomSourceGeometry._warp(
            image, matrix, (canvas_width, height), fill, resample
        )

    @staticmethod
    def _perspective(image: Image.Image, fill, resample) -> Image.Image:
        """透视畸变：四角在画布内随机位移，扩张的画布保证内容不被裁掉。"""
        width, height = image.size
        margin = max(1, int(round(PERSPECTIVE_LIMIT * height)))
        canvas_width, canvas_height = width + 2 * margin, height + 2 * margin
        canvas = Image.new("RGB", (canvas_width, canvas_height), fill)
        canvas.paste(image, (margin, margin))
        natural = np.array(
            [
                [margin, margin],
                [width + margin, margin],
                [width + margin, height + margin],
                [margin, height + margin],
            ],
            dtype=np.float32,
        )
        jitter = np.array(
            [
                [random.uniform(-margin, margin), random.uniform(-margin, margin)]
                for _ in range(4)
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(natural, natural + jitter)
        warped = cv2.warpPerspective(
            np.asarray(canvas),
            matrix,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        return Image.fromarray(warped)

    @staticmethod
    def _elastic(image: Image.Image, fill, resample) -> Image.Image:
        """弹性扭曲：粗网格随机位移插值后重采样，模拟纸张弯曲和镜头畸变。

        画布先按位移上限外扩，内容只会向画布内部移动，不会被推到边界之外。"""
        width, height = image.size
        pad = int(math.ceil(ELASTIC_ALPHA))
        canvas = Image.new("RGB", (width + 2 * pad, height + 2 * pad), fill)
        canvas.paste(image, (pad, pad))
        canvas_width, canvas_height = canvas.size
        cells = 4
        # 用 random 采样位移场，使几何增强的随机性只来自一个被 worker 种子控制的源。
        offsets = np.array(
            [
                random.uniform(-ELASTIC_ALPHA, ELASTIC_ALPHA)
                for _ in range(2 * (cells + 1) ** 2)
            ],
            dtype=np.float32,
        ).reshape(2, cells + 1, cells + 1)
        field = cv2.resize(
            offsets.transpose(1, 2, 0),
            (canvas_width, canvas_height),
            interpolation=cv2.INTER_LINEAR,
        )
        grid_x, grid_y = np.meshgrid(
            np.arange(canvas_width, dtype=np.float32),
            np.arange(canvas_height, dtype=np.float32),
        )
        warped = cv2.remap(
            np.asarray(canvas),
            grid_x + field[:, :, 0],
            grid_y + field[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        return Image.fromarray(warped)

    @staticmethod
    def _trim_one_side(image: Image.Image, fill) -> Image.Image:
        """去掉指定一侧的 1 px；切入正文的比例超过上限时保留原图。

        只处理一侧：其余三边仍保留背景，后续窄图补边仍能正确估计纸面色。"""
        pixels = np.asarray(image).astype(np.int16)
        ink = np.abs(pixels - np.asarray(fill, dtype=np.int16)).max(axis=-1) > INK_THRESHOLD
        if image.width <= 2 or image.height <= 2:
            return image
        side = random.randrange(4)
        if side == 0:  # 左
            strip, box = ink[:, 0], (1, 0, image.width, image.height)
        elif side == 1:  # 右
            strip, box = ink[:, -1], (0, 0, image.width - 1, image.height)
        elif side == 2:  # 上
            strip, box = ink[0, :], (0, 1, image.width, image.height)
        else:  # 下
            strip, box = ink[-1, :], (0, 0, image.width, image.height - 1)

        total_ink = int(ink.sum())
        if total_ink and strip.sum() > MAX_TRIM_INK_FRACTION * total_ink:
            return image
        return image.crop(box)

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            raise ValueError(f"RandomSourceGeometry 需要 RGB 图片，当前为 {image.mode}")

        fill = border_fill_color(image)
        # 每次调用重新抽取重采样核；同一次增强内部的缩放保持同一核，避免多级重采样叠加失真。
        resample = random.choice(RESAMPLE_FILTERS)
        pads = [0, 0, 0, 0]
        trim = random.random() < self.crop_p
        applied = trim

        # 1. 边缘补边：直接决定重裁后文字占画布的高度比例。
        if self.pad_px > 0 and random.random() < self.pad_p:
            pads = [random.randint(0, self.pad_px) for _ in range(4)]
            applied = True

        # 2. 内容等比缩放和横向拉伸：两者都是可分离的重采样，合并成一次 resize，
        #    避免连续两次插值把细笔画磨淡；缩放改变文字占比，拉伸改变字形宽高比。
        width_factor = 1.0
        height_factor = 1.0
        if random.random() < self.scale_p:
            factor = random.uniform(*self.scale)
            width_factor *= factor
            height_factor *= factor
            applied = True
        if random.random() < self.stretch_p:
            width_factor *= random.uniform(*self.stretch)
            applied = True
        if width_factor != 1.0 or height_factor != 1.0:
            image = image.resize(
                (
                    max(1, round(image.width * width_factor)),
                    max(1, round(image.height * height_factor)),
                ),
                resample,
            )

        # 3. 倾斜族：旋转、斜切、透视同属页面倾斜，每次最多施加一种。
        shape_operations = [
            (self.rotate_p, self._rotate),
            (self.shear_p, self._shear),
            (self.perspective_p, self._perspective),
        ]
        random.shuffle(shape_operations)
        for probability, operation in shape_operations:
            if random.random() < probability:
                image = operation(image, fill, resample)
                applied = True
                break

        # 4. 弹性扭曲：可与倾斜叠加，模拟纸张弯曲。
        if random.random() < self.elastic_p:
            image = self._elastic(image, fill, resample)
            applied = True

        if not applied:
            return image

        # 5. 按墨迹外接框重裁并加留白。画布先四周外扩，外扩量取留白上限和最小留白的
        #    较大者，保证重裁不越界；每边至少保留 MIN_CROP_MARGIN，零留白会让后续
        #    无法再估计纸面色，crop 越界也会被 PIL 补成黑色。
        canvas_margin = max(self.pad_px, MIN_CROP_MARGIN)
        canvas = Image.new(
            image.mode,
            (image.width + 2 * canvas_margin, image.height + 2 * canvas_margin),
            fill,
        )
        canvas.paste(image, (canvas_margin, canvas_margin))
        left, top, right, bottom = ink_extent(canvas, fill)
        cropped = canvas.crop(
            (
                left - max(pads[0], MIN_CROP_MARGIN),
                top - max(pads[1], MIN_CROP_MARGIN),
                right + max(pads[2], MIN_CROP_MARGIN),
                bottom + max(pads[3], MIN_CROP_MARGIN),
            )
        )

        # 6. 单侧边缘轻裁：模拟检测框贴边或切进字形一像素。
        if trim:
            cropped = self._trim_one_side(cropped, fill)
        return cropped


class AddGaussianNoise:
    """模拟扫描颗粒和相机传感器轻噪声；截断到 [0, 1]，避免异常亮暗值。"""

    def __init__(self, mean: float = NOISE_MEAN, std: float = NOISE_STD):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return (tensor + noise).clamp(0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def build_transform(*, is_train: bool, num_channels: int) -> transforms.Compose:
    """构建缩放后的图像变换；验证和推理直接复用 runtime 的确定性管线。"""
    if num_channels not in (1, 3):
        raise ValueError(f"num_channels 仅支持 1 或 3，当前为 {num_channels}")

    if not is_train:
        return build_eval_transform(num_channels)

    channel_transform = (
        # 兼容旧的单通道模型；RGB 模型保留颜色信息。这是通道适配，不是增强。
        [transforms.Grayscale(num_output_channels=1)]
        if num_channels == 1
        else []
    )
    normalize = transforms.Normalize(
        # 将 [0, 1] 映射到 [-1, 1]，统一训练和推理数值范围，不是随机增强。
        mean=[0.5] * num_channels,
        std=[0.5] * num_channels,
    )

    return transforms.Compose([
        # 颜色扰动：模拟曝光、纸面反差、印刷浓淡和轻微偏色。
        transforms.RandomApply(
            [transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
                hue=COLOR_JITTER_HUE,
            )],
            p=COLOR_JITTER_PROBABILITY,
        ),
        # 自动对比度：覆盖扫描软件拉伸灰度、截图反差差异。
        transforms.RandomAutocontrast(p=AUTOCONTRAST_PROBABILITY),
        # 反色：覆盖深色主题、黑底白字；也会反转残留章的颜色。
        transforms.RandomInvert(p=INVERT_PROBABILITY),
        *channel_transform,
        # 转为 [C, H, W] 的 [0, 1] 浮点张量，供模糊/噪声/擦除使用。
        transforms.ToTensor(),
        # 轻度模糊：模拟轻微失焦、低分辨率截图和缩放损失。
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA)],
            p=BLUR_PROBABILITY,
        ),
        # 高斯噪声：模拟扫描颗粒、拍照暗部噪点，不改变文字位置。
        transforms.RandomApply(
            [AddGaussianNoise()],
            p=NOISE_PROBABILITY,
        ),
        # 小块擦除：模拟污点、磨损和局部遮挡。仍可能遮住小数点或短文本的关键笔画，
        # 因此保持低概率、小面积。
        transforms.RandomErasing(
            p=ERASING_PROBABILITY,
            scale=ERASING_SCALE,
            ratio=ERASING_RATIO,
            value="random",
        ),
        normalize,
    ])
