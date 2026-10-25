"""按顶部配置评估识别模型对源分辨率扰动的鲁棒性。

对同一批验证样本逐一施加确定性扰动，比较准确率和 CER 相对无扰动基线的变化。
用于核对训练增强是否覆盖了实际敏感方向：几何扰动的跌幅应控制在小范围内，
而不是只在原始几何上表现良好。任一条件超出阈值时以非零状态退出，可直接作为验收脚本。

扰动在缩放前作用于源分辨率图片，和训练几何增强的位置一致。旋转与训练使用同一
长宽比收紧规则，评估条件不会超出训练覆盖范围；其余条件用于观察既有增强的覆盖度。
"""

import os
import random
import sys
from functools import partial

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.components.converter import LabelConverter
from recognition.components.dataset import (
    InferenceDataset,
    dynamic_padding_collate_fn,
)
from recognition.components.augmentation import (
    MIN_CROP_MARGIN,
    ink_extent,
    limited_rotation_degrees,
)
from recognition.components.preprocessing import border_fill_color
from recognition.evaluate_val import levenshtein_distance
from recognition.models.model import CRNN
from recognition.utils.annotation import split_annotation_fields
from recognition.utils.checkpoint import extract_state_dict, infer_num_channels

# 配置区域
MODEL_PATH = "recognition/checkpoints/best_cer_0.004061_epoch_603.pth"  # 本次评估的 CRNN 权重，需与 KEYS_PATH 配套。
KEYS_PATH = "recognition/keys.txt"  # 权重训练时的字符字典，内容及顺序必须一致。
VAL_ANNOTATION = "recognition/data/val_annotation.txt"  # 本次评估的 Tab 标注，图片路径相对于该文件。
SAMPLE_LIMIT = 2000  # 参与评估的样本数；0 表示使用全部样本。条件越多耗时越久，默认抽样。
SAMPLE_SEED = 0  # 抽样种子；固定后同一份数据每次评估使用同一批样本，便于对比不同权重。

IMG_HEIGHT = 32  # 输入固定高度，单位 px；评估不限制宽度。
DEFAULT_NUM_CHANNELS = 3  # 无法从首层权重推断通道时默认使用 RGB 三通道。
BATCH_SIZE = 64  # 每批评估图片数，末批保留。
NUM_WORKERS = 4  # 数据加载子进程数；0 表示主进程加载。

MAX_ACCURACY_DROP = 1.0  # 任一扰动相对无扰动基线的准确率跌幅上限，单位百分点。
MAX_CER_RISE = 1.0  # 任一扰动相对无扰动基线的 CER 上升上限，单位百分点。

# 单张回归图：在所有条件下打印预测，用于确认已知失败样本已经稳定。
# 留空字符串表示跳过；文件不存在时只警告，不影响验证集统计。
FOCUS_IMAGE = "recognition/data/test_images/20261023-1.png"  # 已知几何临界样本；正确结果为 crm.tngs.com。


def _identity(image: Image.Image) -> Image.Image:
    return image


def _pad(image: Image.Image, pixels: int) -> Image.Image:
    """四周补同色背景，改变文字占画布的高度比例。"""
    fill = border_fill_color(image)
    canvas = Image.new(image.mode, (image.width + 2 * pixels, image.height + 2 * pixels), fill)
    canvas.paste(image, (pixels, pixels))
    return canvas


def _stretch(image: Image.Image, factor: float) -> Image.Image:
    """只改变宽度，模拟字形宽高比差异。"""
    return image.resize(
        (max(1, round(image.width * factor)), image.height),
        Image.Resampling.LANCZOS,
    )


def _rotate(image: Image.Image, degrees: float) -> Image.Image:
    """扩边旋转后按墨迹重裁，和训练增强保持同样的角度收紧和留白规则。"""
    fill = border_fill_color(image)
    angle = limited_rotation_degrees(image.width, image.height, abs(degrees))
    if angle <= 0:
        return image
    rotated = image.rotate(
        angle if degrees >= 0 else -angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fill,
    )
    left, top, right, bottom = ink_extent(rotated, fill)
    return rotated.crop(
        (
            left - MIN_CROP_MARGIN,
            top - MIN_CROP_MARGIN,
            right + MIN_CROP_MARGIN,
            bottom + MIN_CROP_MARGIN,
        )
    )


def _blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(sigma))


def _contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    """有损压缩与重新解码，模拟截图和文档存储带来的块状伪影。"""
    import io

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert(image.mode)


# 每个条件对应一组扰动函数；名称中的数值就是实际使用的强度。
CONDITIONS = (
    ("identity", _identity, {}),
    ("pad_1px", _pad, {"pixels": 1}),
    ("pad_2px", _pad, {"pixels": 2}),
    ("stretch_0.95", _stretch, {"factor": 0.95}),
    ("stretch_1.05", _stretch, {"factor": 1.05}),
    ("rotate_+1deg", _rotate, {"degrees": 1.0}),
    ("rotate_-1deg", _rotate, {"degrees": -1.0}),
    ("rotate_+2deg", _rotate, {"degrees": 2.0}),
    ("rotate_-2deg", _rotate, {"degrees": -2.0}),
    ("blur_0.8", _blur, {"sigma": 0.8}),
    ("contrast_0.85", _contrast, {"factor": 0.85}),
    ("jpeg_60", _jpeg, {"quality": 60}),
)


def _load_labels(annotation_path: str) -> tuple[list[str], dict[str, str]]:
    """按标注路径读取图片，返回路径列表和路径到文字的映射。"""
    image_paths = []
    labels = {}
    for line_number, line in enumerate(open(annotation_path, encoding="utf-8"), start=1):
        path_part, label_part = split_annotation_fields(line, annotation_path, line_number)
        label = label_part.strip()
        if not label:
            continue
        image_path = os.path.join(os.path.dirname(annotation_path), path_part)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        image_paths.append(image_path)
        labels[image_path] = label
    return image_paths, labels


def _evaluate(model, converter, image_paths, labels, num_channels, device, perturb):
    """返回给定扰动下的整串准确率（%）和平均 CER（%）。"""
    dataset = InferenceDataset(
        image_paths,
        img_height=IMG_HEIGHT,
        num_channels=num_channels,
        source_augment=perturb,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=dynamic_padding_collate_fn,
    )

    correct = 0
    char_distance = 0
    char_total = 0
    with torch.no_grad():
        for images, paths, img_widths in tqdm(loader, desc="Evaluating", leave=False):
            prediction = model(images.to(device), img_widths=img_widths.to(device))
            texts = converter.decode(prediction.cpu(), layout="TBC")
            for path, text in zip(paths, texts):
                truth = labels[path]
                distance = levenshtein_distance(truth, text.strip())
                correct += distance == 0
                char_distance += distance
                char_total += max(len(truth), 1)
    return correct / len(image_paths) * 100, char_distance / max(char_total, 1) * 100


def main():
    print(f"加载权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    converter = LabelConverter(KEYS_PATH)
    num_channels = infer_num_channels(state_dict, default=DEFAULT_NUM_CHANNELS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(
        img_height=IMG_HEIGHT,
        nc=num_channels,
        n_class=converter.num_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"设备: {device} / 输入通道: {num_channels}")

    image_paths, labels = _load_labels(VAL_ANNOTATION)
    if SAMPLE_LIMIT > 0:
        random.Random(SAMPLE_SEED).shuffle(image_paths)
        image_paths = sorted(image_paths[:SAMPLE_LIMIT])
    print(f"验证样本: {len(image_paths)} 条")

    results = []
    for name, perturb, kwargs in CONDITIONS:
        accuracy, cer = _evaluate(
            model, converter, image_paths, labels, num_channels, device,
            partial(perturb, **kwargs),
        )
        results.append((name, accuracy, cer))

    baseline_accuracy, baseline_cer = results[0][1], results[0][2]
    print()
    print(f"{'条件':<16}{'准确率%':>9}{'CER%':>8}{'Δ准确率':>10}{'ΔCER':>9}  判定")
    print("-" * 66)
    failed = False
    for name, accuracy, cer in results:
        accuracy_drop = accuracy - baseline_accuracy
        cer_rise = cer - baseline_cer
        if name == "identity":
            verdict = "基线"
        else:
            passed = accuracy_drop >= -MAX_ACCURACY_DROP and cer_rise <= MAX_CER_RISE
            failed = failed or not passed
            verdict = "PASS" if passed else "FAIL"
        print(
            f"{name:<16}{accuracy:>9.2f}{cer:>8.3f}"
            f"{accuracy_drop:>+10.2f}{cer_rise:>+9.3f}  {verdict}"
        )
    print("-" * 66)
    print(
        f"判定阈值: 准确率跌幅 ≤ {MAX_ACCURACY_DROP:.1f} 百分点，"
        f"CER 上升 ≤ {MAX_CER_RISE:.1f} 百分点"
    )

    _report_focus_image(model, converter, num_channels, device)

    if failed:
        print("鲁棒性验收未通过。")
        return 1
    print("鲁棒性验收通过。")
    return 0


def _report_focus_image(model, converter, num_channels, device):
    """在同一组条件下打印单张回归图的预测，便于人工核对。"""
    if not FOCUS_IMAGE:
        return
    if not os.path.isfile(FOCUS_IMAGE):
        print(f"跳过回归图检查：文件不存在 {FOCUS_IMAGE}")
        return

    print()
    print(f"回归图: {FOCUS_IMAGE}")
    for name, perturb, kwargs in CONDITIONS:
        dataset = InferenceDataset(
            [FOCUS_IMAGE],
            img_height=IMG_HEIGHT,
            num_channels=num_channels,
            source_augment=partial(perturb, **kwargs),
        )
        images, _, img_widths = dynamic_padding_collate_fn([dataset[0]])
        with torch.no_grad():
            prediction = model(images.to(device), img_widths=img_widths.to(device))
        text = converter.decode(prediction.cpu(), layout="TBC")[0]
        print(f"  {name:<16}{text}")


if __name__ == "__main__":
    sys.exit(main())
