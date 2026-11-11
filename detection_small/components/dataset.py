"""小图检测的数据加载：严格按 stem 配对标签，解析水平 bbox4/quad8 和 ### 忽略区域。"""
import csv
import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .image import IMAGE_EXTENSIONS, load_rgb_image


LABEL_FORMATS = ("auto", "bbox4", "quad8")  # auto 根据坐标字段数识别；其余格式强制按指定字段数解析。
DEFAULT_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS  # 加载器默认图片扩展名；训练入口仍显式限定 PNG。


def _is_horizontal_quad(values):
    """判断 8 个坐标是否恰好构成轴对齐矩形，避免把数字正文误判成 quad8。"""
    quad = np.asarray(values, dtype=np.float32).reshape(4, 2)
    box = np.r_[quad.min(axis=0), quad.max(axis=0)]
    corners = {(float(x), float(y)) for x, y in quad}
    expected = {(float(x), float(y)) for x in box[[0, 2]] for y in box[[1, 3]]}
    return corners == expected


def _parse_line(line, label_format, size):
    """解析一行标签，返回 (四边形, 是否忽略)；格式或坐标非法时抛出 ValueError。"""
    # 只检查第一个坐标后的分隔符，正文中的逗号不改变坐标解析方式。
    comma_coordinates = re.match(r"^\s*[^\s,]+\s*,", line) is not None
    fields = next(csv.reader([line])) if comma_coordinates else line.split()

    numbers = []
    for field in fields:
        try:
            numbers.append(float(field))
        except ValueError:
            break

    if label_format == "bbox4":
        count = 4
    elif label_format == "quad8":
        count = 8
    elif len(numbers) >= 8 and _is_horizontal_quad(numbers[:8]):
        count = 8
    else:
        count = 4

    if len(numbers) < count or not np.isfinite(numbers[:count]).all():
        raise ValueError(f"需要 {count} 个有限坐标，实际为 {line!r}")

    if count == 8:
        quad = np.asarray(numbers[:8], dtype=np.float32).reshape(4, 2)
        box = np.r_[quad.min(axis=0), quad.max(axis=0)]
        if not _is_horizontal_quad(numbers[:8]):
            raise ValueError("quad8 必须是水平矩形；旋转标注需要先转成 bbox4")
    else:
        x_min, y_min, x_max, y_max = numbers[:4]
        quad = np.array([[x_min, y_min], [x_max, y_min],
                         [x_max, y_max], [x_min, y_max]], dtype=np.float32)
        box = np.asarray(numbers[:4], dtype=np.float32)

    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(f"框退化或坐标顺序错误: {box.tolist()}")
    if size is not None and not (0 <= x_min and 0 <= y_min
                                 and x_max <= size[0] and y_max <= size[1]):
        raise ValueError(f"框超出图片范围: {box.tolist()}, 图片尺寸={size}")

    transcription = " ".join(fields[count:]).strip().strip('"')
    return quad, transcription == "###"


def read_quads(label_path, size=None, label_format="auto"):
    """读取标签文件，返回 (quads[N,4,2], ignored[N])。

    `###` 行保留四边形并标记为忽略区域；空文件表示已确认的无文字图片。
    `size` 为原图 (宽, 高)，提供时校验框不越界。格式错误立即报出文件和行号。
    """
    if label_format not in LABEL_FORMATS:
        raise ValueError(f"不支持的标签格式: {label_format}; 可选 {LABEL_FORMATS}")

    label_path = Path(label_path)
    quads, ignored = [], []
    for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            quad, is_ignored = _parse_line(line, label_format, size)
        except ValueError as error:
            raise ValueError(f"标签格式错误 {label_path}:{line_number}: {error}") from error
        quads.append(quad)
        ignored.append(is_ignored)

    if not quads:
        return np.empty((0, 4, 2), dtype=np.float32), np.zeros(0, dtype=bool)
    return np.stack(quads).astype(np.float32), np.asarray(ignored, dtype=bool)


class OCRDataset(Dataset):
    """读取小图检测图片及 bbox4/quad8 标签，返回图片、四边形和忽略标记。

    图片与标签按 stem 配对；`gt_<stem>.txt` 和 `<stem>.txt` 必须恰好存在一份，
    缺失或同时存在都会立即报错，避免静默使用错误标注污染训练。
    """

    def __init__(self, image_dir, label_dir, transform=None,
                 image_extensions=DEFAULT_IMAGE_EXTENSIONS,
                 require_labels=True, label_format="auto"):
        """
        初始化数据集。

        :param image_dir: 图片目录。
        :param label_dir: 标签目录。
        :param transform: 可选的样本变换。
        :param image_extensions: 参与枚举的图片扩展名。
        :param require_labels: 为 True 时缺标签立即报错；为 False 时缺标签作为空框样本。
        :param label_format: auto、bbox4 或 quad8；现有小图数据为 bbox4。
        """
        if label_format not in LABEL_FORMATS:
            raise ValueError(f"不支持的标签格式: {label_format}; 可选 {LABEL_FORMATS}")
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.label_format = label_format
        self.require_labels = require_labels

        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"图片或标签目录不存在: {self.image_dir}, {self.label_dir}")

        extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in image_extensions
        }
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions)
        if not self.image_paths:
            raise ValueError(f"目录中没有支持的图片: {self.image_dir}")

        self.label_paths = []
        for image_path in self.image_paths:
            candidates = [path for path in (self.label_dir / f"gt_{image_path.stem}.txt",
                                            self.label_dir / f"{image_path.stem}.txt")
                          if path.is_file()]
            if len(candidates) > 1:
                raise ValueError(
                    f"图片 {image_path} 存在多份标签: {candidates}；"
                    "同一张图片必须恰好保留一份标签。")
            if not candidates and require_labels:
                raise FileNotFoundError(
                    f"图片缺少对应标签: {image_path}；"
                    "标签名应为 gt_<图片名>.txt 或 <图片名>.txt。")
            self.label_paths.append(candidates[0] if candidates else None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """读取 RGB 图片和标签，返回图片、四边形与忽略标记。"""
        image = load_rgb_image(self.image_paths[idx])
        label_path = self.label_paths[idx]
        if label_path is None:
            quads = np.empty((0, 4, 2), dtype=np.float32)
            ignored = np.zeros(0, dtype=bool)
        else:
            quads, ignored = read_quads(label_path, image.size, self.label_format)

        sample = {"image": image, "quads": quads, "ignored": ignored}
        if self.transform:
            sample = self.transform(sample)
        return sample
