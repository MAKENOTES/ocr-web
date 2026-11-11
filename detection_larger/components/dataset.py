"""按图片目录和标签目录读取完整 PDF 页面与表格图片。"""

import csv
import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from detection_larger.components.image import IMAGE_EXTENSIONS, load_rgb_image


LABEL_FORMATS = {"bbox4": 4, "quad8": 8}  # 支持的水平标签格式及每行坐标数。


def read_boxes(path: Path, size: tuple[int, int], label_format="bbox4"):
    """bbox4 或 quad8 后可带文本；### 为忽略区，空文件为已确认的无文字图片。

    quad8 只接受水平矩形，旋转标注直接报错，避免被静默压成外接矩形。
    """
    if label_format not in LABEL_FORMATS:
        raise ValueError(f"Unsupported label format: {label_format}; expected one of {tuple(LABEL_FORMATS)}")
    count = LABEL_FORMATS[label_format]
    boxes, ignored = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            # 只检查第一个坐标后的分隔符，文本中的逗号不改变坐标解析方式。
            comma_coordinates = re.match(r"^\s*[^\s,]+\s*,", line) is not None
            fields = next(csv.reader([line])) if comma_coordinates else line.split()
            coordinates = np.asarray([float(v) for v in fields[:count]], dtype=np.float32)
            if len(coordinates) != count or not np.isfinite(coordinates).all():
                raise ValueError("coordinates must be finite")
            if count == 8:
                quad = coordinates.reshape(4, 2)
                box = np.r_[quad.min(axis=0), quad.max(axis=0)]
                # 当前几何头只表达水平矩形，不静默吞掉旋转标注。
                corners = {(float(x), float(y)) for x, y in quad}
                expected = {(float(x), float(y)) for x in box[[0, 2]] for y in box[[1, 3]]}
                if corners != expected:
                    raise ValueError("rotated quadrilaterals are unsupported; use horizontal text boxes")
            else:
                box = coordinates
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
                raise ValueError(f"box outside image or degenerate: {box.tolist()}, image size={size}")
            boxes.append(box)
            ignored.append(" ".join(fields[count:]).strip().strip('"') == "###")
        except (ValueError, csv.Error) as exc:
            raise ValueError(f"Invalid label {path}:{line_number}: {exc}") from exc
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(ignored, dtype=bool)


class OCRDataset(Dataset):
    """枚举 JPG/JPEG/PNG/TIFF/BMP，严格配对一份标签；不划分或搬移数据。

    同一张图片同时存在 gt_<stem>.txt 和 <stem>.txt，或两者都不存在时立即
    报错，避免静默使用错误标注污染训练。
    """
    def __init__(self, image_dir, label_dir, transform=None, label_format="bbox4"):
        self.image_dir, self.label_dir = Path(image_dir), Path(label_dir)
        self.transform, self.label_format = transform, label_format
        if label_format not in LABEL_FORMATS:
            raise ValueError(f"Unsupported label format: {label_format}")
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Image/label directory missing: {self.image_dir}, {self.label_dir}")
        self.image_paths = sorted(p for p in self.image_dir.iterdir()
                                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.image_paths:
            raise ValueError(f"No supported images in {self.image_dir}")
        self.label_paths = []
        for image_path in self.image_paths:
            candidates = [p for p in (self.label_dir / f"gt_{image_path.stem}.txt",
                                      self.label_dir / f"{image_path.stem}.txt") if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one label for {image_path}; candidates={candidates}, label_dir={self.label_dir}")
            self.label_paths.append(candidates[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """读取 RGB 图片和标签，返回图片、四边形顶点和忽略标记。"""
        image = load_rgb_image(self.image_paths[index])
        boxes, ignored = read_boxes(self.label_paths[index], image.size, self.label_format)
        if boxes.size:
            quads = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]],
                              boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
        else:
            quads = np.empty((0, 4, 2), dtype=np.float32)
        sample = {"image": image, "quads": quads, "ignored": ignored}
        return self.transform(sample) if self.transform else sample
