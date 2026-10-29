"""一次性迁移：把两个历史检测目录的数据去重合并到统一 detection/data。

本工具不属于统一检测运行时；它只读取旧目录并复制文件，不修改或删除源数据。
去重按图片内容 MD5 分组，同一内容只保留一份代表样本；每个来源域分别做
固定种子的 80/20 划分，同一内容不会同时进入 train 和 val。
"""
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "detection" / "data"
MANIFEST = DESTINATION / "split_manifest.csv"
SEED = 42  # 固定划分种子，保证重复执行得到相同的 train/val 归属。
VAL_RATIO = 0.20  # 验证集占比；按唯一图片内容计数。
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# (来源域, 目录优先级, 图片目录, 标签目录)；来源域用于划分和文件名前缀。
SOURCES = (
    ("ui", 0, ROOT / "detection_small" / "data" / "train_images",
     ROOT / "detection_small" / "data" / "train_labels"),
    ("ui", 1, ROOT / "detection_small" / "data" / "val_images",
     ROOT / "detection_small" / "data" / "val_labels"),
    ("doc", 0, ROOT / "detection_larger" / "data" / "train_images",
     ROOT / "detection_larger" / "data" / "train_labels"),
    ("doc", 1, ROOT / "detection_larger" / "data" / "val_images",
     ROOT / "detection_larger" / "data" / "val_labels"),
)
TEST_SOURCE = ROOT / "detection_larger" / "data" / "test_images"  # 统一推理演示图片，复制顶层图片文件。


def find_label(image_path, label_dir):
    """按 stem 找到唯一标签，兼容 gt_ 前缀，缺失或歧义直接报错。"""
    candidates = [path for path in (label_dir / f"gt_{image_path.stem}.txt",
                                    label_dir / f"{image_path.stem}.txt") if path.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one label for {image_path}: {candidates}")
    return candidates[0]


def collect_unique():
    """按内容 MD5 收集唯一图片，选择一份确定性的代表图片和标签。"""
    groups = defaultdict(list)
    for domain, priority, image_dir, label_dir in SOURCES:
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {image_dir}, {label_dir}")
        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = hashlib.md5(image_path.read_bytes()).hexdigest()
            groups[digest].append((domain, priority, image_path, find_label(image_path, label_dir)))

    unique = {}
    for digest, candidates in groups.items():
        domain, _, image_path, label_path = min(
            candidates, key=lambda item: (item[0] != "ui", item[1], item[2].name))
        unique[digest] = (domain, image_path, label_path)
    return unique


def split_unique(unique):
    """按来源域分别做固定种子 80/20 划分，返回 content_md5 -> split。"""
    by_domain = defaultdict(list)
    for digest, (domain, _, _) in unique.items():
        by_domain[domain].append(digest)

    assignments = {}
    for domain in sorted(by_domain):
        digests = sorted(by_domain[domain])
        random.Random(f"{SEED}-{domain}").shuffle(digests)
        val_count = max(1, round(len(digests) * VAL_RATIO))
        for index, digest in enumerate(digests):
            assignments[digest] = "val" if index < val_count else "train"
    return assignments


def copy_samples(unique, assignments):
    """复制唯一代表样本和标签，返回清单行。"""
    rows = []
    used_stems = {}
    for digest, (domain, image_path, label_path) in unique.items():
        split = assignments[digest]
        destination_stem = f"{domain}_{image_path.stem}"
        if destination_stem in used_stems and used_stems[destination_stem] != digest:
            raise ValueError(f"Duplicate destination stem with different content: {destination_stem}")
        used_stems[destination_stem] = digest
        destination_name = f"{destination_stem}{image_path.suffix.lower()}"
        image_destination = DESTINATION / f"{split}_images" / destination_name
        label_name = f"gt_{destination_stem}.txt"
        label_destination = DESTINATION / f"{split}_labels" / label_name
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_destination)
        shutil.copy2(label_path, label_destination)
        try:
            source_image = str(image_path.relative_to(ROOT))
        except ValueError:
            # 源目录可在仓库外；清单保留绝对路径，便于追溯。
            source_image = str(image_path)
        rows.append({
            "sample_id": destination_stem,
            "image_name": destination_name,
            "label_name": label_name,
            "split": split,
            "source": domain,
            "source_image": source_image,
            "content_md5": digest,
        })
    return sorted(rows, key=lambda row: row["sample_id"])


def copy_test_images():
    """复制旧文档目录顶层的测试图片，作为统一推理默认输入。"""
    if not TEST_SOURCE.is_dir():
        return 0
    destination = DESTINATION / "test_images"
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in TEST_SOURCE.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(path, destination / path.name)
            count += 1
    return count


def verify(rows):
    """确认划分无内容重叠且图片与标签都成对存在。"""
    hashes = defaultdict(set)
    for row in rows:
        hashes[row["content_md5"]].add(row["split"])
        image = DESTINATION / f"{row['split']}_images" / row["image_name"]
        label = DESTINATION / f"{row['split']}_labels" / row["label_name"]
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"Missing copied file: {image}, {label}")
        if hashlib.md5(image.read_bytes()).hexdigest() != row["content_md5"]:
            raise RuntimeError(f"Copied content differs from source: {image}")
        with Image.open(image) as source:
            source.verify()
    leaked = {digest: splits for digest, splits in hashes.items() if len(splits) > 1}
    if leaked:
        raise RuntimeError(f"Content leaked across splits: {list(leaked.items())[:5]}")
    return hashes


def main():
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        raise FileExistsError(f"Destination must be empty or absent: {DESTINATION}")
    unique = collect_unique()
    assignments = split_unique(unique)
    rows = copy_samples(unique, assignments)
    test_count = copy_test_images()

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verify(rows)
    train_count = sum(row["split"] == "train" for row in rows)
    val_count = len(rows) - train_count
    print(f"唯一图片 {len(rows)} 张：train {train_count}，val {val_count}；"
          f"测试图片 {test_count} 张；清单 {MANIFEST}")


if __name__ == "__main__":
    main()
