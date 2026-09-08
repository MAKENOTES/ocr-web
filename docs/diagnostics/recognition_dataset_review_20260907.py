"""只读复核当前数据与历史去红预处理；不训练、不改数据、不加载业务权重。"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from PIL import Image
import scipy
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recognition.components.dataset import (
    BaseTextDataset, CNN_RIGHT_CONTEXT, dynamic_padding_collate_fn,
    load_rgb_image, resize_detection_image,
)
from recognition.models.model import MobileFeatureExtractor, CRNN
from recognition.utils.annotation import split_annotation_fields

OUT = ROOT / "outputs/recognition_dataset_review_20260907"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(20260907)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


config = {}
for node in ast.parse((ROOT / "recognition/train.py").read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try:
            config[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass

keys = (ROOT / config["KEYS_PATH"]).read_text().splitlines()
old_keys = git("show", "119c7b8:recognition/keys.txt").splitlines()
annotation_paths = list(dict.fromkeys([
    *config["TRAIN_ANNOTATION"], config["VALID_ANNOTATION"],
    "recognition/data/synthesis_annotation.txt",
]))
all_records = []
by_name = defaultdict(list)
stats = {}
counts = {}
manifest = {
    "git_head": git("rev-parse", "HEAD").strip(),
    "python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
    "scipy": scipy.__version__, "config": config,
    "hashes": {"recognition/keys.txt": sha256(ROOT / "recognition/keys.txt")},
    "old_function_commit": "b02fd0e",
    "checkpoint_available": (ROOT / "recognition/checkpoints/1788725792best_cer_0.0082_epoch_471.pth").exists(),
    "old_audit_available": (ROOT / "outputs/recognition_audit_20260907").exists(),
}
for relative in ("recognition/train.py", "recognition/components/dataset.py",
                 "recognition/models/model.py", "recognition/components/converter.py",
                 "recognition/utils/synthesis/generate.py",
                 "recognition/datadoc/analysisdata/errdata20260906.md",
                 "docs/diagnostics/recognition_dataset_review_20260907.py"):
    manifest["hashes"][relative] = sha256(ROOT / relative)
onnx_path = ROOT / "best_recognition.onnx"
if onnx_path.is_file():
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    def dimensions(value):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in value.type.tensor_type.shape.dim]
    manifest["default_onnx"] = {
        "inputs": {v.name: dimensions(v) for v in onnx_model.graph.input},
        "outputs": {v.name: dimensions(v) for v in onnx_model.graph.output},
        "metadata": {v.key: v.value for v in onnx_model.metadata_props},
    }
    manifest["hashes"]["best_recognition.onnx"] = sha256(onnx_path)
for rel in annotation_paths:
    path = ROOT / rel
    source = path.stem.removesuffix("_annotation")
    counter = Counter()
    source_records = []
    empty = 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        image, label = split_annotation_fields(line, path, number)
        label = label.strip()
        if not label:
            empty += 1
            continue
        record = {"source": source, "annotation_line": number,
                  "path": str((path.parent / image).relative_to(ROOT)), "label": label}
        source_records.append(record)
        by_name[Path(image).name].append(record)
        counter.update(label)
    counts[source] = counter
    all_records.extend(source_records)
    missing_paths = [r["path"] for r in source_records if not (ROOT / r["path"]).is_file()]
    stats[source] = {"samples": len(source_records), "characters": sum(counter.values()),
                     "unique": len(counter), "empty": empty,
                     "missing_images": len(missing_paths), "missing_examples": missing_paths[:5],
                     "oov": dict(counter - Counter({k: counter[k] for k in keys})),
                     "backslashes": counter["\\"],
                     "escaped_quote_samples": sum('\\"' in r["label"] for r in source_records)}
    if source == "bench2":
        stats[source]["missing_paths_with_exact_basename_in_bench2_images"] = sum(
            (path.parent / "bench2_images" / Path(p).name).is_file() for p in missing_paths
        )
    manifest["hashes"][rel] = sha256(path)

training_counts = sum((counts[Path(p).stem.removesuffix("_annotation")] for p in config["TRAIN_ANNOTATION"]), Counter())
frequency = Counter("0" if training_counts[k] == 0 else "1-10" if training_counts[k] <= 10
                    else "11-50" if training_counts[k] <= 50 else "51-100" if training_counts[k] <= 100
                    else "101+" for k in keys)

# 仅提取已审阅的旧函数，使用完全相同的默认参数和本机 scipy 分支。
old_source = git("show", "b02fd0e:recognition/components/dataset.py")
old_node = next(n for n in ast.parse(old_source).body
                if isinstance(n, ast.FunctionDef) and n.name == "remove_red_seal")
namespace = {"np": np, "Image": Image}
exec(compile(ast.Module(body=[old_node], type_ignores=[]), "b02fd0e:remove_red_seal", "exec"), namespace)
remove_red_seal = namespace["remove_red_seal"]
manifest["legacy_function_sha256"] = hashlib.sha256(ast.get_source_segment(old_source, old_node).encode()).hexdigest()
crop_source = git("show", "b02fd0e:recognition/utils/white_text_crop.py")
crop_nodes = [n for n in ast.parse(crop_source).body if isinstance(n, ast.FunctionDef)
              and n.name in ("find_white_text_bbox", "auto_crop_white_text_image")]
exec(compile(ast.Module(body=crop_nodes, type_ignores=[]), "b02fd0e:white_text_crop", "exec"), namespace)

section = ""
selected = defaultdict(set)
for line in (ROOT / "recognition/datadoc/analysisdata/errdata20260906.md").read_text().splitlines():
    if line.startswith("## "):
        section = line.removeprefix("## ")
    if line.strip().endswith(".png"):
        selected[line.strip()].add(section)

details = []
unresolved = []
for name, categories in selected.items():
    matches = [r for r in by_name[name] if r["source"] in ("train", "val")]
    if len(matches) != 1:
        unresolved.append({"filename": name, "categories": sorted(categories), "matches": matches})
        continue
    row = dict(matches[0])
    path = ROOT / row["path"]
    image = load_rgb_image(path)
    raw = np.asarray(image)
    old = np.asarray(remove_red_seal(image))
    changed = np.any(raw != old, axis=2)
    hsv = np.asarray(image.convert("HSV"))
    red = (((hsv[:, :, 0] <= 16) | (hsv[:, :, 0] >= 239)) &
           (hsv[:, :, 1] >= max(10, int(50 * (1.2 - 0.7 * 0.4)))) & (hsv[:, :, 2] >= 80))
    dark = np.asarray(image.convert("L")) < 150
    neutral_dark = (raw.max(axis=2) < 150) & ((raw.max(axis=2).astype(int) - raw.min(axis=2)) < 20)
    before_dark = int(dark.sum())
    keep_ratio = float((dark & ~red).sum() / before_dark) if before_dark else 1.0
    processed = resize_detection_image(image, 32, 1800)
    old_crop = namespace["auto_crop_white_text_image"](image)
    old_w = int(old_crop.width * 32 / old_crop.height)
    if old_w < int(old_crop.width * 0.5):
        old_w = min(int(old_crop.width * 0.5), 800)
    old_w = max(48, old_w)
    label = row["label"]
    needed = len(label) + sum(a == b for a, b in zip(label, label[1:]))
    row.update({"categories": sorted(categories), "original_wh": list(image.size),
                "processed_wh": list(processed.size),
                "legacy_crop_wh": list(old_crop.size), "legacy_processed_width": old_w,
                "ctc_steps": CRNN._calc_cnn_output_width(processed.width), "ctc_needed": needed,
                "raw_red_fraction": float(red.mean()), "guard_keep_ratio": keep_ratio,
                "legacy_changes": int(changed.sum()), "legacy_change_fraction": float(changed.mean()),
                "legacy_erased_neutral_dark": int((changed & neutral_dark).sum()),
                "legacy_erased_outside_initial_red": int((changed & ~red).sum()),
                "sha256": sha256(path)})
    details.append(row)

seal = [r for r in details if "受红章影响" in r["categories"]]
small = [r for r in details if "尺寸小？" in r["categories"]]

# 固定取每份来源的首/中/末共 32 张，模式仅描述抽样，不冒充全量审计。
mode_samples = {}
for source in stats:
    records = [r for r in all_records if r["source"] == source]
    indices = np.linspace(0, len(records) - 1, 32, dtype=int)
    modes = Counter()
    for i in indices:
        if not (ROOT / records[i]["path"]).is_file():
            modes["MISSING_EXACT_PATH"] += 1
            continue
        with Image.open(ROOT / records[i]["path"]) as image:
            modes[image.mode] += 1
    mode_samples[source] = dict(modes)

# 实际训练增强探针：10 张真实图，每张 100 个固定 seed；不强制擦除发生。
train_transforms = BaseTextDataset(32, 3, is_train=True).transform
augmentation = {"trials": 0, "outside_normalized_range": 0, "min": 0.0, "max": 0.0,
                "range_tolerance": 1e-5, "outlier_without_erasing": 0}
for index, row in enumerate([*seal[:5], *small[:5]]):
    image = resize_detection_image(load_rgb_image(ROOT / row["path"]), 32, 1800)
    for repeat in range(100):
        torch.manual_seed(20260907 + index * 100 + repeat)
        tensor = image
        erased_outlier = False
        for transform in train_transforms.transforms:
            tensor = transform(tensor)
            if isinstance(transform, torchvision.transforms.RandomErasing):
                erased_outlier = bool(((tensor < -1e-5) | (tensor > 1 + 1e-5)).any())
        augmentation["trials"] += 1
        outlier = bool(((tensor < -1 - 1e-5) | (tensor > 1 + 1e-5)).any())
        augmentation["outside_normalized_range"] += int(outlier)
        augmentation["outlier_without_erasing"] += int(outlier and not erased_outlier)
        augmentation["min"] = min(augmentation["min"], float(tensor.min()))
        augmentation["max"] = max(augmentation["max"], float(tensor.max()))

# 固定相同内容、相同 B，只加背景宽度，隔离训练期 BN 的空间统计依赖。
torch.manual_seed(20260907)
extractor = MobileFeatureExtractor(input_channel=3)
sample = torch.rand(3, 32, 64) * 2 - 1
base, _, widths = dynamic_padding_collate_fn([(sample, "x"), (sample.clone(), "y")])
fill = base[:, :, :, -1:]
extended = torch.cat((base, fill.expand(-1, -1, -1, 256)), dim=3)
bn_probe = {"input_shape": list(base.shape), "extended_shape": list(extended.shape),
            "widths": widths.tolist(), "weight_type": "seeded_random_CNN_no_OCR_accuracy_claim"}
with torch.no_grad():
    for mode in ("eval", "train"):
        left, right = copy.deepcopy(extractor), copy.deepcopy(extractor)
        left.train(mode == "train")
        right.train(mode == "train")
        a, b = left(base), right(extended)
        valid = CRNN._calc_cnn_output_width(64)
        bn_probe[mode + "_max_valid_feature_difference"] = float((a[:, :, :, :valid] - b[:, :, :, :valid]).abs().max())
        bn_probe[mode + "_stem_running_mean_difference"] = float((left.stem[1].running_mean - right.stem[1].running_mean).abs().max())

# 同一索引序列每轮先排序再分块，只有 blocks 的访问顺序变动。
num_train = sum(stats[Path(p).stem.removesuffix("_annotation")]["samples"] for p in config["TRAIN_ANNOTATION"])
summary = {
    "current_keys": len(keys), "duplicate_keys": len(keys) - len(set(keys)),
    "historical_keys": len(old_keys), "same_order_as_119c7b8": keys == old_keys,
    "datasets": stats, "training_samples": num_train,
    "batches_per_epoch": (num_train + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"],
    "last_batch": num_train % config["BATCH_SIZE"],
    "training_characters": sum(training_counts.values()), "training_frequency_bins": dict(frequency),
    "rare_examples": {c: {s: counts[s][c] for s in counts} for c in "疽荥嵇淦肽瀚晏笑坤"},
    "synthesis_in_training": "recognition/data/synthesis_annotation.txt" in config["TRAIN_ANNOTATION"],
    "selected_unique_images": len(details), "unresolved_selected": unresolved,
    "red_seal_samples": len(seal),
    "red_seal_sources": dict(Counter(r["source"] for r in seal)),
    "legacy_changes_red_seal_images": sum(r["legacy_changes"] > 0 for r in seal),
    "legacy_unchanged_red_seal": [r["path"] for r in seal if not r["legacy_changes"]],
    "legacy_erases_neutral_dark_images": sum(r["legacy_erased_neutral_dark"] > 0 for r in seal),
    "legacy_neutral_dark_pixels": sum(r["legacy_erased_neutral_dark"] for r in seal),
    "legacy_crop_changes_seal": sum(r["legacy_crop_wh"] != r["original_wh"] for r in seal),
    "legacy_geometry_changes_over_one_pixel_seal": [r["path"] for r in seal if abs(r["legacy_processed_width"] - r["processed_wh"][0]) > 1],
    "seal_height_range": [min(r["original_wh"][1] for r in seal), max(r["original_wh"][1] for r in seal)],
    "seal_ctc_margin_range": [min(r["ctc_steps"] - r["ctc_needed"] for r in seal),
                              max(r["ctc_steps"] - r["ctc_needed"] for r in seal)],
    "selected_ctc_infeasible": [r["path"] for r in details if r["ctc_steps"] < r["ctc_needed"]],
    "sampled_modes_32_per_source": mode_samples, "augmentation": augmentation, "bn_probe": bn_probe,
    "CNN_RIGHT_CONTEXT": CNN_RIGHT_CONTEXT,
}
manifest["hashes_unchanged_after_audit"] = all(sha256(ROOT / p) == h for p, h in manifest["hashes"].items())
if not manifest["hashes_unchanged_after_audit"]:
    raise RuntimeError("Audit inputs changed while running; rerun before using these results")
for name, payload in (("manifest.json", manifest), ("summary.json", summary), ("samples.json", details)):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
