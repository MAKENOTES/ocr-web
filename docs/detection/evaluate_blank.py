"""评估 EAST 是否需要 CTC blank；CPU 机制实验及可选的真实标签抽样。

在仓库根目录运行：
    .venv/bin/python docs/detection/evaluate_blank.py
    .venv/bin/python docs/detection/evaluate_blank.py \
        --image-dir detection/data/val_images --label-dir detection/data/val_labels

不加载权重、不训练、不改写数据。输出是机制证据，不是检测模型准确率。
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import platform
import random
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from detection.components.dataset import OCRDataset
from detection.components.ground_truth import generate_ground_truth
from detection.components.loss import DiceLoss, EASTLoss
from detection.components.transforms import get_transforms
from detection.models.east_model import EASTModel
import detection_inference_640 as inference
from recognition.components.converter import LabelConverter


def evaluate_ctc() -> dict:
    """复用 converter，并穷举路径，避免把删除 blank 误当成普通去重。"""
    with TemporaryDirectory(prefix="ocr_blank_") as directory:
        keys = Path(directory) / "keys.txt"
        keys.write_text("A\n \n", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            converter = LabelConverter(str(keys))
    a = converter.char2idx["A"]
    space = converter.char2idx[" "]
    cases = [[a, a, a], [a, 0, a], [0, 0, 0], [a, space, a]]
    scores = torch.nn.functional.one_hot(torch.tensor(cases), converter.num_classes).float()
    decoded = converter.decode(scores, layout="BTC")
    paths = torch.tensor(list(product(range(converter.num_classes), repeat=3)))
    texts = converter.decode(
        torch.nn.functional.one_hot(paths, converter.num_classes).float(), layout="BTC"
    )
    valid = [path.tolist() for path, text in zip(paths, texts) if text == "AA"]
    targets, lengths = converter.encode(["AA"])
    logits = torch.full((3, 1, converter.num_classes), -8.0)
    logits[0, 0, a] = logits[1, 0, 0] = logits[2, 0, a] = 8.0
    loss = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=False)
    normal = loss(logits.log_softmax(2), targets, torch.tensor([3]), lengths)
    # 对照只禁止 blank 路径，保留原来的类别编号和 CTC 定义。
    logits[:, :, 0] = -torch.inf
    suppressed = loss(logits.log_softmax(2), targets, torch.tensor([3]), lengths)
    return {
        "cases": [{"tokens": p, "decoded": t} for p, t in zip(cases, decoded)],
        "target": "AA", "time_steps": 3,
        "valid_paths_with_blank": len(valid),
        "valid_paths_without_blank": sum(0 not in path for path in valid),
        "ctc_loss_with_blank": float(normal),
        "ctc_loss_without_blank": float(suppressed) if torch.isfinite(suppressed) else "infinity",
        "encoded_targets": targets.tolist(),
        "target_contains_blank": bool((targets == 0).any()),
    }


def evaluate_head(input_size: tuple[int, int]) -> dict:
    model = EASTModel(pretrained=False).eval()
    with torch.inference_mode():
        score, geo = model(torch.zeros(1, 3, *input_size))
    logits = torch.linspace(-12, 12, 1001, dtype=torch.float64)
    explicit_text = torch.stack([torch.zeros_like(logits), logits], 1).softmax(1)[:, 1]
    return {
        "weights": "random initialization; shapes only",
        "score_shape": list(score.shape), "geometry_shape": list(geo.shape),
        "score_head_parameters": sum(p.numel() for p in model.score_conv.parameters()),
        "binary_softmax_sigmoid_max_error": float((explicit_text - logits.sigmoid()).abs().max()),
    }


def evaluate_background(input_size: tuple[int, int]) -> dict:
    """混合前景/背景与全背景分别求梯度；不把纯空白 batch 与混合 batch 混淆。"""
    h, w = input_size
    quad = np.array([[[w * .1, h * .2], [w * .9, h * .2],
                      [w * .9, h * .8], [w * .1, h * .8]]], dtype=np.float32)
    score, geo = generate_ground_truth(input_size, quad)
    target = torch.from_numpy(score)[None, None]
    true_geo = torch.from_numpy(geo)[None]
    with redirect_stdout(io.StringIO()):
        criterion = EASTLoss(lambda_iou=3.0)
    pred = torch.full_like(target, .5, requires_grad=True)
    pred_geo = torch.ones_like(true_geo, requires_grad=True)
    total, score_loss, geo_loss = criterion(target, pred, true_geo, pred_geo)
    total.backward()
    background = target == 0
    positive_only = torch.full_like(target, .5, requires_grad=True)
    DiceLoss()(positive_only * target, target).backward()

    empty_target = torch.zeros_like(target)
    empty_prediction = torch.full_like(target, .5, requires_grad=True)
    empty_geo = torch.ones_like(true_geo, requires_grad=True)
    empty_total, empty_score, empty_geo_loss = criterion(
        empty_target, empty_prediction, torch.zeros_like(true_geo), empty_geo
    )
    empty_total.backward()
    bce_prediction = torch.full_like(target, .5, requires_grad=True)
    bce = torch.nn.functional.binary_cross_entropy(bce_prediction, empty_target)
    bce.backward()
    return {
        "map_pixels": target.numel(), "positive_pixels": int(target.sum()),
        "mixed_score_loss": float(score_loss.detach()),
        "mixed_geo_loss": float(geo_loss.detach()),
        "mixed_background_score_gradient_mean": float(pred.grad[background].mean()),
        "mixed_foreground_score_gradient_mean": float(pred.grad[~background].mean()),
        "mixed_background_geo_gradient_max": float(pred_geo.grad[background.expand_as(pred_geo)].abs().max()),
        "positive_only_background_score_gradient_max": float(positive_only.grad[background].abs().max()),
        "empty_batch_score_loss": float(empty_score.detach()),
        "empty_batch_geo_loss": float(empty_geo_loss.detach()),
        "empty_batch_dice_gradient_mean": float(empty_prediction.grad.mean()),
        "empty_batch_bce_loss": float(bce.detach()),
        "empty_batch_bce_gradient_mean": float(bce_prediction.grad.mean()),
        "gradient_variable": "score probability p, not model weights or logits",
    }


def evaluate_postprocess(input_size: tuple[int, int]) -> dict:
    h, w = input_size
    boxes = np.array([[.1*w, .2*h, .4*w, .8*h], [.6*w, .2*h, .9*w, .8*h]], dtype=np.float32)
    quads = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                      for x1, y1, x2, y2 in boxes], dtype=np.float32)
    score, geo = generate_ground_truth(input_size, quads)
    distances = np.maximum(np.expm1(geo), 0)
    results = {}
    for name, candidate in [("correct_background", score),
                            ("all_text", np.ones_like(score)),
                            ("all_background", np.zeros_like(score))]:
        decoded = inference.boxes_from_connected_components(
            candidate, distances, inference.SCORE_THRESHOLD, inference.MIN_AREA,
            inference.scaled_small_region_area(input_size),
        )
        components = cv2.connectedComponents(
            (candidate > inference.SCORE_THRESHOLD).astype(np.uint8), connectivity=4
        )[0] - 1
        results[name] = {"connected_regions": components, "box_count": len(decoded),
                         "boxes": decoded.tolist()}
    return {"ground_truth_boxes": boxes.tolist(), "cases": results,
            "scope": "synthetic GT geometry held fixed; score intervention, not trained-model ablation"}


def audit_dataset(image_dir: Path, label_dir: Path, limit: int, seed: int,
                  input_size: tuple[int, int]) -> dict:
    for directory in (image_dir, label_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    dataset = OCRDataset(str(image_dir), str(label_dir), image_extensions=(".png",),
                         require_labels=True, transform=get_transforms(False, input_size))
    if not len(dataset):
        raise ValueError(f"No PNG images found in {image_dir}")
    selected = sorted(random.Random(seed).sample(range(len(dataset)), min(limit, len(dataset))))
    records = []
    for index in selected:
        image_path = Path(dataset.image_paths[index])
        label_path = Path(dataset.label_paths[index])
        alternatives = [label_dir / f"{image_path.stem}.txt", label_dir / f"gt_{image_path.stem}.txt"]
        if sum(path.is_file() for path in alternatives) != 1:
            raise ValueError(f"Ambiguous label paths for {image_path}: {alternatives}")
        # 当前 loader 对坏标注会打印后跳过；评估必须显式失败，防止美化结果。
        messages = io.StringIO()
        with redirect_stdout(messages):
            try:
                sample = dataset[index]
            except Exception as error:
                raise ValueError(f"Cannot evaluate image {image_path}, label {label_path}: {error}") from error
        if messages.getvalue():
            raise ValueError(f"Dataset parser warning for {label_path}: {messages.getvalue()}")
        score = sample["score_map"]
        if not torch.isfinite(score).all() or not torch.isfinite(sample["geo_map"]).all():
            raise ValueError(f"Non-finite ground truth for {image_path}, {label_path}")
        records.append({"image": str(image_path), "label": str(label_path),
                        "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                        "positive_pixels": int((score > 0).sum()), "map_pixels": score.numel()})
    total = sum(row["map_pixels"] for row in records)
    positive = sum(row["positive_pixels"] for row in records)
    return {"scope": "directory sample; no manifest/split independence or model accuracy verified",
            "available_png_images": len(dataset), "sampled_images": len(records),
            "seed": seed, "positive_pixels": positive, "map_pixels": total,
            "background_fraction": (total - positive) / total,
            "zero_positive_maps": sum(row["positive_pixels"] == 0 for row in records),
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("blank-evaluation.json"))
    args = parser.parse_args()
    if (args.image_dir is None) != (args.label_dir is None):
        parser.error("--image-dir and --label-dir must be supplied together")
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    input_size = inference.INPUT_SIZE
    source_paths = [Path(__file__).relative_to(ROOT),
                    Path("recognition/components/converter.py"), Path("recognition/train.py"),
                    Path("recognition/models/model.py"), Path("detection/models/east_model.py"),
                    Path("detection/components/dataset.py"), Path("detection/components/ground_truth.py"),
                    Path("detection/components/transforms.py"), Path("detection/components/loss.py"),
                    Path("detection/train_resnet.py"), Path("detection_inference_640.py")]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu"},
        "input_size": list(input_size), "seed": args.seed,
        "source_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "ctc": evaluate_ctc(), "head": evaluate_head(input_size),
        "background": evaluate_background(input_size), "postprocess": evaluate_postprocess(input_size),
        "dataset": audit_dataset(args.image_dir, args.label_dir, args.max_samples, args.seed, input_size)
        if args.image_dir is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluation saved to {args.output}")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"source_sha256", "dataset"}}, ensure_ascii=False, indent=2))
    if result["dataset"]:
        print(json.dumps({key: value for key, value in result["dataset"].items() if key != "records"},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
