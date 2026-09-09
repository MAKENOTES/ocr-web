"""使用真实标注对 CPU 后端测 CER、整行准确率及吞吐；保存可复核的逐图预测。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from Levenshtein import distance
from PIL import Image
from torch.utils.data import DataLoader, Subset

from recognition.components.converter import LabelConverter
from recognition.components.dataset import TextRecognitionDataset, dynamic_padding_collate_fn
from recognition_inference import load_backend


def select_indices(size: int, limit: int = 0, seed: int = 20260908) -> list[int]:
    if size < 1 or limit < 0:
        raise ValueError("数据集不能为空，limit 不能为负")
    if not limit or limit >= size:
        return list(range(size))
    return sorted(np.random.default_rng(seed).choice(size, limit, replace=False).tolist())


def evaluate_backend(backend, dataset, converter, *, batch_size=1, limit=0,
                     seed=20260908, sort_by_width=False, num_workers=0):
    indices = select_indices(len(dataset), limit, seed)
    if sort_by_width:
        def ratio(index):
            with Image.open(dataset.image_paths[index]) as image:
                return image.width / image.height
        indices.sort(key=ratio)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        num_workers=num_workers, collate_fn=dynamic_padding_collate_fn)
    # 单独预热，不计入数据集吞吐；各形状首次编译仍计入真实数据遍历。
    warm_images, _, warm_widths = dynamic_padding_collate_fn([dataset[indices[0]]])
    for _ in range(5):
        backend.infer(warm_images, warm_widths)
    predictions, latencies, decode_ms = [], [], 0.0
    total_errors = total_chars = correct = offset = 0
    started = time.perf_counter()
    for images, labels, widths in loader:
        start = time.perf_counter()
        output = backend.infer(images, widths)
        latencies.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        texts = converter.decode(output, layout="TBC")
        decode_ms += (time.perf_counter() - start) * 1000
        for label, text, width in zip(labels, texts, widths.tolist()):
            errors = distance(label, text)
            total_errors += errors
            total_chars += len(label)
            correct += label == text
            index = indices[offset]
            predictions.append(dict(index=index, path=dataset.image_paths[index],
                                    label=label, prediction=text, errors=errors, width=width))
            offset += 1
        if offset % 1000 < batch_size:
            print(f"  已评估 {offset}/{len(indices)}", flush=True)
    wall_s = time.perf_counter() - started
    metrics = dict(samples=len(indices), characters=total_chars, edits=total_errors,
                   cer=total_errors / total_chars, exact_matches=correct,
                   exact_accuracy=correct / len(indices), wall_seconds=wall_s,
                   images_per_second=len(indices) / wall_s,
                   inference_ms_per_image=sum(latencies) / len(indices),
                   decode_ms_per_image=decode_ms / len(indices),
                   batch_latency_p50_ms=float(np.median(latencies)),
                   batch_latency_p95_ms=float(np.percentile(latencies, 95)),
                   batch_size=batch_size, sort_by_width=sort_by_width)
    predictions.sort(key=lambda row: row["index"])
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="auto", choices=["auto", "pytorch", "onnx", "openvino"])
    parser.add_argument("--annotation", default="recognition/data/val_annotation.txt")
    parser.add_argument("--keys_path", default="recognition/keys.txt")
    parser.add_argument("--output", required=True, help="JSON 报告路径，同时写 .predictions.jsonl")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 为全部，否则确定性随机抽样")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--sort_by_width", action="store_true")
    parser.add_argument("--allow_spinning", action="store_true")
    args = parser.parse_args()
    if args.num_threads < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("线程/batch 必须为正，workers 不能为负")
    torch.set_num_threads(1)
    converter = LabelConverter(args.keys_path)
    backend = load_backend(args.model_path, args.backend, args.keys_path, args.num_threads, args.allow_spinning)
    dataset = TextRecognitionDataset(args.annotation, 32, None, backend.num_channels, is_train=False)
    metrics, predictions = evaluate_backend(
        backend, dataset, converter, batch_size=args.batch_size, limit=args.limit,
        seed=args.seed, sort_by_width=args.sort_by_width, num_workers=args.num_workers)
    report = dict(config=vars(args), metrics=metrics, sha256={})
    for name in ["model_path", "annotation", "keys_path"]:
        report["sha256"][name] = hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
    bin_path = Path(args.model_path).with_suffix(".bin")
    if Path(args.model_path).suffix == ".xml" and bin_path.exists():
        report["sha256"]["model_bin"] = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with target.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
