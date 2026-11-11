"""检测推理流水线与输出：单图预测、标签/裁片写入和耗时汇总。

预处理、解码和裁片命名来自 runtime 内的生成模块，保证与仓库推理和导出契约一致。
"""
import time
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from .image import IMAGE_EXTENSIONS, load_rgb_image
from .postprocess import (
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    crop_path,
    decode_maps,
    expand_predictions,
    remove_stale_crops,
    suppress_boxes,
)
from .preprocessing import dynamic_input_size, prepare_image, restore_boxes


def warmup_backend(backend, dynamic, max_input_size, fixed_input_size, count):
    """按模型实际输入方式预热；动态模型额外预热一个宽高互换的形状。"""
    if count < 0:
        raise ValueError("warmup count must be nonnegative")
    sizes = {fixed_input_size if not dynamic else max_input_size}
    if dynamic:
        height, width = max_input_size
        sizes.add((height, max(64, (width // 2) // 32 * 32)))
    for height, width in sizes:
        sample = np.zeros((1, 3, height, width), dtype=np.float32)
        for _ in range(count):
            backend.infer(sample)


def predict_image(backend, image, threshold, iou_threshold, dynamic,
                  max_input_size, fixed_input_size):
    """检测单张 RGB 图片，返回原图坐标预测列表和前向毫秒数。"""
    target_size = dynamic_input_size(image.size, max_input_size) if dynamic else fixed_input_size
    tensor, metadata = prepare_image(image, target_size)
    started = time.perf_counter()
    score, geometry = backend.infer(tensor[None].numpy())
    infer_ms = (time.perf_counter() - started) * 1000

    boxes, scores = decode_maps(score[0], geometry[0], threshold)
    boxes = restore_boxes(boxes, metadata, image.size)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    keep = suppress_boxes(boxes, scores, iou_threshold)
    keep = sorted(keep, key=lambda index: (boxes[index, 1], boxes[index, 0]))
    return [{"box": boxes[index].tolist(), "score": float(scores[index])}
            for index in keep], infer_ms


def collect_images(image_path):
    """收集单张图片或目录内的支持图片；同名不同扩展名会覆盖标签，直接拒绝。"""
    image_path = Path(image_path)
    if image_path.is_dir():
        paths = sorted(path for path in image_path.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = [image_path]
    if not paths:
        raise ValueError(f"没有可推理的图片: {image_path}")
    stems = {}
    for path in paths:
        if path.stem in stems:
            raise ValueError(f"同名图片会覆盖标签: {stems[path.stem]} and {path}")
        stems[path.stem] = path
    return paths


def run_inference(backend, image_path, output_dir, *, dynamic, max_input_size,
                  fixed_input_size, threshold=DEFAULT_SCORE_THRESHOLD,
                  iou_threshold=DEFAULT_NMS_IOU_THRESHOLD, save_crops=False,
                  crop_dir=None, warmup_runs=0,
                  expand_ratio_w=0.0, expand_ratio_h=0.0):
    """批量保存画框图、bbox4 标签和可选裁片，并打印每图和汇总耗时。"""
    if save_crops and crop_dir is None:
        raise ValueError("crop_dir is required when save_crops=True")
    if warmup_runs:
        warmup_backend(backend, dynamic, max_input_size, fixed_input_size, warmup_runs)

    paths = collect_images(image_path)
    output_dir = Path(output_dir)
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    if save_crops:
        Path(crop_dir).mkdir(parents=True, exist_ok=True)

    total_boxes, total_infer_ms, total_detect_ms = 0, 0.0, 0.0
    wall_started = time.perf_counter()
    for path in paths:
        image = load_rgb_image(path)
        detect_started = time.perf_counter()
        predictions, infer_ms = predict_image(
            backend, image, threshold, iou_threshold, dynamic, max_input_size, fixed_input_size)
        detect_ms = (time.perf_counter() - detect_started) * 1000
        total_boxes += len(predictions)
        total_infer_ms += infer_ms
        total_detect_ms += detect_ms

        predictions = expand_predictions(predictions, image.size, expand_ratio_w, expand_ratio_h)
        visual = image.copy()
        draw = ImageDraw.Draw(visual)
        for index, prediction in enumerate(predictions):
            box = prediction["box"]
            draw.rectangle(box, outline="red", width=2)
            if save_crops:
                crop_box = (int(np.floor(box[0])), int(np.floor(box[1])),
                            int(np.ceil(box[2])), int(np.ceil(box[3])))
                image.crop(crop_box).save(crop_path(crop_dir, path.stem, index))
        if save_crops:
            remove_stale_crops(crop_dir, path.stem, len(predictions))

        visual.save(images_dir / path.name)
        # bbox4 使用原图像素坐标，不附置信度或伪造文字；无检测框时写空标签。
        label_text = "".join(" ".join(str(float(value)) for value in prediction["box"]) + "\n"
                             for prediction in predictions)
        (labels_dir / f"gt_{path.stem}.txt").write_text(label_text, encoding="utf-8")
        print(f"{path}: {len(predictions)} boxes, 前向 {infer_ms:.1f} ms, "
              f"检测 {detect_ms:.1f} ms", flush=True)

    wall_seconds = time.perf_counter() - wall_started
    summary = {"images": len(paths), "boxes": total_boxes,
               "mean_infer_ms": total_infer_ms / len(paths),
               "mean_detect_ms": total_detect_ms / len(paths),
               "wall_seconds": wall_seconds, "images_per_second": len(paths) / wall_seconds}
    print(f"完成: {len(paths)} 张, {total_boxes} 框; "
          f"前向均值 {summary['mean_infer_ms']:.1f} ms; "
          f"检测均值 {summary['mean_detect_ms']:.1f} ms; "
          f"总耗时 {wall_seconds:.2f} s; "
          f"吞吐量 {summary['images_per_second']:.2f} 张/s; 输出: {output_dir}")
    return summary
