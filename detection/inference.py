"""统一检测 PyTorch/GPU 推理：整图缩放、连通域解码、NMS 和原图坐标还原。

PyTorch 检查点默认使用 GPU（不可用时回退 CPU），默认把整图缩放到固定
INPUT_SIZE 方形画布；DYNAMIC_INPUT=True 时才按原图宽高比选择 32 的倍数尺寸。
本模块只接受 `.pth` 检查点；ONNX/OpenVINO 的 CPU 部署入口是根目录
detection_inference_onnx.py，该入口独立实现，不依赖本模块。所有输出坐标均
还原到原图像素空间。
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw

# 兼容从仓库根目录直接运行此文件。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.components.image import IMAGE_EXTENSIONS, load_rgb_image
from detection.utils.postprocess import (
    CROP_INDEX_DIGITS,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    crop_path,
    decode_maps,
    expand_predictions,
    remove_stale_crops,
    suppress_boxes,
)
from detection.components.transforms import (
    dynamic_input_size,
    normalize_input_size,
    prepare_image,
    restore_boxes,
)
from detection.utils import load_checkpoint


# 统一检测模型；更换训练结果时修改此路径。
MODEL_PATH = Path("detection/checkpoints/ocr_experiment_1793283532819507511/best_f1_detection_epoch_560_f1_0.97158.pth")
IMAGE_PATH = Path("detection/data/test_images")  # 输入图片或图片目录，不直接读取 PDF 文件。
OUTPUT_DIR = Path("detection/data/test_results")  # 推理输出目录；重复运行覆盖本次输入对应的同名图片和标签。
DEVICE = "auto"  # PyTorch 自动优先使用 CUDA，无可用 GPU 时使用 CPU。
INPUT_SIZE = 768  # 固定方形输入边长，单位 px；与训练尺寸一致，int 表示正方形，也可设为 (高, 宽)。
DYNAMIC_INPUT = False  # 默认固定方形画布；True 时按原图宽高比选择 32 的倍数尺寸，超宽图短边会被压到 64。
SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD  # 文字置信度阈值，取值范围为 (0, 1)。
NMS_IOU_THRESHOLD = DEFAULT_NMS_IOU_THRESHOLD  # 重复框抑制的交并比阈值，取值范围为 (0, 1]。
SAVE_CROPS = False  # 启用时保存原图文字框 PNG 裁片，供识别使用。
CROP_DIR = Path("detection/data/crops/test_images")  # 裁片平铺目录；启用时覆盖同名裁片并清理该原图多余的旧序号。


class DocumentDetector:
    """加载统一检测 PyTorch 检查点，共用整图预处理、后处理和原图坐标还原。"""

    def __init__(self, model_path, device="cpu", max_input_size=None, dynamic_input=None):
        """max_input_size 和 dynamic_input 为 None 时读取模块常量，便于入口集中配置。"""
        self.max_input_size = normalize_input_size(
            INPUT_SIZE if max_input_size is None else max_input_size)
        self.dynamic_input = bool(DYNAMIC_INPUT if dynamic_input is None else dynamic_input)
        self.static_input_size = None

        if Path(model_path).suffix.lower() != ".pth":
            raise ValueError(f"PyTorch detection requires a .pth checkpoint: {model_path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, _ = load_checkpoint(model_path, device)

    @property
    def input_size(self):
        """固定输入尺寸或动态模式的最大输入尺寸。"""
        return self.static_input_size or self.max_input_size

    def _target_size(self, image_size):
        """返回固定输入尺寸，或动态模式下按原图宽高比选择的尺寸。"""
        if self.dynamic_input:
            return dynamic_input_size(image_size, self.max_input_size)
        return self.input_size

    def infer(self, batch):
        """执行 NCHW 前向，计时含设备传输及同步，不含图片预处理和解码。"""
        started = time.perf_counter()
        with torch.inference_mode():
            score, geometry = self.model(batch.to(self.device))
        score, geometry = score.cpu().numpy(), geometry.cpu().numpy()
        self.last_infer_ms = (time.perf_counter() - started) * 1000
        return score, geometry

    def warmup(self, count):
        """预热正式推理使用的形状；动态模型额外预热一个宽高互换的形状。"""
        if not isinstance(count, int) or count < 0:
            raise ValueError("warmup count must be a nonnegative integer")
        sizes = {self.input_size}
        if self.dynamic_input:
            height, width = self.input_size
            sizes.add((height, max(64, (width // 2) // 32 * 32)))
        for height, width in sizes:
            sample = torch.zeros(1, 3, height, width)
            for _ in range(count):
                self.infer(sample)

    def predict(self, image, threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD):
        """检测 RGB 图片，返回含原图像素 xyxy 框和置信度的列表，不修改输入图片。

        结果经过置信度过滤及 NMS，按从上到下、从左到右排序。
        """
        if not 0 < threshold < 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Invalid detection thresholds")
        tensor, metadata = prepare_image(image, self._target_size(image.size), dynamic=False)
        score, geometry = self.infer(tensor[None])
        boxes, scores = decode_maps(score[0], geometry[0], threshold)
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, iou_threshold)
        # 仅提供便于浏览的视觉顺序，不恢复表格结构。
        keep = sorted(keep, key=lambda index: (boxes[index, 1], boxes[index, 0]))
        return [{"box": boxes[index].tolist(), "score": float(scores[index])} for index in keep]


def draw_predictions(image, predictions):
    """在原图副本上用红色框标出检测结果，返回新图片；不修改传入图片。

    画法与 run_inference 保存的预览图一致，供需要复用同一出框效果的调用方使用。
    """
    visual = image.copy()
    draw = ImageDraw.Draw(visual)
    for prediction in predictions:
        draw.rectangle(prediction["box"], outline="red", width=2)
    return visual


def run_inference(model_path, image_path, output_dir, device="cpu", *,
                  threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD,
                  save_crops=False, crop_dir=None, warmup_runs=0,
                  expand_ratio_w=0.0, expand_ratio_h=0.0):
    """批量保存画框图、bbox4 标签及可选裁片，统计包含预处理和后处理的单图耗时。"""
    if save_crops and crop_dir is None:
        raise ValueError("crop_dir is required when save_crops=True")
    if not all(np.isfinite(ratio) and ratio >= 0 for ratio in (expand_ratio_w, expand_ratio_h)):
        raise ValueError("Expansion ratios must be finite and nonnegative")

    image_path, output_dir = Path(image_path), Path(output_dir)
    paths = (sorted(path for path in image_path.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
             if image_path.is_dir() else [image_path])
    if not paths:
        raise ValueError(f"No images in {image_path}")

    # 标签和裁片名按 stem 生成，同名不同扩展名的输入不能共用一套产物。
    stems = {}
    for path in paths:
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem for labels: {stems[path.stem]} and {path}")
        stems[path.stem] = path

    detector = DocumentDetector(model_path, device)
    if warmup_runs:
        detector.warmup(warmup_runs)
    print(f"模型: {model_path}; 预热: {warmup_runs} 次")

    total_boxes, total_infer_ms, total_detect_ms = 0, 0.0, 0.0
    wall_started = time.perf_counter()
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    if save_crops:
        Path(crop_dir).mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = load_rgb_image(path)
        started = time.perf_counter()
        predictions = detector.predict(image, threshold, iou_threshold)
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_boxes += len(predictions)
        infer_ms = detector.last_infer_ms
        total_infer_ms += infer_ms
        total_detect_ms += elapsed_ms

        predictions = expand_predictions(predictions, image.size, expand_ratio_w, expand_ratio_h)
        if save_crops:
            for index, prediction in enumerate(predictions):
                box = prediction["box"]
                crop_box = (int(np.floor(box[0])), int(np.floor(box[1])),
                            int(np.ceil(box[2])), int(np.ceil(box[3])))
                image.crop(crop_box).save(crop_path(crop_dir, path.stem, index))
            remove_stale_crops(crop_dir, path.stem, len(predictions))

        draw_predictions(image, predictions).save(images_dir / path.name)
        # bbox4 使用原图像素坐标，不附置信度或伪造文字；无检测框时写空标签。
        label_text = "".join(" ".join(str(float(value)) for value in prediction["box"]) + "\n"
                             for prediction in predictions)
        (labels_dir / f"gt_{path.stem}.txt").write_text(label_text, encoding="utf-8")
        print(f"{path}: {len(predictions)} boxes, 前向 {infer_ms:.1f} ms, "
              f"检测 {elapsed_ms:.1f} ms", flush=True)

    wall_seconds = time.perf_counter() - wall_started
    summary = {"images": len(paths), "boxes": total_boxes,
               "mean_infer_ms": total_infer_ms / len(paths),
               "mean_detect_ms": total_detect_ms / len(paths),
               "wall_seconds": wall_seconds, "images_per_second": len(paths) / wall_seconds}
    print(f"完成: {len(paths)} 张, {total_boxes} 框; 前向均值 {summary['mean_infer_ms']:.1f} ms; "
          f"检测均值 {summary['mean_detect_ms']:.1f} ms; 总耗时 {wall_seconds:.2f} s; "
          f"吞吐量 {summary['images_per_second']:.2f} 张/s; 输出: {output_dir}")
    return summary


def main():
    """按顶部常量执行 PyTorch/GPU 推理；ONNX CPU 部署使用根目录独立入口。"""
    run_inference(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR, DEVICE,
                  threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD,
                  save_crops=SAVE_CROPS, crop_dir=CROP_DIR)


if __name__ == "__main__":
    main()
