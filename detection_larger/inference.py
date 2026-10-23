"""使用独立的 EAST 连通域后处理，整图推理后还原 PDF/表格文字框坐标。"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw
from torchvision.ops import nms

# 兼容从仓库根目录直接运行此文件。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recognition.components.dataset import load_rgb_image
from detection_larger.components.dataset import IMAGE_EXTENSIONS
from detection_larger.components.onnx_runtime import OpenVINOBackend, session_options, validate_onnx_metadata
from detection_larger.components.transforms import STRIDE, normalize_input_size, prepare_image, restore_boxes
from detection_larger.components.postprocess import MIN_AREA, boxes_from_connected_components, scaled_small_region_area
from detection_larger.utils import load_checkpoint


# 文档专用检查点或 ONNX 模型；更换训练结果时修改此路径。
MODEL_PATH = Path("detection_larger/checkpoints/best_detection_epoch_603_loss_0.01189.pth")
IMAGE_PATH = Path("detection_larger/data/test_images")  # 输入图片或图片目录，不直接读取 PDF 文件。
OUTPUT_DIR = Path("detection_larger/data/test_results")  # 推理输出目录；重复运行覆盖本次输入对应的同名图片和标签。
DEVICE = "auto"  # PyTorch 自动优先使用 CUDA，无可用 GPU 时使用 CPU；可指定 cpu、cuda:0，ONNX 自动使用 CPU。
INPUT_SIZE = 640  # 推理输入边长，单位 px；也可设为 (高, 宽)，各边至少 64 且为 32 的倍数；ONNX/OpenVINO 须匹配导出尺寸。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，取值范围为 (0, 1)。
NMS_IOU_THRESHOLD = 0.3  # 重复框抑制的交并比阈值，取值范围为 (0, 1]。
SAVE_CROPS = False  # 启用时保存原图文字框 PNG 裁片，供 recognition 使用。
CROP_DIR = Path("detection_larger/data/crops/test_images")  # 裁片目录，所有原图的裁片平铺在此，不再按原图建子目录；启用 SAVE_CROPS 时覆盖同名裁片并清理该原图多余的旧序号裁片。
CROP_INDEX_DIGITS = 5  # 裁片文件名序号位数，序号从 1 开始并左侧补零，例如 page.png 的首个裁片为 page-00001.png。


def suppress_boxes(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD):
    """按置信度抑制重叠的 xyxy 框，返回保留框索引；空输入返回空索引。"""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    return nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32), iou_threshold).numpy()


def decode_maps(score, geometry, threshold=SCORE_THRESHOLD):
    """由置信度连通域解码输入图像素坐标框，按置信度降序返回，不限制数量。

    score 形状为 (1, H, W)，geometry 为 (4, H, W) 的 log1p 特征图距离。
    返回 (N, 4) 的 xyxy 框及 (N,) 置信度；非有限预测不参与出框。
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    score = np.asarray(score).squeeze(0)
    geometry = np.asarray(geometry)
    score = np.where(np.isfinite(score) & np.isfinite(geometry).all(axis=0), score, 0)
    boxes = boxes_from_connected_components(
        score, np.expm1(np.clip(geometry, 0, 12)), threshold, min_area=MIN_AREA,
        small_region_area=scaled_small_region_area(tuple(value * STRIDE for value in score.shape)),
    ).astype(np.float32).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(axis=1)]
    confidence = []
    for box in boxes:
        x0, y0 = np.maximum(np.floor(box[:2] / STRIDE), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(box[2:] / STRIDE), score.shape[::-1]).astype(int)
        region = score[y0:y1, x0:x1]
        confidence.append(float(region.max()) if region.size else 0.0)
    confidence = np.asarray(confidence, dtype=np.float32)
    order = np.argsort(-confidence, kind="stable")
    return boxes[order], confidence[order]


class DocumentDetector:
    """加载文档检测模型，共用整图预处理和原图坐标后处理。"""

    def __init__(self, model_path, device="cpu", backend="auto"):
        """backend 按后缀选择或显式指定 OpenVINO；device=auto 为 PyTorch 优先选择 CUDA。"""
        self.input_size = normalize_input_size(INPUT_SIZE)
        suffix = Path(model_path).suffix.lower()
        if backend == "auto":
            backend = "onnx" if suffix == ".onnx" else "pytorch"
        if backend not in {"onnx", "openvino", "pytorch"}:
            raise ValueError(f"Unknown document backend: {backend}")
        expected_suffix = ".pth" if backend == "pytorch" else ".onnx"
        if suffix != expected_suffix:
            raise ValueError(f"Backend {backend} requires {expected_suffix}: {model_path}")
        self.backend = backend
        self.openvino = None
        if device == "auto":
            device = "cuda" if Path(model_path).suffix.lower() != ".onnx" and torch.cuda.is_available() else "cpu"
        self.device = device
        self.session = None
        if backend == "openvino":
            if device != "cpu":
                raise ValueError("Document OpenVINO backend requires device='cpu'")
            self.openvino = OpenVINOBackend(model_path)
            exported_size = tuple(dim.get_length() for dim in self.openvino.compiled.input("image").partial_shape[2:])
            if exported_size != self.input_size:
                raise ValueError(
                    f"INPUT_SIZE {self.input_size} does not match exported model input size "
                    f"{exported_size}: {model_path}"
                )
        elif backend == "onnx":
            import onnxruntime as ort
            if device != "cpu":
                raise ValueError("Document ONNX backend requires device='cpu'")
            options = session_options()
            self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                                providers=["CPUExecutionProvider"])
            validate_onnx_metadata(self.session.get_modelmeta().custom_metadata_map, model_path)
            inputs = self.session.get_inputs()
            if (len(inputs) != 1 or inputs[0].name != "image" or inputs[0].type != "tensor(float)"
                    or len(inputs[0].shape) != 4
                    or inputs[0].shape[1] != 3
                    or {value.name for value in self.session.get_outputs()} != {"score", "geometry"}):
                raise ValueError(f"Incompatible document ONNX inputs/outputs: {model_path}")
            exported_size = tuple(inputs[0].shape[2:])
            if exported_size != self.input_size:
                raise ValueError(
                    f"INPUT_SIZE {self.input_size} does not match exported model input size "
                    f"{exported_size}: {model_path}"
                )
        else:
            self.model, _ = load_checkpoint(model_path, device)

    def infer(self, batch):
        """执行 NCHW 前向，计时含设备传输及同步，不含图片预处理和解码。"""
        started = time.perf_counter()
        if self.openvino is not None:
            score, geometry = self.openvino.infer(batch.numpy())
        elif self.session is not None:
            score, geometry = self.session.run(["score", "geometry"], {"image": batch.numpy()})
        else:
            with torch.inference_mode():
                score, geometry = self.model(batch.to(self.device))
            score, geometry = score.cpu().numpy(), geometry.cpu().numpy()
        self.last_infer_ms = (time.perf_counter() - started) * 1000
        return score, geometry

    def warmup(self, count):
        """按 INPUT_SIZE 常量预热，计时汇总不包含这些前向。"""
        if not isinstance(count, int) or count < 0:
            raise ValueError("warmup count must be a nonnegative integer")
        sample = torch.zeros(1, 3, *self.input_size)
        for _ in range(count):
            self.infer(sample)

    def predict(self, image, threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD):
        """检测 RGB 图片，返回含原图像素 xyxy 框和置信度的列表，不修改输入图片。

        输入尺寸使用脚本顶部 INPUT_SIZE 常量；结果经过置信度过滤及 NMS，按从上到下、从左到右排序。
        """
        if not 0 < threshold < 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Invalid detection thresholds")
        tensor, metadata = prepare_image(image, self.input_size)
        score, geometry = self.infer(tensor[None])
        boxes, scores = decode_maps(score[0], geometry[0], threshold)
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, iou_threshold)
        # 仅提供便于浏览的视觉顺序，不恢复表格结构。
        keep = sorted(keep, key=lambda index: (boxes[index, 1], boxes[index, 0]))
        return [{"box": boxes[index].tolist(), "score": float(scores[index])} for index in keep]


def crop_path(crop_dir, image_stem, index):
    """按 <原图 stem>-<序号>.png 生成平铺裁片路径；序号从 1 开始，与原图逐一对应。"""
    return Path(crop_dir) / f"{image_stem}-{index + 1:0{CROP_INDEX_DIGITS}d}.png"


def remove_stale_crops(crop_dir, image_stem, keep_count):
    """删除该原图序号超出本次框数的旧裁片，保留其他原图的结果和无关文件。"""
    prefix = f"{image_stem}-"
    for path in Path(crop_dir).iterdir():
        if not path.is_file() or not path.name.startswith(prefix) or path.suffix.lower() != ".png":
            continue
        number = path.stem[len(prefix):]
        if number.isascii() and number.isdigit() and int(number) > keep_count:
            path.unlink()


def expand_predictions(predictions, image_size, ratio_w, ratio_h):
    """按框高向左右和上下扩边，裁片、标签和预览一致；坐标限制在原图内。"""
    if not all(np.isfinite(ratio) and ratio >= 0 for ratio in (ratio_w, ratio_h)):
        raise ValueError("Expansion ratios must be finite and nonnegative")
    width, height = image_size
    result = []
    for prediction in predictions:
        x0, y0, x1, y1 = prediction["box"]
        box_height = y1 - y0
        box = [max(0, x0 - box_height * ratio_w), max(0, y0 - box_height * ratio_h),
               min(width, x1 + box_height * ratio_w), min(height, y1 + box_height * ratio_h)]
        result.append({**prediction, "box": box})
    return result


def run_inference(model_path, image_path, output_dir, device="cpu", *,
                  threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD,
                  save_crops=False, crop_dir=None, backend="auto", warmup_runs=0,
                  expand_ratio_w=0.0, expand_ratio_h=0.0):
    """批量保存画框图、bbox4 标签及可选裁片，统计包含预处理和后处理的单图耗时。"""
    if save_crops and crop_dir is None:
        raise ValueError("crop_dir is required when save_crops=True")
    if not all(np.isfinite(ratio) and ratio >= 0 for ratio in (expand_ratio_w, expand_ratio_h)):
        raise ValueError("Expansion ratios must be finite and nonnegative")
    image_path, output_dir = Path(image_path), Path(output_dir)
    paths = sorted(p for p in image_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if image_path.is_dir() else [image_path]
    if not paths:
        raise ValueError(f"No images in {image_path}")
    # 训练标签和裁片名都按 stem 生成，同名不同扩展名的输入不能共用一套产物。
    stems = {}
    for path in paths:
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem for labels: {stems[path.stem]} and {path}")
        stems[path.stem] = path
    detector = DocumentDetector(model_path, device, backend=backend)
    if warmup_runs:
        detector.warmup(warmup_runs)
    print(f"模型: {model_path}; 后端: {detector.backend}; 预热: {warmup_runs} 次")
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
        visual = image.copy()
        draw = ImageDraw.Draw(visual)
        for index, prediction in enumerate(predictions):
            box = prediction["box"]
            draw.rectangle(box, outline="red", width=2)
            if save_crops:
                crop_box = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
                image.crop(crop_box).save(crop_path(crop_dir, path.stem, index))
        if save_crops:
            remove_stale_crops(crop_dir, path.stem, len(predictions))
        visual.save(images_dir / path.name)
        # bbox4 使用原图像素坐标，不附置信度或伪造文字；无检测框时写空标签。
        label_text = "".join(" ".join(str(float(value)) for value in prediction["box"]) + "\n"
                             for prediction in predictions)
        (labels_dir / f"gt_{path.stem}.txt").write_text(label_text, encoding="utf-8")
        print(f"{path}: {len(predictions)} boxes, 前向 {infer_ms:.1f} ms, 检测 {elapsed_ms:.1f} ms", flush=True)
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
    """按顶部常量推理，覆盖本次输入对应的同名产物。"""
    run_inference(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR, DEVICE,
                  threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD,
                  save_crops=SAVE_CROPS, crop_dir=CROP_DIR)


if __name__ == "__main__":
    main()
