"""沿用原 EAST 连通域后处理，整图推理后还原 PDF/表格文字框坐标。"""

import argparse
import json
import sys
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
from detection_larger.components.transforms import STRIDE, prepare_image, restore_boxes
from detection_inference_640 import boxes_from_connected_components, scaled_small_region_area
from detection_larger.models.east_model import ONNX_CONFIG_KEY, load_checkpoint, validate_config


def suppress_boxes(boxes, scores, iou_threshold=0.3):
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    return nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32), iou_threshold).numpy()


def decode_maps(score, geometry, threshold=0.7, top_k=5000):
    """复用原检测的连通域及边缘几何统计，geo 输入仍为 log1p 特征图距离。"""
    if not 0 < threshold < 1 or top_k <= 0:
        raise ValueError("threshold must be in (0, 1); top_k must be positive")
    score = np.asarray(score).squeeze(0)
    geometry = np.asarray(geometry)
    score = np.where(np.isfinite(score) & np.isfinite(geometry).all(axis=0), score, 0)
    boxes = boxes_from_connected_components(
        score, np.expm1(np.clip(geometry, 0, 12)), threshold, min_area=2,
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
    order = np.argsort(-confidence, kind="stable")[:top_k]
    return boxes[order], confidence[order]


class DocumentDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.session = None
        if Path(model_path).suffix.lower() == ".onnx":
            import onnxruntime as ort
            if device != "cpu":
                raise ValueError("Document ONNX backend currently supports --device cpu")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.config = json.loads(self.session.get_modelmeta().custom_metadata_map[ONNX_CONFIG_KEY])
            validate_config(self.config)
        else:
            self.model, checkpoint = load_checkpoint(model_path, device)
            self.config = checkpoint["config"]

    def predict(self, image, threshold=0.7, iou_threshold=0.3, top_k=5000):
        if not 0 < threshold < 1 or not 0 < iou_threshold <= 1 or top_k <= 0:
            raise ValueError("Invalid detection thresholds or top_k")
        tensor, metadata = prepare_image(image, self.config["input_size"])
        if self.session is not None:
            score, geometry = self.session.run(["score", "geometry"], {"image": tensor[None].numpy()})
        else:
            with torch.inference_mode():
                score, geometry = self.model(tensor[None].to(self.device))
            score, geometry = score.cpu().numpy(), geometry.cpu().numpy()
        boxes, scores = decode_maps(score[0], geometry[0], threshold, top_k)
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, iou_threshold)
        # 仅提供便于浏览的视觉顺序，不恢复表格结构。
        keep = sorted(keep, key=lambda index: (boxes[index, 1], boxes[index, 0]))
        return [{"box": boxes[index].tolist(), "score": float(scores[index])} for index in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="文档专用 .pth 或 .onnx")
    parser.add_argument("--input", required=True, type=Path, help="图片或图片目录，不直接读取 PDF 文件")
    parser.add_argument("--output", type=Path, default=Path("detection_larger/outputs/prediction"), help="新的输出目录")
    parser.add_argument("--device", default="cpu", help="PyTorch 设备；ONNX 使用 cpu")
    parser.add_argument("--threshold", type=float, default=0.7, help="文字置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="重复框抑制的 IoU 阈值")
    parser.add_argument("--top-k", type=int, default=5000, help="每张图片最多保留的连通域候选框数")
    parser.add_argument("--save-crops", action="store_true", help="保存原图文字框裁片，供 recognition 使用")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {args.output}")
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if args.input.is_dir() else [args.input]
    if not paths:
        raise ValueError(f"No images in {args.input}")
    detector = DocumentDetector(args.model, args.device)
    args.output.mkdir(parents=True)
    for path in paths:
        image = load_rgb_image(path)
        predictions = detector.predict(image, args.threshold, args.nms_iou, args.top_k)
        # 文件夹包含扩展名，避免 foo.jpg 与 foo.png 冲突。
        destination = args.output / path.name
        destination.mkdir()
        visual = image.copy()
        draw = ImageDraw.Draw(visual)
        for index, prediction in enumerate(predictions):
            box = prediction["box"]
            draw.rectangle(box, outline="red", width=2)
            if args.save_crops:
                crop_box = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
                image.crop(crop_box).save(destination / f"crop_{index:04d}.png")
        visual.save(destination / "overlay.png")
        result = {"image": str(path.resolve()), "width": image.width, "height": image.height,
                  "coordinate_format": "xyxy-original-pixels", "detections": predictions}
        (destination / "boxes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path}: {len(predictions)} boxes", flush=True)


if __name__ == "__main__":
    main()
