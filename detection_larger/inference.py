"""使用独立的 EAST 连通域后处理，整图推理后还原 PDF/表格文字框坐标。"""

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
from detection_larger.components.postprocess import MIN_AREA, boxes_from_connected_components, scaled_small_region_area
from detection_larger.models.east_model import ONNX_CONFIG_KEY, load_checkpoint, validate_config


# 文档专用检查点或 ONNX 模型；更换训练结果时修改此路径。
MODEL_PATH = Path("detection_larger/checkpoints/ocr_experiment_1789136798452328557/best_detection_epoch_584_loss_0.01150.pth")
IMAGE_PATH = Path("detection_larger/data/test_images")  # 输入图片或图片目录，不直接读取 PDF 文件。
OUTPUT_DIR = Path("detection_larger/data/test_results")  # 本次输出目录，必须尚不存在，避免覆盖结果。
DEVICE = "auto"  # PyTorch 自动优先使用 CUDA，无可用 GPU 时使用 CPU；可指定 cpu、cuda:0，ONNX 自动使用 CPU。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，取值范围为 (0, 1)。
NMS_IOU_THRESHOLD = 0.3  # 重复框抑制的交并比阈值，取值范围为 (0, 1]。
SAVE_CROPS = True  # 启用时保存原图文字框 PNG 裁片，供 recognition 使用。
CROP_DIR = Path("detection_larger/data/crops/test_images")  # 裁片根目录，按原图文件名分目录保存，仅在 SAVE_CROPS 启用时创建。


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

    def __init__(self, model_path, device="cpu"):
        """按后缀加载 ONNX 或 PyTorch 权重及输入配置；auto 优先为 PyTorch 选择 CUDA。"""
        if device == "auto":
            device = "cuda" if Path(model_path).suffix.lower() != ".onnx" and torch.cuda.is_available() else "cpu"
        self.device = device
        self.session = None
        if Path(model_path).suffix.lower() == ".onnx":
            import onnxruntime as ort
            if device != "cpu":
                raise ValueError("Document ONNX backend requires device='cpu'")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.config = json.loads(self.session.get_modelmeta().custom_metadata_map[ONNX_CONFIG_KEY])
            validate_config(self.config)
        else:
            self.model, checkpoint = load_checkpoint(model_path, device)
            self.config = checkpoint["config"]

    def predict(self, image, threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD):
        """检测 RGB 图片，返回含原图像素 xyxy 框和置信度的列表，不修改输入图片。

        输入尺寸使用模型记录的配置；结果经过置信度过滤及 NMS，按从上到下、从左到右排序。
        """
        if not 0 < threshold < 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Invalid detection thresholds")
        tensor, metadata = prepare_image(image, self.config["input_size"])
        if self.session is not None:
            score, geometry = self.session.run(["score", "geometry"], {"image": tensor[None].numpy()})
        else:
            with torch.inference_mode():
                score, geometry = self.model(tensor[None].to(self.device))
            score, geometry = score.cpu().numpy(), geometry.cpu().numpy()
        boxes, scores = decode_maps(score[0], geometry[0], threshold)
        boxes = restore_boxes(boxes, metadata, image.size)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = suppress_boxes(boxes, scores, iou_threshold)
        # 仅提供便于浏览的视觉顺序，不恢复表格结构。
        keep = sorted(keep, key=lambda index: (boxes[index, 1], boxes[index, 0]))
        return [{"box": boxes[index].tolist(), "score": float(scores[index])} for index in keep]


def main():
    """按顶部常量批量推理，写入画框图片、bbox4 标签及可选裁片。

    路径相对于当前工作目录；输出目录或同名裁片目录已存在时拒绝覆盖。
    """
    image_path, output_dir = Path(IMAGE_PATH), Path(OUTPUT_DIR)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output_dir}")
    paths = sorted(p for p in image_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if image_path.is_dir() else [image_path]
    if not paths:
        raise ValueError(f"No images in {image_path}")
    # 训练标签按 stem 配对，同名不同扩展名的输入不能共用一份标签。
    stems = {}
    for path in paths:
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem for labels: {stems[path.stem]} and {path}")
        stems[path.stem] = path
    if SAVE_CROPS:
        for path in paths:
            crop_destination = Path(CROP_DIR) / path.name
            if crop_destination.exists():
                raise FileExistsError(f"Crop output already exists; choose a new CROP_DIR: {crop_destination}")
    detector = DocumentDetector(MODEL_PATH, DEVICE)
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    for path in paths:
        image = load_rgb_image(path)
        predictions = detector.predict(image, SCORE_THRESHOLD, NMS_IOU_THRESHOLD)
        if SAVE_CROPS:
            crop_destination = Path(CROP_DIR) / path.name
            crop_destination.mkdir(parents=True)
        visual = image.copy()
        draw = ImageDraw.Draw(visual)
        for index, prediction in enumerate(predictions):
            box = prediction["box"]
            draw.rectangle(box, outline="red", width=2)
            if SAVE_CROPS:
                crop_box = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
                image.crop(crop_box).save(crop_destination / f"crop_{index:04d}.png")
        visual.save(images_dir / path.name)
        # bbox4 使用原图像素坐标，不附置信度或伪造文字；无检测框时写空标签。
        label_text = "".join(" ".join(str(float(value)) for value in prediction["box"]) + "\n"
                             for prediction in predictions)
        (labels_dir / f"gt_{path.stem}.txt").write_text(label_text, encoding="utf-8")
        print(f"{path}: {len(predictions)} boxes", flush=True)


if __name__ == "__main__":
    main()
