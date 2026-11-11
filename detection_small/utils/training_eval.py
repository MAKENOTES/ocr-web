"""收集一次小图检测训练运行的结果，生成可离线打开的静态 HTML 分析报告。

报告覆盖训练曲线、验证集框级指标、不同输入尺寸/阈值工作点、检查点对比、
错误集中的图片和数据/部署检查。所有图表以 SVG 内嵌，问题图片以 JPEG data URI
内嵌，生成的 HTML 不依赖外部资源。
"""
import ast
import base64
import html as html_lib
import io
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from detection_small.components.dataset import OCRDataset, read_quads
from detection_small.components.eval_metrics import (
    DEFAULT_SCORE_THRESHOLD,
    decode_predictions,
    match_box_status,
    quads_to_aabb,
    scaled_small_region_area,
    summarize_counts,
)
from detection_small.components.postprocess import (
    DEFAULT_NMS_IOU_THRESHOLD,
    FEATURE_GEOMETRY_CLIP,
    suppress_boxes,
)
from detection_small.components.image import IMAGE_EXTENSIONS, load_rgb_image
from detection_small.components.preprocessing import (
    normalize_input_size,
    prepare_image,
    restore_boxes,
)
from detection_small.utils.checkpoint import load_model


REPO_ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录；报告内所有默认路径都相对它解析。
SMALL_DIR = REPO_ROOT / "detection_small"  # 小图检测包目录。
RUNS_DIR = SMALL_DIR / "runs"  # TensorBoard 事件目录的父目录。
REPORTS_DIR = SMALL_DIR / "reports"  # 报告默认输出目录，属于本地产物，不提交。
TRAIN_IMAGE_DIR = SMALL_DIR / "data" / "train_images"  # 训练图片目录，仅用于统计数量。
TRAIN_LABEL_DIR = SMALL_DIR / "data" / "train_labels"  # 训练标签目录，仅用于统计标签行数。
VAL_IMAGE_DIR = SMALL_DIR / "data" / "val_images"  # 验证图片目录，报告会逐个评估。
VAL_LABEL_DIR = SMALL_DIR / "data" / "val_labels"  # 验证标签目录，与验证图片严格配对。
INFERENCE_SOURCE = SMALL_DIR / "inference.py"  # 历史 PyTorch 推理入口，读取默认检查点和尺寸。
EXPORT_SOURCE = REPO_ROOT / "detection_small_export_onnx.py"  # 根目录 ONNX 导出入口。
TRAIN_SOURCE = SMALL_DIR / "train.py"  # 训练入口，读取输入尺寸和验证目录默认值。

INPUT_SCALE_FACTOR = 1.25  # 额外对比的输入尺寸倍率；256×768 会得到 320×960。
MAX_INPUT_SIDE = 1536  # 额外对比输入边长的上限，避免报告在超大图上过慢。
REPORT_THRESHOLDS = tuple(dict.fromkeys((float(DEFAULT_SCORE_THRESHOLD), 0.85)))  # 报告对比的置信度阈值。
REPORT_MATCH_IOU = 0.5  # 预测框与真值框的匹配 IoU 下限，与训练期指标口径一致。
REPORT_NMS_IOU = float(DEFAULT_NMS_IOU_THRESHOLD)  # 重复框抑制阈值，与部署入口一致。
MAX_VISUALIZED_IMAGES = 12  # 错误可视化最多展示的图片数。
VISUAL_MAX_WIDTH = 1200  # 可视化图片的最大渲染宽度，单位 px。
VISUAL_MAX_HEIGHT = 900  # 可视化图片的最大渲染高度，单位 px。
VISUAL_JPEG_QUALITY = 72  # 可视化图片的 JPEG 质量，兼顾清晰度和 HTML 体积。

BEST_F1_PATTERN = re.compile(r"^best_f1_detection_epoch_(\d+)_f1_([0-9]+\.[0-9]+)\.pth$")  # 最佳 F1 检查点。
BEST_LOSS_PATTERN = re.compile(r"^best_detection_epoch_(\d+)_loss_([0-9]+\.[0-9]+)\.pth$")  # 最佳 loss 检查点。
REGULAR_PATTERN = re.compile(r"^detection_epoch_(\d+)(?:_early_stop)?\.pth$")  # 常规或早停检查点。

CHECKPOINT_KIND_LABELS = {  # 检查点类型到中文说明的映射。
    "best_f1": "最佳 F1",
    "best_loss": "最佳 Loss",
    "regular": "常规",
    "early_stop": "早停",
    "other": "其他",
}


def read_event_scalars(directory):
    """读取目录顶层事件文件中的标量；目录不存在或没有事件时返回空字典。"""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    try:
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags()["scalars"]
        return {tag: [(int(event.step), float(event.value))
                      for event in accumulator.Scalars(tag)] for tag in tags}
    except Exception as exc:
        print(f"读取 TensorBoard 失败，跳过 {directory}: {exc}")
        return {}


def load_curves(log_dir):
    """读取报告需要的标量曲线；兼容新版根标签和旧版 add_scalars 子目录。"""
    scalars = read_event_scalars(log_dir)
    wanted = (
        "Loss/Train", "Loss/Validation", "ScoreLoss/Train", "ScoreLoss/Validation",
        "GeoLoss/Train", "GeoLoss/Validation", "Val/F1", "Val/Precision", "Val/Recall",
        "Val/best_F1", "Learning Rate", "Augmentation/light", "Augmentation/standard",
        "Augmentation/strong", "Performance/train_epoch_seconds",
        "Performance/validation_seconds", "Performance/train_images_per_second",
    )
    curves = {tag: scalars[tag] for tag in wanted if tag in scalars}

    # 历史 train.py 用 add_scalars 把训练/验证 loss 写到独立子目录。
    legacy = {
        "epoch_loss_train": "Loss/Train",
        "epoch_loss_validation": "Loss/Validation",
        "epoch_score Loss_train": "ScoreLoss/Train",
        "epoch_score Loss_validation": "ScoreLoss/Validation",
        "epoch_geo Loss_train": "GeoLoss/Train",
        "epoch_geo Loss_validation": "GeoLoss/Validation",
    }
    for subdir, name in legacy.items():
        if name in curves:
            continue
        sub_scalars = read_event_scalars(Path(log_dir) / subdir)
        if sub_scalars:
            curves[name] = next(iter(sub_scalars.values()))

    if not curves:
        print(f"未找到 TensorBoard 曲线: {log_dir}")
    return curves


def describe_checkpoint(path):
    """解析检查点文件名和文件信息，F1/loss 只取自文件名以保持可复核。"""
    name = path.name
    kind, epoch, f1, loss = "other", 0, None, None
    match = BEST_F1_PATTERN.match(name)
    if match:
        kind, epoch, f1 = "best_f1", int(match.group(1)), float(match.group(2))
    else:
        match = BEST_LOSS_PATTERN.match(name)
        if match:
            kind, epoch, loss = "best_loss", int(match.group(1)), float(match.group(2))
        else:
            match = REGULAR_PATTERN.match(name)
            if match:
                kind = "early_stop" if name.endswith("_early_stop.pth") else "regular"
                epoch = int(match.group(1))
    stat = path.stat()
    return {
        "name": name,
        "path": str(path),
        "kind": kind,
        "epoch": epoch,
        "f1": f1,
        "loss": loss,
        "size_mb": round(stat.st_size / (1024 * 1024), 1),
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "eval": None,
    }


def scan_checkpoints(run_dir):
    """扫描检查点目录，按轮数排序；文件名不符合预期时归入“其他”。"""
    checkpoints = [describe_checkpoint(path) for path in sorted(Path(run_dir).glob("*.pth"))]
    if not checkpoints:
        raise ValueError(f"检查点目录中没有 .pth 文件: {run_dir}")
    checkpoints.sort(key=lambda item: (item["epoch"], item["name"]))
    return checkpoints


def select_checkpoints(checkpoints):
    """选出最佳 F1、最佳 loss 和最后一轮检查点。"""
    best_f1_list = [item for item in checkpoints if item["kind"] == "best_f1"]
    best_loss_list = [item for item in checkpoints if item["kind"] == "best_loss"]
    regular_list = [item for item in checkpoints if item["kind"] in ("regular", "early_stop")]
    best_f1 = max(best_f1_list, key=lambda item: item["f1"]) if best_f1_list else None
    best_loss = min(best_loss_list, key=lambda item: item["loss"]) if best_loss_list else None
    last = max(regular_list or checkpoints, key=lambda item: item["epoch"])
    primary = best_f1 or best_loss or last
    return {"best_f1": best_f1, "best_loss": best_loss, "last": last, "primary": primary}


def read_run_config(run_dir):
    """读取训练写入的 run_config.json；不存在时返回空字典。"""
    path = Path(run_dir) / "run_config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取运行配置失败，按默认尺寸处理 {path}: {exc}")
        return {}


def read_source_constant(path, name):
    """从 Python 源码中读取简单常量，避免为了检查默认值导入重量级模块。"""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^\n#]+)", source, re.MULTILINE)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError):
        return None


def aligned_size(size):
    """把尺寸对齐到 32 的倍数，并限制最大边长。"""
    height, width = normalize_input_size(size)
    return (min(MAX_INPUT_SIDE, height), min(MAX_INPUT_SIDE, width))


def resolve_path(path):
    """把训练配置里的相对路径按仓库根目录解析。"""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def extra_input_size(size):
    """按固定倍率生成额外的对比尺寸；尺寸不变时返回 None。"""
    height, width = normalize_input_size(size)
    scaled = (int(math.ceil(height * INPUT_SCALE_FACTOR / 32) * 32),
              int(math.ceil(width * INPUT_SCALE_FACTOR / 32) * 32))
    scaled = aligned_size(scaled)
    return None if scaled == (height, width) else scaled


def evaluate_model_on_val(model, dataset, device, input_sizes, thresholds, collect_images=False):
    """在验证集上评估一个模型，返回各工作点指标和可选的逐图缓存。

    每个输入尺寸只做一次前向，多个阈值复用同一份 score/geo 图。
    """
    input_sizes = tuple(dict.fromkeys(normalize_input_size(size) for size in input_sizes))
    thresholds = tuple(dict.fromkeys(float(value) for value in thresholds))
    totals = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    images = {} if collect_images else None
    model.eval()

    with torch.no_grad():
        for index, (image_path, label_path) in enumerate(
                zip(dataset.image_paths, dataset.label_paths), 1):
            image = load_rgb_image(image_path)
            if label_path is None:
                quads = np.empty((0, 4, 2), dtype=np.float32)
                ignored = np.zeros(0, dtype=bool)
            else:
                quads, ignored = read_quads(label_path, image.size, dataset.label_format)
            boxes = quads_to_aabb(quads)
            truth_boxes = boxes[~ignored]
            heights = truth_boxes[:, 3] - truth_boxes[:, 1]
            entry = None
            if collect_images:
                entry = {
                    "path": str(image_path),
                    "width": image.size[0],
                    "height": image.size[1],
                    "gt": int(len(truth_boxes)),
                    "truth_boxes": truth_boxes,
                    "median_gt_height": float(np.median(heights)) if len(heights) else 0.0,
                    "metadata": {},
                    "combos": {},
                }

            for size in input_sizes:
                tensor, metadata = prepare_image(image, size)
                score, geometry = model(tensor[None].to(device))
                score = score[0, 0].cpu().numpy()
                geometry = np.expm1(np.clip(geometry[0].cpu().numpy(), 0, FEATURE_GEOMETRY_CLIP))
                if entry is not None:
                    entry["metadata"][size] = metadata

                for threshold in thresholds:
                    decoded, scores = decode_predictions(
                        score, geometry, score_threshold=threshold,
                        small_region_area=scaled_small_region_area(size))
                    if len(decoded):
                        decoded = restore_boxes(decoded, metadata, image.size)
                        valid = (decoded[:, 2] > decoded[:, 0]) & (decoded[:, 3] > decoded[:, 1])
                        decoded, scores = decoded[valid], scores[valid]
                        keep = suppress_boxes(decoded, scores, REPORT_NMS_IOU)
                        decoded, scores = decoded[keep], scores[keep]
                    predictions = [{"box": decoded[i].tolist(), "score": float(scores[i])}
                                   for i in range(len(decoded))]
                    status, used, counts = match_box_status(
                        predictions, boxes, ignored, REPORT_MATCH_IOU)
                    totals[(size, threshold)] += counts
                    if entry is not None:
                        entry["combos"][(size, threshold)] = {
                            "predictions": predictions,
                            "status": [int(value) for value in status],
                            "used": sorted(int(value) for value in used),
                            "counts": [int(value) for value in counts],
                        }

            if entry is not None:
                images[image_path.name] = entry
            if index % 500 == 0:
                print(f"  已评估 {index}/{len(dataset.image_paths)} 张")

    rows = {key: {"overall": summarize_counts(counts)} for key, counts in totals.items()}
    return rows, images


def build_workpoints(rows):
    """把评估结果整理成工作点列表，并标出验证集总体 F1 最高的组合。"""
    workpoints = []
    for (size, threshold), groups in sorted(rows.items(), key=lambda item: (item[0][0], item[0][1])):
        overall = groups.get("overall")
        if overall is None:
            continue
        workpoints.append({
            "input_size": tuple(int(value) for value in size),
            "threshold": float(threshold),
            "overall": overall,
        })
    best = max(workpoints, key=lambda item: item["overall"]["f1"]) if workpoints else None
    return workpoints, best


def render_error_image(entry, combo):
    """把漏检真值框画成橙色、误检预测框画成红色，返回 JPEG data URI。"""
    image = load_rgb_image(Path(entry["path"]))
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(max(image.size) / 500))
    used = set(combo["used"])

    # 先画漏检真值，再画误检预测，避免误检框被真值框覆盖。
    for index, box in enumerate(entry["truth_boxes"]):
        if index in used:
            continue
        draw.rectangle(tuple(float(value) for value in box),
                       outline="#f59e0b", width=line_width)
    for prediction, status in zip(combo["predictions"], combo["status"]):
        if status == 0:
            draw.rectangle(tuple(float(value) for value in prediction["box"]),
                           outline="#dc2626", width=line_width)

    width, height = image.size
    scale = min(1.0, VISUAL_MAX_WIDTH / width, VISUAL_MAX_HEIGHT / height)
    if scale < 1.0:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                             resample=Image.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=VISUAL_JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _orientation(width, height):
    """按原图宽高比把图片分为横向、纵向和近方形，便于定位系统性错误。"""
    if width >= height * 1.2:
        return "wide"
    if height >= width * 1.2:
        return "portrait"
    return "square"


def build_error_analysis(images, combo):
    """统计逐图错误，并生成错误集中度和最差图片列表。"""
    if not images:
        return {"summary": {}, "orientation": {}, "images": []}

    records = []
    for name, entry in images.items():
        result = entry["combos"][combo]
        tp, fp, fn = result["counts"]
        records.append({
            "name": name,
            "orientation": _orientation(entry["width"], entry["height"]),
            "width": entry["width"],
            "height": entry["height"],
            "gt": entry["gt"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "error_rate": (fp + fn) / max(1, entry["gt"]),
            "median_text_height": round(entry["median_gt_height"], 2)
            if entry["median_gt_height"] else 0.0,
            "entry": entry,
        })

    total_errors = sum(item["fp"] + item["fn"] for item in records)
    error_records = [item for item in records if item["fp"] + item["fn"] > 0]
    ordered = sorted(records, key=lambda item: (-(item["fp"] + item["fn"]), item["name"]))
    top_ten = ordered[:10]
    summary = {
        "images": len(records),
        "gt": sum(item["gt"] for item in records),
        "total_errors": total_errors,
        "error_images": len(error_records),
        "zero_error_images": len(records) - len(error_records),
        "top_ten_errors": sum(item["fp"] + item["fn"] for item in top_ten),
        "top_ten_share": (sum(item["fp"] + item["fn"] for item in top_ten) / total_errors)
        if total_errors else 0.0,
    }

    orientation = {}
    for key in ("wide", "portrait", "square"):
        subset = [item for item in records if item["orientation"] == key]
        if subset:
            orientation[key] = {
                "images": len(subset),
                "errors": sum(item["fp"] + item["fn"] for item in subset),
                "fp": sum(item["fp"] for item in subset),
                "fn": sum(item["fn"] for item in subset),
            }

    visuals = []
    for item in ordered[:MAX_VISUALIZED_IMAGES]:
        if item["fp"] + item["fn"] == 0:
            break
        visuals.append({
            "name": item["name"],
            "width": item["width"],
            "height": item["height"],
            "gt": item["gt"],
            "tp": item["tp"],
            "fp": item["fp"],
            "fn": item["fn"],
            "error_rate": item["error_rate"],
            "median_text_height": item["median_text_height"],
            "image": render_error_image(item["entry"], item["entry"]["combos"][combo]),
        })
    return {"summary": summary, "orientation": orientation, "images": visuals}


def count_dataset_split(image_dir, label_dir):
    """统计一个数据划分的图片数、标签框数和 ### 忽略框数。"""
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    result = {"images": 0, "label_boxes": 0, "ignored_boxes": 0}
    if image_dir.is_dir():
        result["images"] = sum(1 for path in image_dir.iterdir()
                               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if label_dir.is_dir():
        # 小图标签是坐标在前、### 在行尾；按最后一个字段判断忽略区域。
        for path in label_dir.glob("*.txt"):
            try:
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    fields = line.replace(",", " ").split()
                    if not fields:
                        continue
                    if fields[-1].strip('"') == "###":
                        result["ignored_boxes"] += 1
                    else:
                        result["label_boxes"] += 1
            except OSError as exc:
                print(f"跳过无法读取的标签 {path}: {exc}")
    return result


def build_dataset_summary(train_dirs=(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR),
                          val_dirs=(VAL_IMAGE_DIR, VAL_LABEL_DIR)):
    """返回训练集和验证集的基础统计。"""
    return {
        "train": count_dataset_split(*train_dirs),
        "val": count_dataset_split(*val_dirs),
    }


def summarize_run(curves, checkpoints, chosen):
    """汇总最佳 F1、最佳 loss、轮数和耗时等运行级信息。"""
    f1_points = curves.get("Val/F1", [])
    loss_points = curves.get("Loss/Validation", [])
    best_f1_epoch = best_f1_value = None
    if f1_points:
        best_f1_epoch, best_f1_value = max(f1_points, key=lambda item: item[1])
    elif chosen["best_f1"]:
        best_f1_epoch = chosen["best_f1"]["epoch"]
        best_f1_value = chosen["best_f1"]["f1"]

    best_loss_epoch = best_loss_value = None
    if loss_points:
        best_loss_epoch, best_loss_value = min(loss_points, key=lambda item: item[1])
    elif chosen["best_loss"]:
        best_loss_epoch = chosen["best_loss"]["epoch"]
        best_loss_value = chosen["best_loss"]["loss"]

    last_f1 = f1_points[-1][1] if f1_points else None
    epochs = max([point[0] for points in curves.values() for point in points]
                 + [item["epoch"] for item in checkpoints] + [0])
    train_seconds = sum(value for _, value in curves.get("Performance/train_epoch_seconds", []))
    val_seconds = sum(value for _, value in curves.get("Performance/validation_seconds", []))
    throughput = curves.get("Performance/train_images_per_second", [])
    return {
        "epochs": epochs,
        "best_f1_epoch": best_f1_epoch,
        "best_f1": best_f1_value,
        "best_loss_epoch": best_loss_epoch,
        "best_loss": best_loss_value,
        "last_f1": last_f1,
        "training_hours": round((train_seconds + val_seconds) / 3600, 2)
        if train_seconds or val_seconds else None,
        "throughput": round(sum(value for _, value in throughput) / len(throughput), 1)
        if throughput else None,
        "has_curves": bool(curves),
    }


def read_source_path(path, name):
    """读取源码中的 ``NAME = Path("...")`` 字符串，供导出检查点核对。"""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(rf"^{re.escape(name)}\s*=\s*Path\(\s*[\"']([^\"']+)[\"']\s*\)",
                      source, re.MULTILINE)
    return match.group(1) if match else None


def inspect_deployment(primary_checkpoint, primary_size, best_workpoint, run_info):
    """对比训练结果和当前推理/导出默认值，生成可执行的检查项。"""
    checks = []
    configured_model = read_source_constant(INFERENCE_SOURCE, "MODEL_PATH")
    if configured_model is None:
        checks.append({
            "status": "info",
            "title": "历史推理配置无法自动读取",
            "detail": "未能从 detection_small/inference.py 读取 MODEL_PATH。",
            "action": "新部署请使用 detection_small_inference_onnx.py 和 runtime。",
        })
    else:
        configured_path = Path(configured_model)
        if not configured_path.is_absolute():
            configured_path = REPO_ROOT / configured_path
        if configured_path.resolve() != Path(primary_checkpoint["path"]).resolve():
            checks.append({
                "status": "warn",
                "title": "历史 PyTorch 推理入口检查点未更新",
                "detail": f"detection_small/inference.py 当前指向 {configured_path.name}，"
                          f"本次主检查点是 {primary_checkpoint['name']}；该入口不是当前 ONNX 部署路径。",
                "action": "如需继续使用历史推理入口再更新 MODEL_PATH；ONNX 部署请检查下方导出项。",
            })
        else:
            checks.append({
                "status": "ok",
                "title": "历史 PyTorch 推理入口检查点",
                "detail": "detection_small/inference.py 已指向本次主检查点。",
                "action": "",
            })

    export_checkpoint = read_source_path(EXPORT_SOURCE, "CHECKPOINT_PATH")
    if export_checkpoint is None:
        checks.append({
            "status": "info",
            "title": "ONNX 导出检查点无法自动读取",
            "detail": "未能从 detection_small_export_onnx.py 读取 CHECKPOINT_PATH。",
            "action": "导出前手动核对 CHECKPOINT_PATH 和 INPUT_SIZE。",
        })
    else:
        configured_path = resolve_path(export_checkpoint)
        if configured_path.resolve() != Path(primary_checkpoint["path"]).resolve():
            checks.append({
                "status": "warn",
                "title": "ONNX 导出检查点未更新",
                "detail": f"detection_small_export_onnx.py 的 CHECKPOINT_PATH 指向 "
                          f"{configured_path.name}，本次主检查点是 {primary_checkpoint['name']}。",
                "action": "更新 CHECKPOINT_PATH 后重新导出并核验 ONNX 数值。",
            })
        else:
            checks.append({
                "status": "ok",
                "title": "ONNX 导出检查点",
                "detail": "detection_small_export_onnx.py 已指向本次主检查点。",
                "action": "",
            })

    configured_size = read_source_constant(INFERENCE_SOURCE, "INPUT_SIZE")
    if configured_size is not None:
        try:
            configured_size = normalize_input_size(configured_size)
        except ValueError:
            configured_size = None
    if configured_size is not None:
        if configured_size != primary_size:
            checks.append({
                "status": "warn",
                "title": "推理输入尺寸与训练尺寸不一致",
                "detail": f"detection_small/inference.py 使用 "
                          f"{configured_size[0]}×{configured_size[1]}，"
                          f"训练和主指标使用 {primary_size[0]}×{primary_size[1]}。",
                "action": "把推理入口 INPUT_SIZE 调整为训练尺寸，或确认新尺寸经过独立验证。",
            })
        else:
            checks.append({
                "status": "ok",
                "title": "推理输入尺寸",
                "detail": f"推理入口使用 {primary_size[0]}×{primary_size[1]}，与训练尺寸一致。",
                "action": "",
            })

    export_size = read_source_constant(EXPORT_SOURCE, "INPUT_SIZE")
    if export_size is not None:
        try:
            export_size = normalize_input_size(export_size)
        except ValueError:
            export_size = None
    if export_size is not None:
        export_dynamic = bool(read_source_constant(EXPORT_SOURCE, "DYNAMIC_SPATIAL"))
        export_mode = "动态高宽" if export_dynamic else "固定高宽"
        if export_size != primary_size:
            checks.append({
                "status": "warn",
                "title": "ONNX 导出尺寸未同步",
                "detail": f"detection_small_export_onnx.py 的 INPUT_SIZE 为 "
                          f"{export_size[0]}×{export_size[1]}（{export_mode}），主指标使用 "
                          f"{primary_size[0]}×{primary_size[1]}。",
                "action": "导出前把 INPUT_SIZE 调整为训练尺寸，或与部署输入保持同一尺寸。",
            })
        else:
            checks.append({
                "status": "ok",
                "title": "ONNX 导出尺寸",
                "detail": f"导出入口使用 {primary_size[0]}×{primary_size[1]}（{export_mode}），"
                          "与主指标一致。",
                "action": "",
            })

    train_images = read_source_constant(TRAIN_SOURCE, "TRAIN_IMAGE_DIR")
    val_images = read_source_constant(TRAIN_SOURCE, "VAL_IMAGE_DIR")
    if train_images and val_images:
        same = (REPO_ROOT / str(train_images)).resolve() == (REPO_ROOT / str(val_images)).resolve()
        checks.append({
            "status": "warn" if same else "ok",
            "title": "训练/验证目录隔离",
            "detail": ("train.py 的验证目录仍指向训练目录，指标是在训练数据上得到的。"
                       if same else
                       "train.py 使用独立验证图片目录，指标不包含训练图片。"),
            "action": "把 VAL_IMAGE_DIR/VAL_LABEL_DIR 指向独立验证集。" if same else "",
        })

    if best_workpoint and best_workpoint["input_size"] != primary_size:
        gain = best_workpoint["overall"]["f1"]
        if gain > 0 and run_info.get("best_f1"):
            difference = gain - run_info["best_f1"]
            if difference >= 0.002:
                checks.append({
                    "status": "info",
                    "title": "更大输入在验证集上更好",
                    "detail": f"{best_workpoint['input_size'][0]}×{best_workpoint['input_size'][1]} / "
                              f"阈值 {best_workpoint['threshold']:.2f} 的总体 F1 为 {gain:.5f}，"
                              f"比训练尺寸高 {difference * 100:.2f} 个百分点。",
                    "action": "先用未参与调参的独立留出集复核，再决定是否修改部署尺寸。",
                })

    checks.append({
        "status": "info",
        "title": "验证集复用提醒",
        "detail": "检查点选择、输入尺寸和阈值对比都使用了同一个验证集，最佳工作点的数字存在乐观偏差。",
        "action": "上线前用未参与训练和调参的独立标注集复核一次。",
    })
    return checks


def line_chart_svg(series, title, y_label, log=False):
    """把若干条 (名称, 点列表, 颜色) 曲线渲染成内嵌 SVG。"""
    series = [(name, points, color) for name, points, color in series if points]
    if not series:
        return '<p class="muted">没有可绘制的数据。</p>'

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(9, 2.9), dpi=100)
    for name, points, color in series:
        axis.plot([point[0] for point in points], [point[1] for point in points],
                  label=name, color=color, linewidth=1.4)
    axis.set_title(title, loc="left")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if log:
        axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=8, ncol=min(3, len(series)))
    figure.tight_layout(pad=0.4)

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    svg = buffer.getvalue()
    svg = svg[svg.find("<svg"):]
    return svg.replace("<svg", '<svg class="chart"', 1)


def build_charts(curves):
    """构建报告里的四张图：F1、loss、学习率和增强阶段。"""
    return {
        "f1": line_chart_svg([
            ("overall", curves.get("Val/F1", []), "#2563eb"),
        ], "Validation F1", "F1"),
        "loss": line_chart_svg([
            ("train", curves.get("Loss/Train", []), "#64748b"),
            ("validation", curves.get("Loss/Validation", []), "#dc2626"),
        ], "Total loss (log scale)", "Loss", log=True),
        "lr": line_chart_svg([
            ("learning rate", curves.get("Learning Rate", []), "#7c3aed"),
        ], "Learning rate", "LR"),
        "aug": line_chart_svg([
            ("light", curves.get("Augmentation/light", []), "#0ea5e9"),
            ("standard", curves.get("Augmentation/standard", []), "#f59e0b"),
            ("strong", curves.get("Augmentation/strong", []), "#dc2626"),
        ], "Augmentation mix", "Weight"),
    }


def format_number(value, digits=4):
    """格式化数字；None 显示为短横线。"""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_percent(value, digits=2):
    """把 0-1 的比例格式化为百分比。"""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def esc(value):
    """转义动态文本，避免文件名或运行名破坏 HTML。"""
    return html_lib.escape(str(value), quote=True)


def format_size(size):
    """把 (高, 宽) 格式化为 高×宽。"""
    if not size:
        return "—"
    return f"{int(size[0])}×{int(size[1])}"


def primary_checkpoint_name(primary):
    """兼容报告数据里的 name/checkpoint 两种字段，返回主检查点文件名。"""
    return primary.get("name") or primary.get("checkpoint") or "—"


def render_group_table(groups):
    """渲染总体 TP/FP/FN/P/R/F1 表。"""
    metrics = groups.get("overall")
    if metrics is None:
        return '<p class="muted">没有指标数据。</p>'
    return (
        "<table><thead><tr><th>分组</th><th>TP</th><th>FP</th><th>FN</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>"
        f"<tr><td>全部</td><td>{metrics['tp']}</td><td>{metrics['fp']}</td>"
        f"<td>{metrics['fn']}</td><td>{format_number(metrics['precision'])}</td>"
        f"<td>{format_number(metrics['recall'])}</td>"
        f"<td><strong>{format_number(metrics['f1'])}</strong></td></tr>"
        "</tbody></table>")


def render_workpoint_table(workpoints, best):
    """渲染输入尺寸 × 阈值的工作点对比表。"""
    if not workpoints:
        return '<p class="muted">没有工作点数据。</p>'
    best_key = (best["input_size"], best["threshold"]) if best else None
    rows = []
    for item in workpoints:
        metrics = item["overall"]
        css = ' class="highlight"' if (item["input_size"], item["threshold"]) == best_key else ""
        rows.append(
            f"<tr{css}>"
            f"<td>{format_size(item['input_size'])}</td><td>{item['threshold']:.2f}</td>"
            f"<td>{format_number(metrics['precision'])}</td>"
            f"<td>{format_number(metrics['recall'])}</td>"
            f"<td><strong>{format_number(metrics['f1'])}</strong></td>"
            f"<td>{metrics['fp']}</td><td>{metrics['fn']}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>输入 高×宽</th><th>阈值</th><th>P</th><th>R</th><th>F1</th>"
        "<th>FP</th><th>FN</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_checkpoint_table(checkpoints, primary):
    """渲染检查点列表，并显示各检查点在主配置下的 F1。"""
    rows = []
    primary_name = primary_checkpoint_name(primary)
    for item in checkpoints:
        metrics = item.get("eval") or {}
        overall = metrics.get("overall") or {}
        css = ' class="highlight"' if item["name"] == primary_name else ""
        note = "本次报告主检查点" if item["name"] == primary_name else ""
        rows.append(
            f"<tr{css}>"
            f"<td class='mono'>{esc(item['name'])}</td>"
            f"<td>{item['epoch']}</td>"
            f"<td>{CHECKPOINT_KIND_LABELS.get(item['kind'], item['kind'])}</td>"
            f"<td>{format_number(overall.get('f1'))}</td>"
            f"<td>{item['size_mb']:.1f} MB</td>"
            f"<td>{item['mtime']}</td><td>{note}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>文件</th><th>轮数</th><th>类型</th><th>主配置 F1</th>"
        "<th>大小</th><th>更新时间</th><th>说明</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_error_summary(error_analysis):
    """渲染错误集中度和按原图方向的统计。"""
    summary = error_analysis.get("summary") or {}
    orientation = error_analysis.get("orientation") or {}
    if not summary:
        return '<p class="muted">没有逐图错误数据。</p>'

    parts = [
        '<div class="stat-row">',
        f'<div><span class="stat">{summary["total_errors"]}</span>'
        '<span class="muted">个错误（FP+FN）</span></div>',
        f'<div><span class="stat">{summary["zero_error_images"]}/{summary["images"]}</span>'
        '<span class="muted">张图零错误</span></div>',
        f'<div><span class="stat">{format_percent(summary["top_ten_share"], 1)}</span>'
        '<span class="muted">错误集中在前 10 张图</span></div>',
        '</div>',
    ]

    orientation_rows = []
    for key, label in (("wide", "横向"), ("portrait", "纵向"), ("square", "近方形")):
        item = orientation.get(key)
        if item is None:
            continue
        orientation_rows.append(
            f"<tr><td>{label}</td><td>{item['images']}</td><td>{item['errors']}</td>"
            f"<td>{item['fp']}</td><td>{item['fn']}</td></tr>")
    if orientation_rows:
        parts.append(
            "<h3>按原图方向拆分</h3>"
            "<table><thead><tr><th>方向</th><th>图片数</th><th>错误数</th><th>FP</th><th>FN</th>"
            "</tr></thead><tbody>" + "".join(orientation_rows) + "</tbody></table>")
    return "".join(parts)


def render_error_cards(error_images):
    """渲染最差图片卡片；橙色为漏检真值，红色为误检预测。"""
    if not error_images:
        return '<p class="muted">没有可展示的错误图片。</p>'
    cards = []
    for item in error_images:
        cards.append(
            '<figure class="error-card">'
            f'<img src="{item["image"]}" loading="lazy" alt="{esc(item["name"])}">'
            "<figcaption>"
            f"<div class='mono'>{esc(item['name'])}</div>"
            f"<div>GT {item['gt']} · TP {item['tp']} · "
            f"<span class='bad'>FP {item['fp']}</span> · "
            f"<span class='warn-text'>FN {item['fn']}</span> · "
            f"错误率 {format_percent(item['error_rate'], 1)}</div>"
            f"<div class='muted'>原图 {item['width']}×{item['height']}，"
            f"文字框中位高 {item['median_text_height']:.1f}px</div>"
            "</figcaption></figure>")
    return (
        '<div class="legend"><span class="legend-item"><i class="swatch warn"></i>橙色：漏检真值框</span>'
        '<span class="legend-item"><i class="swatch bad"></i>红色：误检预测框</span></div>'
        '<div class="error-grid">' + "".join(cards) + "</div>")


def render_dataset_table(dataset):
    """渲染训练集/验证集构成。"""
    rows = []
    for split, label in (("train", "训练集"), ("val", "验证集")):
        item = dataset.get(split) or {}
        rows.append(
            f"<tr><td>{label}</td><td>{item.get('images', 0)}</td>"
            f"<td>{item.get('label_boxes', 0)}</td>"
            f"<td>{item.get('ignored_boxes', 0)}</td></tr>")
    return (
        "<table><thead><tr><th>划分</th><th>图片数</th><th>有效标签框</th>"
        "<th>### 忽略框</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_deployment_checks(checks):
    """渲染部署检查清单。"""
    if not checks:
        return '<p class="muted">没有检查项。</p>'
    labels = {"warn": "需处理", "ok": "通过", "info": "提示"}
    items = []
    for check in checks:
        status = check.get("status", "info")
        action = check.get("action", "")
        items.append(
            f'<li class="check {status}">'
            f'<span class="badge {status}">{labels.get(status, status)}</span>'
            f"<div><strong>{esc(check.get('title', ''))}</strong>"
            f"<p>{esc(check.get('detail', ''))}</p>"
            + (f"<p class='action'>建议：{esc(action)}</p>" if action else "")
            + "</div></li>")
    return '<ul class="checklist">' + "".join(items) + "</ul>"


def build_highlights(report):
    """根据关键数据生成报告开头的行动摘要。"""
    highlights = []
    run = report["run"]
    primary = report["primary"]
    best = report.get("best_workpoint")
    error_summary = (report.get("error_analysis") or {}).get("summary") or {}
    checks = report.get("deployment") or []

    if run.get("best_f1") is not None:
        highlights.append(
            f"最佳验证 F1 为 <strong>{run['best_f1']:.5f}</strong>"
            f"（Epoch {run.get('best_f1_epoch', '—')}），主配置为 "
            f"{format_size(primary['input_size'])} / 阈值 {primary['threshold']:.2f}。")
    if best and tuple(best["input_size"]) != tuple(primary["input_size"]):
        difference = best["overall"]["f1"] - (run.get("best_f1") or best["overall"]["f1"])
        highlights.append(
            f"验证集最高 F1 出现在 {format_size(best['input_size'])} / 阈值 "
            f"{best['threshold']:.2f}：<strong>{best['overall']['f1']:.5f}</strong>"
            + (f"，比训练尺寸高 {difference * 100:.2f} 个百分点。" if difference > 0 else "。"))
    if error_summary:
        highlights.append(
            f"共 {error_summary['total_errors']} 个框级错误，"
            f"{error_summary['zero_error_images']}/{error_summary['images']} 张图零错误；"
            f"前 10 张图占 {format_percent(error_summary['top_ten_share'], 1)} 的错误。")
    warnings = [check for check in checks if check.get("status") == "warn"]
    if warnings:
        highlights.append(
            f"部署检查有 <strong>{len(warnings)}</strong> 项需要处理，优先看下方“部署检查清单”。")
    return highlights


CSS = """
:root { --bg:#f5f7fb; --panel:#fff; --text:#172033; --muted:#64748b; --line:#e2e8f0;
  --accent:#2563eb; --warn:#dc2626; --ok:#16a34a; --amber:#f59e0b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
  line-height:1.65; }
a { color:var(--accent); }
header { background:linear-gradient(135deg,#0f172a,#1e3a8a); color:#fff; padding:30px 32px 24px; }
header h1 { margin:0 0 8px; font-size:26px; letter-spacing:.2px; }
header .meta { color:#cbd5e1; font-size:13px; }
nav { position:sticky; top:0; z-index:2; background:rgba(255,255,255,.95);
  border-bottom:1px solid var(--line); padding:8px 24px; backdrop-filter:blur(6px); }
nav a { margin-right:16px; font-size:13px; text-decoration:none; color:#334155; }
main { max-width:1180px; margin:0 auto; padding:22px 20px 60px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; margin:16px 0; box-shadow:0 1px 2px rgba(15,23,42,.04); }
section > h2 { margin:0 0 12px; font-size:19px; }
h3 { font-size:15px; margin:16px 0 8px; }
.kpi { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }
.card { border:1px solid var(--line); border-radius:10px; padding:12px 14px; background:#fbfdff; }
.card.accent { border-color:#bfdbfe; background:#eff6ff; }
.card.warn { border-color:#fde68a; background:#fffbeb; }
.card .label { color:var(--muted); font-size:12px; }
.card .value { font-size:25px; font-weight:700; margin:2px 0; font-variant-numeric:tabular-nums; }
.card .note { color:var(--muted); font-size:12px; }
table { width:100%; border-collapse:collapse; margin:8px 0 4px; font-size:13px; }
th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:#475569; background:#f8fafc; font-weight:600; }
tr.highlight td { background:#eff6ff; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px;
  word-break:break-all; }
.muted { color:var(--muted); }
.chart-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
.chart-box { border:1px solid var(--line); border-radius:10px; padding:8px 10px; }
.chart-box h3 { margin:2px 0 4px; font-size:13px; color:#475569; }
svg.chart { width:100%; height:auto; display:block; }
.callout { border-left:4px solid var(--accent); background:#eff6ff; padding:10px 14px;
  border-radius:6px; margin:10px 0; }
.callout.warn { border-color:var(--warn); background:#fef2f2; }
ul.highlights { margin:6px 0 0; padding-left:20px; }
ul.highlights li { margin:4px 0; }
.stat-row { display:flex; gap:28px; flex-wrap:wrap; margin:4px 0 12px; }
.stat { font-size:24px; font-weight:700; margin-right:6px; font-variant-numeric:tabular-nums; }
.error-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
.error-card { margin:0; border:1px solid var(--line); border-radius:10px; overflow:hidden;
  background:#fbfdff; }
.error-card img { width:100%; display:block; background:#fff; }
.error-card figcaption { padding:8px 10px; font-size:12px; }
.error-card .bad { color:var(--warn); font-weight:600; }
.error-card .warn-text { color:#b45309; font-weight:600; }
.legend { display:flex; gap:18px; font-size:12px; color:#475569; margin:4px 0 12px; }
.legend-item { display:inline-flex; align-items:center; gap:6px; }
.swatch { width:12px; height:4px; display:inline-block; border-radius:2px; }
.swatch.warn { background:var(--amber); }
.swatch.bad { background:var(--warn); }
ul.checklist { list-style:none; padding:0; margin:0; }
li.check { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--line); }
li.check:last-child { border-bottom:0; }
li.check p { margin:2px 0; font-size:13px; }
li.check .action { color:#334155; }
.badge { flex:0 0 auto; height:22px; padding:0 8px; border-radius:999px; font-size:12px;
  line-height:22px; font-weight:600; }
.badge.warn { background:#fee2e2; color:#991b1b; }
.badge.ok { background:#dcfce7; color:#166534; }
.badge.info { background:#dbeafe; color:#1e40af; }
footer { color:var(--muted); font-size:12px; text-align:center; padding:20px; }
details { margin-top:8px; }
summary { cursor:pointer; color:var(--accent); }
@media (max-width:820px) { .kpi,.chart-grid,.error-grid { grid-template-columns:1fr; }
  header { padding:22px 18px; } main { padding:14px 10px 40px; } }
@media print { nav { display:none; } section { break-inside:avoid; box-shadow:none; } }
"""


def render_report(report):
    """把收集到的数据渲染为单文件 HTML。"""
    charts = build_charts(report.get("curves") or {})
    run = report["run"]
    primary = report["primary"]
    groups = report.get("groups") or {}
    workpoints = report.get("workpoints") or []
    best_workpoint = report.get("best_workpoint")
    error_analysis = report.get("error_analysis") or {}
    error_summary = error_analysis.get("summary") or {}
    highlights = build_highlights(report)

    primary_f1 = (groups.get("overall") or {}).get("f1")
    primary_note = (f"P {format_number(groups['overall']['precision'])} / "
                    f"R {format_number(groups['overall']['recall'])}") \
        if groups.get("overall") else ""
    best_note = (f"{format_size(best_workpoint['input_size'])} / "
                 f"阈值 {best_workpoint['threshold']:.2f}" if best_workpoint else "无")
    error_note = (f"零错误图片 {error_summary.get('zero_error_images', 0)}/"
                  f"{error_summary.get('images', 0)}" if error_summary else "无逐图数据")

    cards = [
        ("最佳验证 F1", format_number(run.get("best_f1"), 5),
         f"Epoch {run.get('best_f1_epoch', '—')} · Val/F1 峰值", "accent"),
        ("主配置 F1", format_number(primary_f1, 5),
         f"{format_size(primary['input_size'])} · 阈值 {primary['threshold']:.2f} · {primary_note}", ""),
        ("验证集最佳工作点", format_number((best_workpoint or {}).get("overall", {}).get("f1"), 5),
         best_note, "warn"),
        ("框级错误", str(error_summary.get("total_errors", "—")), error_note, ""),
    ]
    card_html = "".join(
        f'<div class="card {css}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{value}</div><div class="note">{esc(note)}</div></div>'
        for label, value, note, css in cards)

    highlight_html = "".join(f"<li>{item}</li>" for item in highlights)
    chart_html = "".join(
        f'<div class="chart-box"><h3>{title}</h3>{charts[key]}</div>'
        for key, title in (("f1", "验证集 F1"), ("loss", "训练/验证 loss"),
                           ("lr", "学习率"), ("aug", "增强阶段权重")))

    meta = (
        f"运行 <span class='mono'>{esc(run['name'])}</span> · "
        f"检查点 <span class='mono'>{esc(primary_checkpoint_name(primary))}</span> · "
        f"训练 {run.get('epochs', '—')} 轮"
        + (f" · {run['training_hours']} 小时" if run.get("training_hours") else "")
        + (f" · {run['throughput']} 张/秒" if run.get("throughput") else "")
        + f" · 生成于 {esc(report['generated_at'])}"
    )

    body = f"""
<header>
  <h1>小图检测训练分析报告</h1>
  <div class="meta">{meta}</div>
</header>
<nav>
  <a href="#overview">概览</a><a href="#curves">训练曲线</a><a href="#metrics">主配置指标</a>
  <a href="#workpoints">工作点对比</a><a href="#errors">错误分析</a>
  <a href="#checkpoints">检查点</a><a href="#data">数据构成</a>
  <a href="#deploy">部署检查</a><a href="#definitions">指标口径</a>
</nav>
<main>
<section id="overview">
  <h2>概览</h2>
  <div class="kpi">{card_html}</div>
  <ul class="highlights">{highlight_html}</ul>
</section>

<section id="curves">
  <h2>训练曲线</h2>
  <div class="chart-grid">{chart_html}</div>
  <p class="muted">曲线来自 detection_small/runs/{esc(run['name'])} 的 TensorBoard 事件；loss 使用对数纵轴。</p>
</section>

<section id="metrics">
  <h2>主配置指标</h2>
  <p class="muted">主配置：输入 {format_size(primary['input_size'])}、置信度阈值 {primary['threshold']:.2f}、
  NMS IoU {primary['nms_iou']:.2f}、匹配 IoU {primary['match_iou']:.2f}。所有指标只评估检测框，不涉及文字识别。</p>
  {render_group_table(groups)}
</section>

<section id="workpoints">
  <h2>工作点对比</h2>
  <p class="muted">同一检查点在验证集上比较输入尺寸和阈值。高亮行只是当前验证集最优，
  不代表独立测试集结论；上线前请用未参与调参的留出集复核。</p>
  {render_workpoint_table(workpoints, best_workpoint)}
</section>

<section id="errors">
  <h2>错误分析</h2>
  <p class="muted">逐图统计使用验证集最佳工作点：{esc(best_note)}；
  训练期主配置为 {format_size(primary['input_size'])} / 阈值 {primary['threshold']:.2f}。</p>
  {render_error_summary(error_analysis)}
  <h3>最差图片</h3>
  {render_error_cards(error_analysis.get('images') or [])}
</section>

<section id="checkpoints">
  <h2>检查点对比</h2>
  <p class="muted">F1 列统一在主配置（{format_size(primary['input_size'])} / 阈值 {primary['threshold']:.2f}）下复算；
  未评估的检查点显示为“—”。部署选型优先比较 F1，而不是只看 loss。</p>
  {render_checkpoint_table(report.get('checkpoints') or [], primary)}
</section>

<section id="data">
  <h2>数据构成</h2>
  {render_dataset_table(report.get('dataset') or {})}
  <p class="muted">有效标签框和 ### 忽略框按标签行统计，用于快速核对 GT 规模。</p>
</section>

<section id="deploy">
  <h2>部署检查清单</h2>
  {render_deployment_checks(report.get('deployment') or [])}
</section>

<section id="definitions">
  <h2>指标口径</h2>
  <details open>
    <summary>GT / TP / FP / FN 的定义</summary>
    <p><strong>GT 框</strong>：人工标注的有效真值框，### 忽略区域不计入，满足 GT = TP + FN。</p>
    <p><strong>TP</strong>：预测框与某个真值框 IoU ≥ {primary['match_iou']:.2f} 且一对一匹配成功。
    <strong>FP</strong>：预测框没有匹配到任何真值框，且不是主要落在忽略区内。
    <strong>FN</strong>：真值框没有被任何预测框匹配。</p>
    <p><strong>Precision = TP / (TP + FP)</strong>，衡量报出的框有多少是对的；
    <strong>Recall = TP / (TP + FN)</strong>，衡量真值框有多少被找到；
    <strong>F1</strong> 是两者的调和平均。IoU 0.3–0.5 的近匹配会同时产生 FP 和 FN，
    因此定位偏差也会明显影响指标。</p>
  </details>
  <p class="muted">解码使用四连通域，小区域面积按输入像素面积缩放；总体指标按框数汇总，不是按图片数平均。</p>
</section>
</main>
<footer>
  报告由 detection_small/report_training.py 生成。验证集同时用于检查点选择和工作点对比，数字存在乐观偏差；
  训练、验证和评估共享的预处理以仓库当前代码为准。
</footer>
"""

    return ("<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>小图检测训练分析报告 · {esc(run['name'])}</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>{body}</body>\n</html>\n")


def json_safe(value):
    """把 numpy/Path 等对象递归转换为可 JSON 序列化的 Python 值。"""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def report_summary(report):
    """返回适合写入 summary.json 的轻量摘要，去掉曲线和图片 data URI。"""
    summary = {key: value for key, value in report.items()
               if key not in ("curves", "error_analysis")}
    error_analysis = report.get("error_analysis") or {}
    summary["error_summary"] = {
        key: value for key, value in error_analysis.items() if key != "images"
    }
    return summary


def collect_report(run_dir, device):
    """读取运行产物、评估验证集并返回渲染报告所需的数据。"""
    run_dir = Path(run_dir)
    run_name = run_dir.name
    log_dir = RUNS_DIR / run_name
    print(f"扫描检查点: {run_dir}")
    checkpoints = scan_checkpoints(run_dir)
    chosen = select_checkpoints(checkpoints)
    primary = chosen["primary"]

    print(f"读取 TensorBoard: {log_dir}")
    curves = load_curves(log_dir)
    run_info = summarize_run(curves, checkpoints, chosen)
    run_info["name"] = run_name
    run_info["log_dir"] = str(log_dir)
    run_info["checkpoint_dir"] = str(run_dir)

    config = read_run_config(run_dir)
    configured_size = config.get("input_size") or read_source_constant(TRAIN_SOURCE, "INPUT_SIZE")
    try:
        primary_size = normalize_input_size(configured_size or (256, 768))
    except ValueError:
        primary_size = (256, 768)
    train_dirs = (resolve_path(config.get("train_image_dir") or TRAIN_IMAGE_DIR),
                  resolve_path(config.get("train_label_dir") or TRAIN_LABEL_DIR))
    val_dirs = (resolve_path(config.get("val_image_dir") or VAL_IMAGE_DIR),
                resolve_path(config.get("val_label_dir") or VAL_LABEL_DIR))

    print("统计训练/验证数据构成")
    dataset_summary = build_dataset_summary(train_dirs, val_dirs)
    dataset = OCRDataset(str(val_dirs[0]), str(val_dirs[1]))

    print(f"评估主检查点: {primary['name']}")
    model = load_model(primary["path"], device)
    input_sizes = [primary_size]
    larger = extra_input_size(primary_size)
    if larger is not None:
        input_sizes.append(larger)
    rows, images = evaluate_model_on_val(
        model, dataset, device, input_sizes, REPORT_THRESHOLDS, collect_images=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    primary_key = (primary_size, float(DEFAULT_SCORE_THRESHOLD))
    if primary_key not in rows:
        raise ValueError(f"主配置 {primary_key} 没有评估结果")
    primary["eval"] = rows[primary_key]
    evaluated = {primary["name"]}

    # 只对比最佳 loss 和最后一轮，避免报告为目录里所有历史检查点重复推理。
    for candidate in (chosen["best_loss"], chosen["last"]):
        if candidate is None or candidate["name"] in evaluated:
            continue
        print(f"评估对照检查点: {candidate['name']}")
        candidate_model = load_model(candidate["path"], device)
        candidate_rows, _ = evaluate_model_on_val(
            candidate_model, dataset, device, (primary_size,),
            (float(DEFAULT_SCORE_THRESHOLD),), collect_images=False)
        candidate["eval"] = candidate_rows.get((primary_size, float(DEFAULT_SCORE_THRESHOLD)))
        evaluated.add(candidate["name"])
        del candidate_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    workpoints, best_workpoint = build_workpoints(rows)
    best_combo = ((best_workpoint["input_size"], best_workpoint["threshold"])
                  if best_workpoint else primary_key)
    print(f"统计逐图错误，工作点: {format_size(best_combo[0])} / 阈值 {best_combo[1]:.2f}")
    error_analysis = build_error_analysis(images, best_combo)
    deployment = inspect_deployment(primary, primary_size, best_workpoint, run_info)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run": run_info,
        "curves": curves,
        "dataset": dataset_summary,
        "primary": {
            "name": primary["name"],
            "checkpoint": primary["name"],
            "input_size": list(primary_size),
            "threshold": float(DEFAULT_SCORE_THRESHOLD),
            "nms_iou": float(REPORT_NMS_IOU),
            "match_iou": float(REPORT_MATCH_IOU),
        },
        "groups": rows[primary_key],
        "workpoints": workpoints,
        "best_workpoint": best_workpoint,
        "checkpoints": checkpoints,
        "error_analysis": error_analysis,
        "deployment": deployment,
    }
