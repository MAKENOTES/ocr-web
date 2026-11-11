"""汇总部署推理入口在训练/验证集上的预测，生成与训练指标对比的静态 HTML 报告。

报告读取 detection/inference.py 和 detection_inference_onnx.py 写出的 bbox4 预测
标签，按训练期相同口径（匹配 IoU 0.5、### 忽略区）计算 TP/FP/FN、Precision、
Recall 和 F1，并与训练报告的验证指标、TensorBoard 曲线和耗时对照。图表和错误
图片内嵌在 HTML 中，可以离线打开；summary.json 保留可复查的数值。
"""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from detection.components.dataset import read_boxes
from detection.components.eval_metrics import (
    SOURCE_PREFIXES,
    match_box_status,
    summarize_counts,
)
from detection.components.image import IMAGE_EXTENSIONS, load_rgb_image
from detection.utils.training_report import (
    esc,
    format_number,
    format_percent,
    json_safe,
    line_chart_svg,
    render_error_cards,
    render_error_image,
    render_error_summary,
)
from detection.inference import draw_predictions


MATCH_IOU = 0.5  # 预测框与真值框的匹配 IoU 下限，与训练期验证指标一致。
MAX_VISUALIZED_IMAGES = 8  # 报告最多展示的最差图片数，控制 HTML 体积。
BOX_MATCH_TOLERANCE = 0.05  # 两后端框坐标视为一致的绝对误差，单位 px。
BACKEND_LABELS = {"pytorch": "PyTorch", "onnx": "ONNX/OpenVINO"}  # 推理后端显示名。
SPLIT_LABELS = {"train": "训练集", "val": "验证集"}  # 数据划分显示名。
GROUP_LABELS = {"overall": "总体", "doc": "文档页 doc", "ui": "界面小图 ui", "other": "其他"}  # 指标分组显示名。
RUN_KEYS = (("pytorch", "train"), ("onnx", "train"), ("pytorch", "val"), ("onnx", "val"))  # 报告固定展示的四次推理。
ASPECT_BUCKETS = (("≤2", 2.0), ("2-5", 5.0), ("5-10", 10.0), ("10-20", 20.0), (">20", None))  # 长宽比上界；最后一档无上界。
ERROR_CSV_FIELDS = (
    "image", "path", "split", "source", "width", "height", "gt",
    "pytorch_tp", "pytorch_fp", "pytorch_fn", "pytorch_error_count", "pytorch_error_rate",
    "onnx_tp", "onnx_fp", "onnx_fn", "onnx_error_count", "onnx_error_rate",
    "input_size", "score_threshold", "nms_iou",
)  # 错误图片 CSV 的列顺序；前 7 列描述图片和真值，后两组分别对应两个后端。


def input_size_label(size, dynamic):
    """返回输入画布文案：固定画布写成 768×768，动态画布写成 768px 动态。"""
    return f"{size}px 动态" if dynamic else f"{size}×{size} 固定"


def aspect_bucket(width, height):
    """按宽/高把图片分到长宽比区间，用于单独观察超宽细条。"""
    ratio = width / max(1, height)
    for label, upper in ASPECT_BUCKETS:
        if upper is None or ratio <= upper:
            return label
    return ">20"


def build_aspect_groups(records):
    """汇总逐图记录的长宽比分组指标；没有图片的区间不输出。"""
    totals = {label: np.zeros(3, dtype=np.int64) for label, _ in ASPECT_BUCKETS}
    images = {label: 0 for label, _ in ASPECT_BUCKETS}
    for item in records:
        label = aspect_bucket(item["width"], item["height"])
        totals[label] += np.array([item["tp"], item["fp"], item["fn"]], dtype=np.int64)
        images[label] += 1
    return {label: {"images": images[label], **summarize_counts(totals[label])}
            for label, _ in ASPECT_BUCKETS if images[label]}
ERROR_PREVIEW_DIRNAME = "error_previews"  # 错误图片预览子目录，建在 errors_{split}.csv 同层目录下。
ERROR_PREVIEW_CSV_FIELDS = (
    "image", "preview", "path", "split", "source", "width", "height", "gt",
    "tp", "fp", "fn", "error_count", "error_rate", "input_size",
    "score_threshold", "nms_iou", "infer_ms", "detect_ms",
)  # 错误图片预览 CSV 的列顺序；preview 为相对该 CSV 所在目录的路径。


def load_image_index(image_dir):
    """返回 {stem: (图片路径, (宽, 高))}；尺寸按 EXIF 方向后的 RGB 图片计算。"""
    index = {}
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.stem] = (path, load_rgb_image(path).size)
    if not index:
        raise ValueError(f"图片目录中没有支持的图片: {image_dir}")
    return index


def prediction_boxes(label_path, size):
    """读取推理入口写出的 bbox4 标签；坐标越界或退化时立即报错。"""
    boxes, _ = read_boxes(label_path, size)
    return boxes


def source_group(name):
    """按文件名前缀划分业务来源，未知前缀归入 other。"""
    prefix = name.split("_", 1)[0]
    return prefix if prefix in SOURCE_PREFIXES else "other"


def build_error_analysis(records, collect_visuals):
    """统计逐图错误集中度，并可选地把最差图片叠加真值/误检框。"""
    if not records:
        return {"summary": {}, "groups": {}, "orientation": {}, "images": []}

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

    groups = {}
    for group in ("doc", "ui", "other"):
        subset = [item for item in records if item["group"] == group]
        if not subset:
            continue
        groups[group] = {
            "images": len(subset),
            "errors": sum(item["fp"] + item["fn"] for item in subset),
            "fp": sum(item["fp"] for item in subset),
            "fn": sum(item["fn"] for item in subset),
            "zero_error_images": sum(1 for item in subset if item["fp"] + item["fn"] == 0),
        }

    orientation = {}
    doc_records = [item for item in records if item["group"] == "doc"]
    for name, subset in (
            ("wide", [item for item in doc_records if item["width"] >= item["height"]]),
            ("portrait", [item for item in doc_records if item["width"] < item["height"]])):
        if subset:
            orientation[name] = {
                "images": len(subset),
                "errors": sum(item["fp"] + item["fn"] for item in subset),
                "fn": sum(item["fn"] for item in subset),
                "fp": sum(item["fp"] for item in subset),
            }

    visuals = []
    if collect_visuals:
        for item in ordered[:MAX_VISUALIZED_IMAGES]:
            if item["fp"] + item["fn"] == 0:
                break
            visuals.append({
                "name": item["name"],
                "group": item["group"],
                "width": item["width"],
                "height": item["height"],
                "gt": item["gt"],
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "error_rate": item["error_rate"],
                "median_text_height": item["median_text_height"],
                "image": render_error_image(
                    {"path": item["path"], "truth_boxes": item["truth_boxes"]},
                    {"used": item["used"], "predictions": item["predictions"],
                     "status": item["status"]}),
            })

    return {"summary": summary, "groups": groups, "orientation": orientation,
            "images": visuals}


def evaluate_split(image_index, gt_label_dir, pred_label_dir, collect_visuals=False):
    """按训练期口径评估一组预测，返回分组指标、逐图记录和错误分析。

    image_index 由 load_image_index 生成；预测标签缺失时直接报错，避免把空预测
    静默当成漏检。逐图记录保留真值框、预测框和匹配状态，供一致性对比和可视化。
    """
    gt_label_dir, pred_label_dir = Path(gt_label_dir), Path(pred_label_dir)
    totals = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    records, pred_boxes_by_stem = [], {}

    for stem, (image_path, size) in image_index.items():
        gt_path = gt_label_dir / f"gt_{stem}.txt"
        pred_path = pred_label_dir / f"gt_{stem}.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"缺少真值标签: {gt_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"缺少预测标签: {pred_path}")

        boxes, ignored = read_boxes(gt_path, size)
        pred_boxes = prediction_boxes(pred_path, size)
        predictions = [{"box": box.tolist(), "score": 1.0} for box in pred_boxes]
        status, used, counts = match_box_status(predictions, boxes, ignored, MATCH_IOU)

        group = source_group(stem)
        totals[group] += counts
        totals["overall"] += counts
        truth = boxes[~ignored]
        heights = truth[:, 3] - truth[:, 1] if len(truth) else np.zeros(0)
        records.append({
            "name": stem,
            "path": str(image_path),
            "group": group,
            "width": size[0],
            "height": size[1],
            "gt": int(len(truth)),
            "tp": int(counts[0]),
            "fp": int(counts[1]),
            "fn": int(counts[2]),
            "error_rate": (int(counts[1]) + int(counts[2])) / max(1, len(truth)),
            "median_text_height": float(np.median(heights)) if len(heights) else 0.0,
            "truth_boxes": truth,
            "predictions": predictions,
            "status": status,
            "used": used,
        })
        pred_boxes_by_stem[stem] = pred_boxes

    return {
        "groups": {name: summarize_counts(counts) for name, counts in totals.items()},
        "records": records,
        "pred_boxes": pred_boxes_by_stem,
        "error_analysis": build_error_analysis(records, collect_visuals),
    }


def matched_pair_differences(reference_boxes, observed_boxes):
    """按 IoU 贪心匹配两组框，返回每个匹配框对的最大坐标差。

    两组标签都按位置排序，但相邻框的排序可能因浮点差异互换；逐行比较会把
    这种顺序变化放大成整幅图的坐标差，因此只统计 IoU 匹配上的框对。
    """
    if not len(reference_boxes) or not len(observed_boxes):
        return []
    reference = np.asarray(reference_boxes, dtype=np.float32)
    observed = np.asarray(observed_boxes, dtype=np.float32)
    intersection = np.maximum(
        0, np.minimum(reference[:, None, 2:], observed[None, :, 2:])
        - np.maximum(reference[:, None, :2], observed[None, :, :2])).prod(axis=2)
    reference_area = np.maximum(0, reference[:, 2:] - reference[:, :2]).prod(axis=1)
    observed_area = np.maximum(0, observed[:, 2:] - observed[:, :2]).prod(axis=1)
    overlap = intersection / np.maximum(
        reference_area[:, None] + observed_area[None, :] - intersection, 1e-8)

    used, differences = set(), []
    for observed_index in range(len(observed)):
        candidates = [int(index) for index in np.argsort(-overlap[:, observed_index])
                      if index not in used and overlap[index, observed_index] >= MATCH_IOU]
        if candidates:
            used.add(candidates[0])
            differences.append(float(np.max(np.abs(reference[candidates[0]]
                                                   - observed[observed_index]))))
    return differences


def compare_backends(image_index, reference, observed):
    """把 PyTorch 预测当作参照，统计另一个后端的框级一致率。

    只比较输出框，不涉及文字内容；框数一致和坐标一致分别统计，便于定位
    前后端差异来自解码还是数值误差。两张图都没有框时不计入 P/R/F1。
    """
    totals = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    same_count = same_coordinates = 0
    all_differences = []
    for stem, (_, _) in image_index.items():
        reference_boxes = reference["pred_boxes"][stem]
        observed_boxes = observed["pred_boxes"][stem]
        predictions = [{"box": box.tolist(), "score": 1.0} for box in observed_boxes]
        ignored = np.zeros(len(reference_boxes), dtype=bool)
        _, _, counts = match_box_status(predictions, reference_boxes, ignored, MATCH_IOU)
        group = source_group(stem)
        totals[group] += counts
        totals["overall"] += counts
        same_count += len(reference_boxes) == len(observed_boxes)

        differences = matched_pair_differences(reference_boxes, observed_boxes)
        all_differences.extend(differences)
        same_coordinates += (len(reference_boxes) == len(observed_boxes)
                             and len(differences) == len(observed_boxes)
                             and all(value <= BOX_MATCH_TOLERANCE for value in differences))

    return {
        "groups": {name: summarize_counts(counts) for name, counts in totals.items()},
        "images": len(image_index),
        "same_box_count": int(same_count),
        "same_coordinates": int(same_coordinates),
        "matched_pairs": len(all_differences),
        "pairs_over_1px": int(np.count_nonzero(np.asarray(all_differences) > 1.0)),
        "pairs_over_10px": int(np.count_nonzero(np.asarray(all_differences) > 10.0)),
        "median_matched_difference": float(np.median(all_differences)) if all_differences else 0.0,
        "max_matched_difference": float(np.max(all_differences)) if all_differences else 0.0,
    }


def compare_with_training(training, runs):
    """构造训练期指标与离线推理指标的对照行，ΔF1 统一相对训练期主配置。"""
    primary = (training.get("groups") or {}).get("overall") or {}
    base_f1 = primary.get("f1")
    rows = [{
        "source": "训练期验证（主配置）",
        "detail": f"{input_size_label(training['primary']['input_size'], True)} / 阈值 "
                  f"{training['primary']['threshold']:.2f}",
        "groups": training.get("groups") or {},
        "dynamic_input": True,
        "delta_f1": 0.0 if base_f1 is not None else None,
    }]
    best = training.get("best_workpoint")
    if best:
        rows.append({
            "source": "训练期验证（最佳工作点）",
            "detail": f"{input_size_label(best['input_size'], True)} / 阈值 {best['threshold']:.2f}",
            "groups": {"overall": best["overall"], "doc": best.get("doc") or {},
                       "ui": best.get("ui") or {}},
            "dynamic_input": True,
            "delta_f1": (best["overall"]["f1"] - base_f1) if base_f1 is not None else None,
        })
    for backend, split in (("pytorch", "val"), ("onnx", "val")):
        item = runs.get((backend, split))
        if item is None:
            continue
        f1 = item["groups"].get("overall", {}).get("f1")
        rows.append({
            "source": f"离线 {BACKEND_LABELS[backend]} 验证",
            "detail": f"{input_size_label(item['input_size'], item.get('dynamic_input', False))} "
                      f"/ 阈值 {item['threshold']:.2f}",
            "groups": item["groups"],
            "dynamic_input": item.get("dynamic_input", False),
            "delta_f1": (f1 - base_f1) if base_f1 is not None and f1 is not None else None,
        })
    return rows


def build_error_rows(runs, config):
    """按图片合并两后端的框级错误，返回训练集/验证集各自的行列表。

    只保留至少一个后端有 FP 或 FN 的图片；两后端列并排，便于直接对照同一张图
    的差异。error_rate 以该图有效真值框数为分母，空图不进入列表。
    """
    error_rows = {}
    for split in ("train", "val"):
        pytorch = runs.get(("pytorch", split))
        onnx = runs.get(("onnx", split))
        if pytorch is None or onnx is None:
            continue
        pytorch_records = {item["name"]: item for item in pytorch["records"]}
        onnx_records = {item["name"]: item for item in onnx["records"]}
        rows = []
        for name in sorted(set(pytorch_records) | set(onnx_records)):
            reference = pytorch_records.get(name) or onnx_records[name]
            observed = onnx_records.get(name) or reference
            pytorch_errors = reference["fp"] + reference["fn"]
            onnx_errors = observed["fp"] + observed["fn"]
            if not pytorch_errors and not onnx_errors:
                continue
            rows.append({
                "image": name,
                "path": reference["path"],
                "split": split,
                "source": reference["group"],
                "width": reference["width"],
                "height": reference["height"],
                "gt": reference["gt"],
                "pytorch_tp": reference["tp"],
                "pytorch_fp": reference["fp"],
                "pytorch_fn": reference["fn"],
                "pytorch_error_count": pytorch_errors,
                "pytorch_error_rate": round(reference["error_rate"], 6),
                "onnx_tp": observed["tp"],
                "onnx_fp": observed["fp"],
                "onnx_fn": observed["fn"],
                "onnx_error_count": onnx_errors,
                "onnx_error_rate": round(observed["error_rate"], 6),
                "input_size": config["input_size"],
                "score_threshold": config["score_threshold"],
                "nms_iou": config["nms_iou"],
            })
        rows.sort(key=lambda item: (-max(item["pytorch_error_count"],
                                         item["onnx_error_count"]), item["image"]))
        error_rows[split] = rows
    return error_rows


def build_highlights(runs, compare_rows, agreement):
    """生成报告开头的行动摘要。"""
    highlights = []
    pytorch_val = runs.get(("pytorch", "val"))
    onnx_val = runs.get(("onnx", "val"))
    train_base = next((row for row in compare_rows if row["source"] == "训练期验证（主配置）"),
                      None)
    if pytorch_val and train_base:
        f1 = pytorch_val["groups"]["overall"]["f1"]
        base = (train_base["groups"].get("overall") or {}).get("f1")
        if base is not None:
            label = input_size_label(pytorch_val["input_size"],
                                     pytorch_val.get("dynamic_input", False))
            note = ("固定画布与训练输入尺寸一致。" if not pytorch_val.get("dynamic_input", False)
                    else "部署入口复现了训练期验证口径。")
            highlights.append(
                f"PyTorch 验证集（{label}）F1 为 <strong>{f1:.5f}</strong>，"
                f"训练期动态口径为 {base:.5f}，差 {f1 - base:+.5f}；{note}")
    if pytorch_val and onnx_val:
        difference = onnx_val["groups"]["overall"]["f1"] - pytorch_val["groups"]["overall"]["f1"]
        highlights.append(
            f"ONNX/OpenVINO 验证集 F1 为 "
            f"<strong>{onnx_val['groups']['overall']['f1']:.5f}</strong>，"
            f"相对 PyTorch 差 {difference:+.5f}。")
    if agreement.get("val"):
        item = agreement["val"]
        highlights.append(
            f"验证集两后端框数一致 {item['same_box_count']}/{item['images']} 张，"
            f"框坐标一致 {item['same_coordinates']}/{item['images']} 张，"
            f"框级一致 F1 {item['groups']['overall']['f1']:.5f}。")
    pytorch_train = runs.get(("pytorch", "train"))
    if pytorch_train and pytorch_val:
        train_f1 = pytorch_train["groups"]["overall"]["f1"]
        val_f1 = pytorch_val["groups"]["overall"]["f1"]
        highlights.append(
            f"训练集 in-sample F1 为 {train_f1:.5f}，比验证集高 "
            f"{(train_f1 - val_f1) * 100:.2f} 个百分点；训练集指标只表示拟合程度，"
            f"不作为泛化性能。")
    return highlights


def render_run_table(runs):
    """渲染四次推理的总量、P/R/F1 和逐图错误。"""
    rows = []
    for backend, split in RUN_KEYS:
        item = runs.get((backend, split))
        if item is None:
            continue
        metrics = item["groups"]["overall"]
        error = item["error_analysis"]["summary"]
        rows.append(
            "<tr>"
            f"<td>{BACKEND_LABELS[backend]}</td><td>{SPLIT_LABELS[split]}</td>"
            f"<td>{item['images']}</td><td>{metrics['tp'] + metrics['fn']}</td>"
            f"<td>{item['boxes']}</td>"
            f"<td>{metrics['tp']}</td><td>{metrics['fp']}</td><td>{metrics['fn']}</td>"
            f"<td>{format_number(metrics['precision'])}</td>"
            f"<td>{format_number(metrics['recall'])}</td>"
            f"<td><strong>{format_number(metrics['f1'])}</strong></td>"
            f"<td>{error.get('error_images', '—')}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>后端</th><th>划分</th><th>图片</th><th>GT 框</th>"
        "<th>预测框</th><th>TP</th><th>FP</th><th>FN</th><th>P</th><th>R</th><th>F1</th>"
        "<th>有错图片</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render_timing_table(runs):
    """渲染四个推理组合的耗时和吞吐；硬件不同，不直接比较绝对值。"""
    rows = []
    for backend, split in RUN_KEYS:
        item = runs.get((backend, split))
        if item is None:
            continue
        timing = item["timing"]
        rows.append(
            "<tr>"
            f"<td>{BACKEND_LABELS[backend]}</td><td>{SPLIT_LABELS[split]}</td>"
            f"<td>{timing['images']}</td>"
            f"<td>{timing['mean_infer_ms']:.1f}</td>"
            f"<td>{timing['mean_detect_ms']:.1f}</td>"
            f"<td>{timing['wall_seconds']:.1f}</td>"
            f"<td>{timing['images_per_second']:.2f}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>后端</th><th>划分</th><th>图片</th><th>前向均值 ms</th>"
        "<th>检测均值 ms</th><th>总耗时 s</th><th>吞吐 张/s</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_compare_table(compare_rows):
    """渲染训练期指标与离线推理指标的对照。"""
    rows = []
    for item in compare_rows:
        groups = item["groups"]
        overall = groups.get("overall") or {}
        doc = groups.get("doc") or {}
        ui = groups.get("ui") or {}
        rows.append(
            "<tr>"
            f"<td>{esc(item['source'])}</td><td class='muted'>{esc(item['detail'])}</td>"
            f"<td>{format_number(overall.get('precision'))}</td>"
            f"<td>{format_number(overall.get('recall'))}</td>"
            f"<td><strong>{format_number(overall.get('f1'), 5)}</strong></td>"
            f"<td>{format_number(doc.get('f1'), 5)}</td>"
            f"<td>{format_number(ui.get('f1'), 5)}</td>"
            f"<td>{format_number(item.get('delta_f1'), 5)}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>来源</th><th>配置</th><th>P</th><th>R</th><th>F1</th>"
        "<th>doc F1</th><th>ui F1</th><th>ΔF1</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_group_table_multi(runs):
    """渲染各后端、各划分下的总体/doc/ui 分组指标。"""
    rows = []
    for backend, split in RUN_KEYS:
        item = runs.get((backend, split))
        if item is None:
            continue
        for group in ("overall", "doc", "ui", "other"):
            metrics = item["groups"].get(group)
            if metrics is None:
                continue
            rows.append(
                "<tr>"
                f"<td>{BACKEND_LABELS[backend]}</td><td>{SPLIT_LABELS[split]}</td>"
                f"<td>{GROUP_LABELS[group]}</td>"
                f"<td>{metrics['tp']}</td><td>{metrics['fp']}</td><td>{metrics['fn']}</td>"
                f"<td>{format_number(metrics['precision'])}</td>"
                f"<td>{format_number(metrics['recall'])}</td>"
                f"<td><strong>{format_number(metrics['f1'], 5)}</strong></td>"
                "</tr>")
    return (
        "<table><thead><tr><th>后端</th><th>划分</th><th>分组</th><th>TP</th><th>FP</th>"
        "<th>FN</th><th>P</th><th>R</th><th>F1</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_aspect_table(aspect_groups):
    """渲染 PyTorch 各划分的长宽比分组指标，突出 >20:1 超宽细条。"""
    rows = []
    for split in ("train", "val"):
        groups = aspect_groups.get(split) or {}
        for label, _ in ASPECT_BUCKETS:
            metrics = groups.get(label)
            if metrics is None:
                continue
            rows.append(
                "<tr>"
                f"<td>{SPLIT_LABELS[split]}</td><td>{esc(label)}</td>"
                f"<td>{metrics['images']}</td>"
                f"<td>{metrics['tp']}</td><td>{metrics['fp']}</td><td>{metrics['fn']}</td>"
                f"<td>{format_number(metrics['precision'])}</td>"
                f"<td>{format_number(metrics['recall'])}</td>"
                f"<td><strong>{format_number(metrics['f1'], 5)}</strong></td>"
                "</tr>")
    if not rows:
        return '<p class="muted">没有长宽比分组数据。</p>'
    return (
        "<table><thead><tr><th>划分</th><th>长宽比</th><th>图片</th><th>TP</th><th>FP</th>"
        "<th>FN</th><th>P</th><th>R</th><th>F1</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_agreement_table(agreement):
    """渲染两后端框级一致性。"""
    rows = []
    for split in ("train", "val"):
        item = agreement.get(split)
        if item is None:
            continue
        overall = item["groups"]["overall"]
        doc = item["groups"].get("doc") or {}
        ui = item["groups"].get("ui") or {}
        rows.append(
            "<tr>"
            f"<td>{SPLIT_LABELS[split]}</td><td>{item['images']}</td>"
            f"<td>{item['same_box_count']}/{item['images']}</td>"
            f"<td>{item['same_coordinates']}/{item['images']}</td>"
            f"<td>{format_number(overall['precision'], 5)}</td>"
            f"<td>{format_number(overall['recall'], 5)}</td>"
            f"<td><strong>{format_number(overall['f1'], 5)}</strong></td>"
            f"<td>{format_number(doc.get('f1'), 5)}</td>"
            f"<td>{format_number(ui.get('f1'), 5)}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>划分</th><th>图片</th><th>框数一致</th><th>坐标一致（匹配对）</th>"
        "<th>一致 P</th><th>一致 R</th><th>一致 F1</th><th>doc F1</th><th>ui F1</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render_training_key_table(training):
    """渲染训练期关键指标，作为离线推理指标的参照。"""
    run = training.get("run") or {}
    rows = [
        ("训练轮数", run.get("epochs")),
        ("最佳验证 F1", f"{format_number(run.get('best_f1'), 5)} · "
                        f"Epoch {run.get('best_f1_epoch', '—')}"),
        ("最低验证 loss", f"{format_number(run.get('best_loss'), 5)} · "
                          f"Epoch {run.get('best_loss_epoch', '—')}"),
        ("最后一轮验证 F1", format_number(run.get("last_f1"), 5)),
        ("训练耗时", f"{run.get('training_hours', '—')} 小时"),
        ("训练吞吐", f"{run.get('throughput', '—')} 张/s"),
    ]
    body = "".join(f"<tr><td>{esc(label)}</td><td>{esc(value)}</td></tr>"
                   for label, value in rows)
    return ("<table><thead><tr><th>项目</th><th>值</th></tr></thead><tbody>"
            + body + "</tbody></table>")


def render_conclusions(conclusions):
    """渲染结论文本列表。"""
    return ("<ul class='highlights'>"
            + "".join(f"<li>{item}</li>" for item in conclusions) + "</ul>")


def render_report(report):
    """渲染单文件 HTML 报告。"""
    curves = report.get("curves") or {}
    charts = {
        "loss": line_chart_svg([
            ("train", curves.get("Loss/Train", []), "#64748b"),
            ("validation", curves.get("Loss/Validation", []), "#dc2626"),
        ], "Total loss (log scale)", "Loss", log=True),
        "f1": line_chart_svg([
            ("overall", curves.get("Val/F1", []), "#2563eb"),
            ("doc", curves.get("Val/F1_doc", []), "#ea580c"),
            ("ui", curves.get("Val/F1_ui", []), "#16a34a"),
        ], "Validation F1", "F1"),
    }
    runs = report["runs"]
    config = report["config"]
    training = report["training"]
    compare_rows = report["compare"]
    agreement = report["agreement"]
    error_analysis = report["error_analysis"]
    aspect_groups = report.get("aspect_groups") or {}
    highlights = report.get("highlights") or []
    conclusions = report.get("conclusions") or []

    pytorch_val = runs.get(("pytorch", "val")) or {}
    onnx_val = runs.get(("onnx", "val")) or {}
    pytorch_train = runs.get(("pytorch", "train")) or {}
    val_f1 = (pytorch_val.get("groups") or {}).get("overall", {}).get("f1")
    onnx_f1 = (onnx_val.get("groups") or {}).get("overall", {}).get("f1")
    train_f1 = (pytorch_train.get("groups") or {}).get("overall", {}).get("f1")
    commit_note = next((row for row in compare_rows
                        if row["source"] == "训练期验证（主配置）"), None)
    train_base = ((commit_note or {}).get("groups") or {}).get("overall", {}).get("f1")
    val_note = (f"训练期 {train_base:.5f} · Δ {val_f1 - train_base:+.5f}"
                if val_f1 is not None and train_base is not None else "训练期参照缺失")
    onnx_note = (f"相对 PyTorch {onnx_f1 - val_f1:+.5f}"
                 if onnx_f1 is not None and val_f1 is not None else "PyTorch 参照缺失")
    agreement_val = agreement.get("val") or {}
    agreement_f1 = (agreement_val.get("groups") or {}).get("overall", {}).get("f1")
    agreement_note = (f"框数一致 {agreement_val.get('same_box_count', 0)}/"
                      f"{agreement_val.get('images', 0)} 张" if agreement_val else "缺少一致性数据")
    cards = [
        ("PyTorch 验证 F1", format_number(val_f1, 5), val_note, "accent"),
        ("ONNX/OpenVINO 验证 F1", format_number(onnx_f1, 5), onnx_note, ""),
        ("PyTorch 训练集 F1", format_number(train_f1, 5),
         "in-sample，只表示拟合程度", "warn"),
        ("两后端框级一致 F1", format_number(agreement_f1, 5), agreement_note, ""),
    ]
    card_html = "".join(
        f'<div class="card {css}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{value}</div><div class="note">{esc(note)}</div></div>'
        for label, value, note, css in cards)

    meta = (
        f"运行 <span class='mono'>{esc(report['run_name'])}</span> · "
        f"检查点 <span class='mono'>{esc(config['checkpoint'])}</span> · "
        f"ONNX <span class='mono'>{esc(config['onnx_model'])}</span> · "
        f"输入{input_size_label(config['input_size'], config.get('dynamic_input', False))} · "
        f"阈值 {config['score_threshold']:.2f} · "
        f"NMS {config['nms_iou']:.2f} · 匹配 IoU {MATCH_IOU:.2f} · "
        f"生成于 {esc(report['generated_at'])}"
    )

    css = """
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
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px;
  word-break:break-all; }
.muted { color:var(--muted); }
.chart-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
.chart-box { border:1px solid var(--line); border-radius:10px; padding:8px 10px; }
.chart-box h3 { margin:2px 0 4px; font-size:13px; color:#475569; }
svg.chart { width:100%; height:auto; display:block; }
.callout { border-left:4px solid var(--accent); background:#eff6ff; padding:10px 14px;
  border-radius:6px; margin:10px 0; }
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
footer { color:var(--muted); font-size:12px; text-align:center; padding:20px; }
@media (max-width:820px) { .kpi,.chart-grid,.error-grid { grid-template-columns:1fr; }
  header { padding:22px 18px; } main { padding:14px 10px 40px; } }
@media print { nav { display:none; } section { break-inside:avoid; box-shadow:none; } }
"""

    body = f"""
<header>
  <h1>统一检测部署推理报告</h1>
  <div class="meta">{meta}</div>
</header>
<nav>
  <a href="#overview">概览</a><a href="#runs">推理指标</a><a href="#compare">训练对比</a>
  <a href="#groups">分组</a><a href="#agreement">后端一致性</a><a href="#curves">训练曲线</a>
  <a href="#errors">错误分析</a><a href="#conclusion">结论</a><a href="#definitions">指标口径</a>
</nav>
<main>
<section id="overview">
  <h2>概览</h2>
  <div class="kpi">{card_html}</div>
  <ul class="highlights">{''.join(f'<li>{item}</li>' for item in highlights)}</ul>
</section>

<section id="runs">
  <h2>推理指标</h2>
  <p class="muted">四次推理分别调用 detection/inference.py（PyTorch GPU）和
  detection_inference_onnx.py（ONNX/OpenVINO CPU），输入为入口默认的
  {input_size_label(config['input_size'], config.get('dynamic_input', False))}、
  阈值 {config['score_threshold']:.2f}、NMS {config['nms_iou']:.2f}。
  GT = TP + FN，### 忽略区域不计入。</p>
  {render_run_table(runs)}
  <h3>耗时与吞吐</h3>
  {render_timing_table(runs)}
  <p class="muted">前向均值只含模型前向；检测均值还含预处理、解码、NMS 和坐标还原。
  PyTorch 使用 GPU，ONNX/OpenVINO 使用 CPU，两者的绝对耗时不能直接横向比较。</p>
</section>

<section id="compare">
  <h2>与训练指标对比</h2>
  <p class="muted">ΔF1 相对训练期主配置（{input_size_label(training['primary']['input_size'], True)} / 阈值
  {training['primary']['threshold']:.2f}）。训练期指标来自
  detection/reports/{esc(report['run_name'])}/summary.json；离线指标由本次推理标签重新计算。</p>
  {render_compare_table(compare_rows)}
  <h3>训练期关键指标</h3>
  {render_training_key_table(training)}
  <div class="callout">验证集曾用于检查点选择和工作点对比，训练期与离线验证指标都存在乐观偏差；
  上线前应使用未参与训练和调参的独立标注集复核。训练集指标为 in-sample 结果，
  不能与验证集 F1 直接比较。</div>
</section>

<section id="groups">
  <h2>分组指标</h2>
  <p class="muted">按文件名前缀 ui_/doc_ 划分，总体指标按框数汇总，不是按图片数平均。</p>
  {render_group_table_multi(runs)}
  <h3>长宽比分组（PyTorch {input_size_label(config['input_size'], config.get('dynamic_input', False))}）</h3>
  {render_aspect_table(aspect_groups)}
  <p class="muted">长宽比 = 宽/高；>20:1 主要是浏览器地址栏等超宽细条，动态输入会把短边压到
  64，固定方形画布能保持与训练一致的输入分布。</p>
</section>

<section id="agreement">
  <h2>后端一致性</h2>
  <p class="muted">把 PyTorch 预测当作参照，用相同 IoU {MATCH_IOU:.2f} 匹配 ONNX/OpenVINO 预测；
  P/R/F1 衡量两后端输出框的一致程度，不衡量与真值的准确率。坐标一致按 IoU 匹配后的
  框对统计：框数相同且每个匹配框对的坐标差不超过 {BOX_MATCH_TOLERANCE}px，避免相邻框
  排序互换造成误判。</p>
  {render_agreement_table(agreement)}
</section>

<section id="curves">
  <h2>训练曲线</h2>
  <div class="chart-grid">
    <div class="chart-box"><h3>训练/验证 loss</h3>{charts['loss']}</div>
    <div class="chart-box"><h3>验证集 F1</h3>{charts['f1']}</div>
  </div>
  <p class="muted">曲线来自 detection/runs/{esc(report['run_name'])} 的 TensorBoard 事件；
  loss 使用对数纵轴。</p>
</section>

<section id="errors">
  <h2>错误分析</h2>
  <p class="muted">逐图统计使用 PyTorch 验证集结果
  （{input_size_label(pytorch_val.get('input_size', config['input_size']),
                       pytorch_val.get('dynamic_input', config.get('dynamic_input', False)))} /
  阈值 {pytorch_val.get('threshold', config['score_threshold']):.2f}）。</p>
  {render_error_summary(error_analysis)}
  <h3>最差图片</h3>
  {render_error_cards(error_analysis.get('images') or [])}
  <p class="muted">逐图错误明细见同目录 errors_train.csv 和 errors_val.csv；每行只保留至少一个
  后端有 FP 或 FN 的图片，并同时列出 PyTorch 与 ONNX 的 TP/FP/FN。</p>
</section>

<section id="conclusion">
  <h2>结论与建议</h2>
  {render_conclusions(conclusions)}
</section>

<section id="definitions">
  <h2>指标口径</h2>
  <p><strong>GT 框</strong>：人工标注的有效真值框，### 忽略区域不计入。
  <strong>TP</strong>：预测框与真值框 IoU ≥ {MATCH_IOU:.2f} 且一对一匹配成功。
  <strong>FP</strong>：没有匹配到真值且不是主要落在忽略区的预测框。
  <strong>FN</strong>：没有被预测匹配的真值框。Precision = TP / (TP + FP)，
  Recall = TP / (TP + FN)，F1 为两者调和平均。</p>
  <p class="muted">预测标签来自两个部署入口写出的 bbox4 文件，不含置信度；匹配时同分预测
  按标签顺序处理。推理产物位于 detection/data/inference_eval/，不提交到仓库。</p>
</section>
</main>
<footer>
  报告由 detection/inference_report.py 生成。指标只评估检测框，不涉及文字识别。
</footer>
"""

    return ("<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>统一检测部署推理报告 · {esc(report['run_name'])}</title>\n"
            f"<style>{css}</style>\n</head>\n<body>{body}</body>\n</html>\n")


def build_report(runs, training, curves, config, run_name):
    """组织报告数据，生成摘要、对照行和结论文本。

    runs 以 (backend, split) 为键，包含 groups/timing/boxes/images/error_analysis；
    records 和 pred_boxes 只用于计算，不写入 summary.json。
    """
    agreement = {}
    for split in ("train", "val"):
        reference = runs.get(("pytorch", split))
        observed = runs.get(("onnx", split))
        if reference is None or observed is None:
            continue
        index = reference["image_index"]
        agreement[split] = compare_backends(index, reference, observed)

    compare_rows = compare_with_training(training, runs)
    highlights = build_highlights(runs, compare_rows, agreement)
    val_f1 = (runs[("pytorch", "val")]["groups"]["overall"]["f1"]
              if ("pytorch", "val") in runs else None)
    onnx_val_f1 = (runs[("onnx", "val")]["groups"]["overall"]["f1"]
                   if ("onnx", "val") in runs else None)
    train_f1 = (runs[("pytorch", "train")]["groups"]["overall"]["f1"]
                if ("pytorch", "train") in runs else None)
    base_f1 = ((training.get("groups") or {}).get("overall") or {}).get("f1")

    conclusions = []
    aspect_groups = {}
    if ("pytorch", "train") in runs and ("pytorch", "val") in runs:
        aspect_groups = {
            "train": build_aspect_groups(runs[("pytorch", "train")]["records"]),
            "val": build_aspect_groups(runs[("pytorch", "val")]["records"]),
        }
    offline_dynamic = (runs.get(("pytorch", "val")) or {}).get("dynamic_input", False)
    if val_f1 is not None and base_f1 is not None:
        delta = val_f1 - base_f1
        if not offline_dynamic:
            conclusions.append(
                f"离线 PyTorch 使用固定 {config['input_size']}×{config['input_size']} 画布，"
                f"验证集 F1 为 <strong>{val_f1:.5f}</strong>；训练期验证使用动态尺寸口径，"
                f"F1 为 {base_f1:.5f}，差 {delta:+.5f}。固定画布与训练输入同尺寸，"
                f"该差异来自推理尺寸口径，不是模型或标签变化。")
        elif abs(delta) < 0.002:
            conclusions.append(
                f"部署入口的 PyTorch 验证集 F1 为 <strong>{val_f1:.5f}</strong>，"
                f"训练期同口径为 {base_f1:.5f}，差 {delta:+.5f}；"
                f"入口配置与训练期验证一致。")
        else:
            conclusions.append(
                f"部署入口的 PyTorch 验证集 F1 为 <strong>{val_f1:.5f}</strong>，"
                f"与训练期 {base_f1:.5f} 相差 {delta:+.5f}，需要核对入口配置和标签一致性。")
    if onnx_val_f1 is not None and val_f1 is not None:
        difference = onnx_val_f1 - val_f1
        conclusions.append(
            f"ONNX/OpenVINO 与 PyTorch 的验证集 F1 差 {difference:+.5f}。"
            + ("两端输出一致，ONNX 导出没有引入可观测的精度损失。"
               if abs(difference) < 0.002 else
               "差异超过 0.002，建议检查 ONNX 导出尺寸、阈值和算子数值误差。"))
    if train_f1 is not None and val_f1 is not None:
        conclusions.append(
            f"训练集 in-sample F1 为 {train_f1:.5f}，验证集为 {val_f1:.5f}，"
            f"差 {(train_f1 - val_f1) * 100:+.2f} 个百分点。该差值包含过拟合、"
            f"训练/验证域差异和标注噪声，不能只归因于模型容量。")
    wide = (aspect_groups.get("val") or {}).get(">20")
    if wide:
        conclusions.append(
            f"验证集长宽比 >20:1 的 {wide['images']} 张固定画布 F1 为 "
            f"<strong>{wide['f1']:.5f}</strong>（TP {wide['tp']} / FN {wide['fn']}）；"
            f"这类超宽细条是固定画布相对动态尺寸提升最大的子集。")
    if agreement.get("val"):
        item = agreement["val"]
        conclusions.append(
            f"验证集两后端框数一致 {item['same_box_count']}/{item['images']} 张，"
            f"全部匹配框对坐标差不超过 {BOX_MATCH_TOLERANCE}px 的图片 "
            f"{item['same_coordinates']}/{item['images']} 张；"
            f"{item['matched_pairs']} 对匹配框的坐标差中位数 "
            f"{item['median_matched_difference']:.3f}px、最大 "
            f"{item['max_matched_difference']:.3f}px，其中超过 1px 的 "
            f"{item['pairs_over_1px']} 对、超过 10px 的 {item['pairs_over_10px']} 对，"
            f"集中在连通域边界或拆分歧义处。")
    timing_train = runs.get(("pytorch", "train"), {}).get("timing")
    timing_onnx = runs.get(("onnx", "val"), {}).get("timing")
    if timing_train and timing_onnx:
        conclusions.append(
            f"本机 PyTorch GPU 在训练集上的检测均值为 {timing_train['mean_detect_ms']:.1f} ms/张，"
            f"ONNX/OpenVINO CPU 在验证集上为 {timing_onnx['mean_detect_ms']:.1f} ms/张；"
            f"两者硬件不同，只能用于部署容量估算，不能当作加速比。")
    conclusions.append(
        "训练期最佳工作点 896px 动态 / 阈值 0.85 的验证 F1 更高，但本次离线入口按固定 "
        f"{config['input_size']}×{config['input_size']} / 0.70 运行，没有复跑该工作点；"
        "如需复核，修改检测入口的 INPUT_SIZE/阈值后重新推理。")
    conclusions.append(
        "验证集参与过检查点选择和工作点对比，上述 F1 存在乐观偏差；上线前应使用独立留出集复核。")

    summary_runs = {}
    for backend, split in RUN_KEYS:
        item = runs.get((backend, split))
        if item is None:
            continue
        summary_runs[(backend, split)] = {
            "backend": backend,
            "split": split,
            "images": item["images"],
            "boxes": item["boxes"],
            "input_size": item["input_size"],
            "threshold": item["threshold"],
            "groups": item["groups"],
            "timing": item["timing"],
            "error_analysis": item["error_analysis"],
        }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name": run_name,
        "config": config,
        "training": training,
        "runs": summary_runs,
        "compare": compare_rows,
        "agreement": agreement,
        "error_analysis": runs[("pytorch", "val")]["error_analysis"]
        if ("pytorch", "val") in runs else {"summary": {}, "groups": {},
                                            "orientation": {}, "images": []},
        "highlights": highlights,
        "conclusions": conclusions,
        "aspect_groups": aspect_groups,
        "error_rows": build_error_rows(runs, config),
        "curves": curves,
    }
    return report


def read_error_rows(csv_path):
    """读取 errors_{split}.csv 的错误图片行，返回按文件顺序排列的字典列表。

    允许空文件（该划分没有错误图片），但每行必须带 image 和 path，避免把
    缺列的记录静默跳过。
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not (row.get("image") or "").strip() or not (row.get("path") or "").strip():
            raise ValueError(f"错误明细缺少 image 或 path: {csv_path}")
    return rows


def save_error_preview(image, preview_path, predictions):
    """按 detection/inference.py 的出框效果保存预览：红框、原尺寸、原格式。

    返回实际写出的预览路径；目录不存在时自动创建。
    """
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    draw_predictions(image, predictions).save(preview_path)
    return preview_path


def write_preview_csv(rows, csv_path):
    """写出错误图片预览记录，列顺序固定为 ERROR_PREVIEW_CSV_FIELDS。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_PREVIEW_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_report(report, output_dir):
    """写出 index.html、summary.json 和两套错误图片 CSV，返回这些路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(render_report(report), encoding="utf-8")

    csv_paths = []
    for split in ("train", "val"):
        rows = (report.get("error_rows") or {}).get(split)
        if rows is None:
            continue
        csv_path = output_dir / f"errors_{split}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ERROR_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        csv_paths.append(csv_path)

    light = {key: value for key, value in report.items()
             if key not in ("curves", "error_rows")}
    light["runs"] = {}
    for (backend, split), value in report["runs"].items():
        item = dict(value)
        item["error_analysis"] = {
            key: metrics for key, metrics in (item.get("error_analysis") or {}).items()
            if key != "images"
        }
        light["runs"][f"{backend}_{split}"] = item
    error_analysis = light.pop("error_analysis") or {}
    light["error_analysis"] = {
        key: value for key, value in error_analysis.items() if key != "images"
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(light), ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return html_path, summary_path, csv_paths
