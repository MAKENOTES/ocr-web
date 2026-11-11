"""汇总“小图用 256、其余用 768”输入策略的精度和耗时，生成静态 HTML 报告。

报告对比两种空间尺寸策略：
- baseline：所有图片固定 768×768；
- policy：宽高都小于 256 的图片使用 256×256，其余仍用 768×768。

PyTorch 覆盖训练集和验证集；ONNX/OpenVINO 只在验证集上复算 CPU 耗时和精度。
所有指标按训练期口径（匹配 IoU 0.5、### 忽略区）计算，图表和表格内嵌在 HTML 中。
"""
import json
from datetime import datetime
from pathlib import Path

from detection.utils.training_report import (
    esc,
    format_number,
    format_percent,
    json_safe,
)


SMALL_MAX_SIDE = 256  # 小图判定阈值：宽和高都小于该值才使用小尺寸，单位 px。
SMALL_INPUT_SIZE = 256  # 小图输入边长，单位 px；与训练尺寸 768 不同，会改变文字缩放。
LARGE_INPUT_SIZE = 768  # 其余图片的固定方形输入边长，单位 px；与训练尺寸一致。
BASELINE_POLICY = "baseline"  # 全部使用固定 768×768 的策略名。
MIXED_POLICY = "policy"  # 小图 256、其余 768 的策略名。
POLICY_LABELS = {BASELINE_POLICY: "固定 768×768", MIXED_POLICY: "小图 256 / 其余 768"}  # 策略显示名。
GROUP_LABELS = {"overall": "总体", "small": "小图 <256×256", "large": "大图 ≥256×256",
                "ui": "界面小图 ui", "doc": "文档页 doc", "other": "其他"}  # 指标分组显示名。
GROUP_ORDER = ("overall", "small", "large", "ui", "doc", "other")  # 表格分组顺序。


def select_input_size(width, height):
    """返回该图片使用的方形输入边长；宽高都小于 256 时用 256，否则用 768。"""
    return SMALL_INPUT_SIZE if width < SMALL_MAX_SIDE and height < SMALL_MAX_SIDE else LARGE_INPUT_SIZE


def timing_summary(timing):
    """把累计耗时整理成均值、吞吐和分组均值。"""
    images = timing["images"]
    total_ms = timing["total_ms"]
    return {
        "images": images,
        "total_ms": total_ms,
        "total_seconds": total_ms / 1000,
        "mean_detect_ms": total_ms / images if images else 0.0,
        "images_per_second": images / (total_ms / 1000) if total_ms else 0.0,
        "small_images": timing["small_images"],
        "small_mean_detect_ms": (timing["small_ms"] / timing["small_images"]
                                 if timing["small_images"] else 0.0),
        "large_images": timing["large_images"],
        "large_mean_detect_ms": (timing["large_ms"] / timing["large_images"]
                                 if timing["large_images"] else 0.0),
    }


def render_kpis(report):
    """渲染概览 KPI：验证集精度变化和 PyTorch 耗时加速比。"""
    baseline = report["runs"]["pytorch"]["val"]["baseline"]
    policy = report["runs"]["pytorch"]["val"]["policy"]
    baseline_f1 = baseline["groups"]["overall"]["f1"]
    policy_f1 = policy["groups"]["overall"]["f1"]
    baseline_ms = baseline["latency"]["mean_detect_ms"]
    policy_ms = policy["latency"]["mean_detect_ms"]
    cards = [
        ("验证集 F1（固定 768）", format_number(baseline_f1, 5),
         "所有图片 768×768", ""),
        ("验证集 F1（小图 256）", format_number(policy_f1, 5),
         f"Δ {policy_f1 - baseline_f1:+.5f}", "accent"),
        ("验证集检测耗时", f"{baseline_ms:.1f} → {policy_ms:.1f} ms",
         f"加速 {baseline_ms / policy_ms:.2f}×" if policy_ms else "—", "warn"),
        ("小图覆盖", format_percent(report["dataset"]["val"]["small_ratio"], 1),
         f"{report['dataset']['val']['small']}/{report['dataset']['val']['images']} 张",
         ""),
    ]
    return "".join(
        f'<div class="card {css}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{value}</div><div class="note">{esc(note)}</div></div>'
        for label, value, note, css in cards)


def render_dataset_table(dataset):
    """渲染训练/验证集的尺寸分布和像素计算量比例。"""
    rows = []
    for split, label in (("train", "训练集"), ("val", "验证集")):
        item = dataset[split]
        rows.append(
            f"<tr><td>{label}</td><td>{item['images']}</td>"
            f"<td>{item['small']}</td><td>{format_percent(item['small_ratio'], 1)}</td>"
            f"<td>{item['large']}</td><td>{item['pixel_ratio']:.2f}×</td></tr>")
    return (
        "<table><thead><tr><th>划分</th><th>图片</th><th>小图 &lt;256×256</th>"
        "<th>小图占比</th><th>大图</th><th>像素计算量/全 768</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_metric_comparison(runs):
    """按后端、划分和分组对比 baseline 与 policy 的 P/R/F1。"""
    rows = []
    for backend in ("pytorch", "onnx"):
        for split in ("train", "val"):
            item = runs.get(backend, {}).get(split)
            if item is None:
                continue
            baseline = item[BASELINE_POLICY]["groups"]
            policy = item[MIXED_POLICY]["groups"]
            for group in GROUP_ORDER:
                b = baseline.get(group)
                p = policy.get(group)
                if b is None and p is None:
                    continue
                b = b or {}
                p = p or {}
                delta = (p.get("f1", 0.0) - b.get("f1", 0.0)
                         if b.get("f1") is not None and p.get("f1") is not None else None)
                rows.append(
                    "<tr>"
                    f"<td>{'PyTorch' if backend == 'pytorch' else 'ONNX/OpenVINO'}</td>"
                    f"<td>{'训练集' if split == 'train' else '验证集'}</td>"
                    f"<td>{GROUP_LABELS[group]}</td>"
                    f"<td>{format_number(b.get('precision'))}</td>"
                    f"<td>{format_number(b.get('recall'))}</td>"
                    f"<td>{format_number(b.get('f1'), 5)}</td>"
                    f"<td>{format_number(p.get('precision'))}</td>"
                    f"<td>{format_number(p.get('recall'))}</td>"
                    f"<td><strong>{format_number(p.get('f1'), 5)}</strong></td>"
                    f"<td>{format_number(delta, 5)}</td>"
                    "</tr>")
    return (
        "<table><thead><tr><th>后端</th><th>划分</th><th>分组</th>"
        "<th>Baseline P</th><th>Baseline R</th><th>Baseline F1</th>"
        "<th>Policy P</th><th>Policy R</th><th>Policy F1</th><th>ΔF1</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render_counts_table(runs):
    """渲染两种策略的 TP/FP/FN，便于核对错误构成。"""
    rows = []
    for backend in ("pytorch", "onnx"):
        for split in ("train", "val"):
            item = runs.get(backend, {}).get(split)
            if item is None:
                continue
            for policy in (BASELINE_POLICY, MIXED_POLICY):
                for group in GROUP_ORDER:
                    metrics = item[policy]["groups"].get(group)
                    if metrics is None:
                        continue
                    rows.append(
                        "<tr>"
                        f"<td>{'PyTorch' if backend == 'pytorch' else 'ONNX/OpenVINO'}</td>"
                        f"<td>{'训练集' if split == 'train' else '验证集'}</td>"
                        f"<td>{POLICY_LABELS[policy]}</td><td>{GROUP_LABELS[group]}</td>"
                        f"<td>{metrics['tp']}</td><td>{metrics['fp']}</td><td>{metrics['fn']}</td>"
                        "</tr>")
    return (
        "<table><thead><tr><th>后端</th><th>划分</th><th>策略</th><th>分组</th>"
        "<th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def render_latency_table(runs):
    """渲染各策略的检测耗时和吞吐；小图/大图均值分开列出。"""
    rows = []
    for backend in ("pytorch", "onnx"):
        for split in ("train", "val"):
            item = runs.get(backend, {}).get(split)
            if item is None:
                continue
            for policy in (BASELINE_POLICY, MIXED_POLICY):
                latency = item[policy]["latency"]
                rows.append(
                    "<tr>"
                    f"<td>{'PyTorch' if backend == 'pytorch' else 'ONNX/OpenVINO'}</td>"
                    f"<td>{'训练集' if split == 'train' else '验证集'}</td>"
                    f"<td>{POLICY_LABELS[policy]}</td><td>{latency['images']}</td>"
                    f"<td>{latency['mean_detect_ms']:.1f}</td>"
                    f"<td>{latency['images_per_second']:.2f}</td>"
                    f"<td>{latency['small_mean_detect_ms']:.1f}</td>"
                    f"<td>{latency['large_mean_detect_ms']:.1f}</td>"
                    "</tr>")
    return (
        "<table><thead><tr><th>后端</th><th>划分</th><th>策略</th><th>图片</th>"
        "<th>检测均值 ms</th><th>吞吐 张/s</th><th>小图均值 ms</th><th>大图均值 ms</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render_conclusions(conclusions):
    """渲染结论列表。"""
    return ("<ul class='highlights'>"
            + "".join(f"<li>{item}</li>" for item in conclusions) + "</ul>")


def render_report(report):
    """渲染单文件 HTML 报告。"""
    config = report["config"]
    runs = report["runs"]
    conclusions = report["conclusions"]
    meta = (
        f"运行 <span class='mono'>{esc(report['run_name'])}</span> · "
        f"检查点 <span class='mono'>{esc(config['checkpoint'])}</span> · "
        f"小图阈值 {config['small_max_side']}×{config['small_max_side']} · "
        f"小图输入 {config['small_input_size']}×{config['small_input_size']} · "
        f"其余输入 {config['large_input_size']}×{config['large_input_size']} · "
        f"阈值 {config['score_threshold']:.2f} · NMS {config['nms_iou']:.2f} · "
        f"生成于 {esc(report['generated_at'])}"
    )

    css = """
:root { --bg:#f5f7fb; --panel:#fff; --text:#172033; --muted:#64748b; --line:#e2e8f0;
  --accent:#2563eb; --warn:#dc2626; --ok:#16a34a; --amber:#f59e0b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
  line-height:1.65; }
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
.callout { border-left:4px solid var(--accent); background:#eff6ff; padding:10px 14px;
  border-radius:6px; margin:10px 0; }
ul.highlights { margin:6px 0 0; padding-left:20px; }
ul.highlights li { margin:4px 0; }
footer { color:var(--muted); font-size:12px; text-align:center; padding:20px; }
@media (max-width:820px) { .kpi { grid-template-columns:1fr; }
  header { padding:22px 18px; } main { padding:14px 10px 40px; } }
@media print { nav { display:none; } section { break-inside:avoid; box-shadow:none; } }
"""

    body = f"""
<header>
  <h1>统一检测输入尺寸策略对比报告</h1>
  <div class="meta">{meta}</div>
</header>
<nav>
  <a href="#overview">概览</a><a href="#data">尺寸分布</a><a href="#metrics">精度对比</a>
  <a href="#latency">耗时对比</a><a href="#counts">TP/FP/FN</a><a href="#conclusion">结论</a>
</nav>
<main>
<section id="overview">
  <h2>概览</h2>
  <div class="kpi">{render_kpis(report)}</div>
  {render_conclusions(conclusions)}
</section>

<section id="data">
  <h2>尺寸分布</h2>
  <p class="muted">小图定义为宽和高都小于 {config['small_max_side']}px；小图使用
  {config['small_input_size']}×{config['small_input_size']} 输入，其余使用
  {config['large_input_size']}×{config['large_input_size']}。像素计算量按
  （小图数×256² + 大图数×768²）/（全部图片×768²）估算，不含补边以外的实际访存差异。</p>
  {render_dataset_table(report['dataset'])}
</section>

<section id="metrics">
  <h2>精度对比</h2>
  <p class="muted">指标按框级一对一匹配、IoU 0.5 计算；小图/大图分组按原图宽高判定，
  与输入尺寸策略一致。ΔF1 = Policy - Baseline。</p>
  {render_metric_comparison(runs)}
</section>

<section id="latency">
  <h2>耗时对比</h2>
  <p class="muted">检测均值包含预处理、模型前向、解码、NMS 和坐标还原；PyTorch 使用 GPU，
  ONNX/OpenVINO 使用 CPU，两者不能直接横向比较。ONNX 训练集未重跑，精度和耗时由
  PyTorch 训练集与 ONNX 验证集代表。</p>
  {render_latency_table(runs)}
</section>

<section id="counts">
  <h2>TP/FP/FN 明细</h2>
  {render_counts_table(runs)}
</section>

<section id="conclusion">
  <h2>结论与建议</h2>
  {render_conclusions(conclusions)}
</section>

<section id="definitions">
  <h2>指标口径</h2>
  <p><strong>GT</strong>：人工标注有效框，### 忽略区不计入。
  <strong>TP</strong>：预测框与真值框 IoU ≥ 0.50 且一对一匹配成功；
  <strong>FP</strong>：未匹配到真值的预测；<strong>FN</strong>：未匹配的真值。
  Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，F1 为调和平均。</p>
  <p class="muted">验证集参与过检查点选择，精度数字存在乐观偏差；策略取舍应结合独立留出集和实际业务图片复核。</p>
</section>
</main>
<footer>报告由 detection/report_size_policy.py 生成。指标只评估检测框，不涉及文字识别。</footer>
"""
    return ("<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>统一检测输入尺寸策略对比报告 · {esc(report['run_name'])}</title>\n"
            f"<style>{css}</style>\n</head>\n<body>{body}</body>\n</html>\n")


def write_report(report, output_dir):
    """写出 index.html 和 summary.json，返回两个路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(render_report(report), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return html_path, summary_path
