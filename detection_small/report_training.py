"""生成小图检测训练运行的静态 HTML 分析报告。

用法：
    .venv/bin/python detection_small/report_training.py \
        --run detection_small/checkpoints/ocr_experiment_<时间戳>

默认输出到 detection_small/reports/<run>/index.html；报告自包含图表和问题图片，
可直接用浏览器打开，不需要启动 TensorBoard。
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from detection_small.utils.training_eval import (
    REPORTS_DIR,
    collect_report,
    json_safe,
    render_report,
    report_summary,
)


def main(argv=None):
    """命令行入口：生成指定运行目录的静态 HTML 报告。"""
    parser = argparse.ArgumentParser(
        description="生成小图检测训练运行的静态 HTML 分析报告。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", required=True,
                        help="训练检查点目录，例如 detection_small/checkpoints/ocr_experiment_...")
    parser.add_argument("--output", default=None,
                        help="报告输出目录；默认为 detection_small/reports/<run 名>")
    args = parser.parse_args(argv)

    run_dir = Path(args.run).expanduser()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    if not run_dir.is_dir():
        parser.error(f"检查点目录不存在: {run_dir}")

    output_dir = Path(args.output).expanduser() if args.output else REPORTS_DIR / run_dir.name
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"分析运行: {run_dir.name}")
    print(f"推理设备: {device}")
    report = collect_report(run_dir, device)
    html_path = output_dir / "index.html"
    html_path.write_text(render_report(report), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_safe(report_summary(report)), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"HTML 报告: {html_path}")
    print(f"JSON 摘要: {summary_path}")


if __name__ == "__main__":
    main()
