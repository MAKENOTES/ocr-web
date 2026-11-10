"""生成统一检测训练运行的静态 HTML 分析报告。

用法：
    .venv/bin/python detection/report.py --run detection/checkpoints/<run>

默认输出到 detection/reports/<run>/index.html；报告自包含图表和问题图片，
可直接用浏览器打开，不需要启动 TensorBoard。
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.components.training_report import main


if __name__ == "__main__":
    main()
