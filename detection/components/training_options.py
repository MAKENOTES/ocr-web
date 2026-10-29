"""仅处理本次运行的检查点输出目录；训练配置统一在 train.py。"""
import argparse
from pathlib import Path


def parse_args(argv=None):
    """只保留每次运行可能变化的输出目录。"""
    parser = argparse.ArgumentParser(description="训练统一文字检测模型，覆盖 UI 小图、PDF 页面和表格")
    parser.add_argument("--output", type=Path, help="本次检查点空目录；默认按时间新建")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"Training output must be empty: {args.output}")
    return args
