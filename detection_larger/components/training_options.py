"""仅处理本次运行的权重和输出路径；训练配置统一在 train.py。"""
import argparse
from pathlib import Path

import torch


def parse_args(argv=None):
    """只保留每次运行可能变化的初始权重和输出目录。"""
    parser = argparse.ArgumentParser(description="沿用原 ResNet18 EAST 流程训练 PDF 和表格文字")
    parser.add_argument("--weights", type=Path, help="初始模型权重；优化器和轮次重新开始")
    parser.add_argument("--output", type=Path, help="本次检查点空目录；默认按时间新建")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"Training output must be empty: {args.output}")
    return args


def load_initial_weights(model, path):
    """支持原纯 state_dict 和带 model 字段的检查点，不恢复旧优化器。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    model.load_state_dict(state, strict=True)
