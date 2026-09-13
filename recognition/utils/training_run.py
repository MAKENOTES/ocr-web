"""管理训练运行的参数、续训状态和保存策略，由 train.py 显式传入配置。"""
import argparse
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from recognition.utils.training_checkpoint import (
    atomic_save, file_digest, load_initial_weights, load_training_checkpoint,
    make_checkpoint, restore_rng_state,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="OCR 训练：完整检查点续训或旧权重微调")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", help="完整检查点路径，从已完成轮次的下一轮继续")
    source.add_argument("--weights", help="仅加载模型权重，重新初始化优化器、轮次和调度器")
    parser.add_argument("--weights-keys", help="旧纯权重训练时使用的原始字典；必须与当前字典一致")
    parser.add_argument("--checkpoint-dir", help="本次检查点空目录；默认在训练入口配置的根目录下自动新建")
    args = parser.parse_args(argv)
    if args.weights_keys and not args.weights:
        parser.error("--weights-keys requires --weights")
    return args


def build_training_config(*, train_annotation, valid_annotation, img_height, img_width,
                          num_channels, batch_size, num_workers, start_valid_epoch,
                          start_save_epoch, save_interval, keep_top_k):
    """记录标注内容、关键配置和核心源码全文 SHA-256；注释改动也影响续训，不扫描图片内容。"""
    annotations = [train_annotation] if isinstance(train_annotation, str) else list(train_annotation)
    if valid_annotation and os.path.exists(valid_annotation):
        annotations.append(valid_annotation)
    source_paths = [
        "recognition/components/dataset.py", "recognition/components/converter.py",
        "recognition/utils/annotation.py",
        "recognition/models/model.py",
    ]
    return {
        "annotations": [(str(Path(path).resolve()), file_digest(path)) for path in annotations],
        "validation": str(Path(valid_annotation).resolve()) if valid_annotation and os.path.exists(valid_annotation) else None,
        "implementation": {path: file_digest(Path(__file__).resolve().parents[2] / path) for path in source_paths},
        "img_height": img_height, "img_width": img_width, "num_channels": num_channels,
        "batch_size": batch_size, "num_workers": num_workers,
        "optimizer": "AdamW", "weight_decay": 1e-4, "gradient_clip": 2.0,
        "scheduler": {"mode": "min", "factor": 0.8, "patience": 20, "min_lr": 1e-6},
        "start_valid_epoch": start_valid_epoch, "start_save_epoch": start_save_epoch,
        "save_interval": save_interval, "keep_top_k": keep_top_k,
    }


def prepare_run_directories(args, checkpoint_root, log_root):
    """续训也创建独立输出，避免覆盖原检查点及 TensorBoard 历史。"""
    run_name = f"ocr_exp_{time.time_ns()}"
    checkpoint_dir = Path(args.checkpoint_dir or Path(checkpoint_root) / run_name).resolve()
    log_dir = (Path(log_root) / run_name).resolve()
    for path in (checkpoint_dir, log_dir):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Output directory must be empty: {path}")
    for path in (checkpoint_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, log_dir


class TrainingRun:
    """持有一次运行的恢复状态；模型前向、反向和验证仍由 train.py 负责。"""

    def __init__(self, args, config, keys, checkpoint_root, *, epochs, seed, log_root):
        self.args = args
        self.epochs = epochs
        self.log_root = log_root
        self.config = config
        self.keys = list(keys)
        self.checkpoint_root = checkpoint_root
        self.checkpoint_dir = None
        self.checkpoint = load_training_checkpoint(args.resume, config, self.keys) if args.resume else None
        self.start_epoch = self.checkpoint["epoch"] + 1 if self.checkpoint else 1
        self.global_step = self.checkpoint["global_step"] if self.checkpoint else 0
        self.best_checkpoints = list(self.checkpoint["best_checkpoints"]) if self.checkpoint else []
        self.complete = self.start_epoch > epochs
        if self.complete:
            print(f"检查点已完成 {self.start_epoch - 1} 轮，已达到目标 {epochs}，无需续训。")
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def restore(self, model, optimizer, scheduler):
        """初始化对象后恢复权重与优化器，随机状态留到数据加载器创建后恢复。"""
        if self.checkpoint:
            model.load_state_dict(self.checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(self.checkpoint["optimizer"])
            scheduler.load_state_dict(self.checkpoint["scheduler"])
        elif self.args.weights:
            load_initial_weights(model, self.args.weights, self.keys, self.args.weights_keys)

    def open_writer(self):
        self.checkpoint_dir, log_dir = prepare_run_directories(self.args, self.checkpoint_root, self.log_root)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"检查点目录: {self.checkpoint_dir}")
        print(f"TensorBoard 目录: {log_dir}")
        print(f"准备开始训练 - Epoch {self.start_epoch} 至 {self.epochs}")
        print(f"  - 验证开始: Epoch {self.config['start_valid_epoch']}")
        print(f"  - 每轮保存 last.pth；周期备份 / Top-K 开始: Epoch {self.config['start_save_epoch']}")
        return writer

    def restore_random_state(self, optimizer):
        """在模型、数据加载器及日志初始化后，进入下一轮迭代前调用。"""
        if self.checkpoint:
            restore_rng_state(self.checkpoint["rng_state"])
            print(f"已恢复: epoch={self.start_epoch - 1}, global_step={self.global_step}, lr={optimizer.param_groups[0]['lr']:.8g}")

    def save_epoch(self, model, optimizer, scheduler, epoch, global_step, metric, has_validation):
        """先持久化整轮状态，再清理本次运行的过期 Top-K 文件。"""
        config, keys = self.config, self.keys
        best_checkpoints, checkpoint_dir = self.best_checkpoints, self.checkpoint_dir
        start_save_epoch = config["start_save_epoch"]
        save_interval, keep_top_k = config["save_interval"], config["keep_top_k"]
        best_path = None
        removed = None
        if epoch >= start_save_epoch and has_validation and np.isfinite(metric):
            if len(best_checkpoints) < keep_top_k or metric < best_checkpoints[-1]["metric"]:
                best_path = checkpoint_dir / f"best_cer_{metric:.6f}_epoch_{epoch}.pth"
                best_checkpoints.append({"path": str(best_path), "metric": metric, "epoch": epoch})
                best_checkpoints.sort(key=lambda item: item["metric"])
                if len(best_checkpoints) > keep_top_k:
                    removed = Path(best_checkpoints.pop()["path"])

        payload = make_checkpoint(model, optimizer, scheduler, epoch, global_step, config, keys, best_checkpoints)
        if best_path is not None:
            atomic_save(payload, best_path)
            print(f"  [保存] 最佳模型: {best_path}")
        if epoch >= start_save_epoch and epoch % save_interval == 0:
            atomic_save(payload, checkpoint_dir / f"epoch_{epoch}.pth")
        atomic_save(payload, checkpoint_dir / "last.pth")
        if removed is not None and removed.parent.resolve() == checkpoint_dir.resolve():
            removed.unlink(missing_ok=True)
