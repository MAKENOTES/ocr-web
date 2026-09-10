"""保存整轮训练状态；纯权重初始化与完整续训使用不同入口。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from recognition.utils.checkpoint import extract_state_dict

FORMAT_VERSION = 1


def file_digest(path):
    """记录实际文件内容，缺文件时直接报告路径。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_keys(path):
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def capture_rng_state():
    """将 NumPy 状态转换为基础类型，使完整文件也支持 weights_only=True。"""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, keys.tolist(), position, has_gauss, cached_gaussian],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    cuda = state["cuda"]
    if cuda and len(cuda) != torch.cuda.device_count():
        raise ValueError("CUDA device count differs from checkpoint; use --weights for a new run")
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def atomic_save(payload, destination):
    """先写同目录临时文件，再替换目标，避免中断破坏上一轮 last.pth。"""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_training_checkpoint(path, config, keys):
    """先验证恢复契约，再交给训练入口加载参数和优化器。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "state_dict", "optimizer", "scheduler", "epoch", "global_step",
                "rng_state", "config", "keys", "best_checkpoints"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError(f"Not a complete training checkpoint: {path}; use --weights with --weights-keys for legacy weights")
    if checkpoint["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported training checkpoint version: {path}")
    if checkpoint["keys"] != keys:
        raise ValueError(f"Dictionary content/order differs from checkpoint: {path}")
    differences = sorted(key for key in config.keys() | checkpoint["config"].keys()
                         if config.get(key) != checkpoint["config"].get(key))
    if differences:
        raise ValueError(f"Resume configuration differs ({', '.join(differences)}): {path}; use --weights for a new run")
    if checkpoint["epoch"] < 0 or checkpoint["global_step"] < 0:
        raise ValueError(f"Invalid epoch/global_step in checkpoint: {path}")
    return checkpoint


def load_initial_weights(model, path, keys, source_keys_path=None):
    """仅加载权重，不迁移字符类别；旧纯权重须显式提供其原字典。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved_keys = checkpoint.get("keys") if isinstance(checkpoint, dict) else None
    if saved_keys is None:
        if source_keys_path is None:
            raise ValueError(f"Legacy weights require --weights-keys pointing to their original dictionary: {path}")
        saved_keys = read_keys(source_keys_path)
    elif source_keys_path is not None and read_keys(source_keys_path) != saved_keys:
        raise ValueError(f"--weights-keys differs from the dictionary stored in: {path}")
    if saved_keys != keys:
        raise ValueError(f"Dictionary content/order differs from weights: {path}; character remapping is not supported")
    try:
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Model shape differs from weights: {path}: {exc}") from exc


def make_checkpoint(model, optimizer, scheduler, epoch, global_step, config, keys, best_checkpoints):
    return {
        "format_version": FORMAT_VERSION, "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "global_step": global_step, "config": config, "keys": list(keys),
        "best_checkpoints": list(best_checkpoints), "rng_state": capture_rng_state(),
    }
