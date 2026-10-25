"""验证训练检查点保存与 Top-K 维护。"""
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from recognition import train as training
from recognition import training_data


class TinyRecognizer(nn.Module):
    def __init__(self, img_height, nc, n_class):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(3, n_class)

    def forward(self, images, img_widths):
        return self.classifier(self.dropout(images)).log_softmax(-1)

    def _calc_cnn_output_widths(self, widths):
        return widths


class TinyLoader:
    def __iter__(self):
        for _ in range(2):
            images = torch.randn(4, 2, 3) + random.random() + np.random.random()
            yield images, ["甲", "乙"], torch.tensor([4, 4])


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n乙\n", encoding="utf-8")
    annotation = tmp_path / "train.txt"
    annotation.write_text('image.png, "甲"\n', encoding="utf-8")
    monkeypatch.setattr(training, "KEYS_PATH", str(keys))
    monkeypatch.setattr(training_data, "TRAIN_ANNOTATION", str(annotation))
    monkeypatch.setattr(training_data, "VALID_ANNOTATION", str(annotation))
    monkeypatch.setattr(training, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(training, "CRNN", TinyRecognizer)
    monkeypatch.setattr(training_data, "create_dataloader", lambda **kwargs: TinyLoader())
    # 固定验证指标，避免依赖真实数据集内容。
    monkeypatch.setattr(training, "evaluate", lambda *args: (0.5, 0.5, 1.0))
    return tmp_path


def run_training(root, name, epochs):
    checkpoint_root = root / name
    with patch.object(training, "EPOCHS", epochs), \
            patch.object(training, "CHECKPOINT_DIR", str(checkpoint_root)), \
            patch.object(training, "LOG_DIR", str(root / f"{name}_logs")):
        training.train()
    path, = checkpoint_root.glob("ocr_exp_*/last.pth")
    return path


def test_last_saved_before_best_stage(tiny_training):
    path = run_training(tiny_training, "first", 1)
    assert list(path.parent.glob("*.pth")) == [path]
    assert torch.load(path, weights_only=True)["epoch"] == 1


def test_atomic_save_failure_preserves_previous_file(monkeypatch, tmp_path):
    path = tmp_path / "last.pth"
    training.atomic_save({"epoch": 1}, path)

    def fail_save(payload, handle):
        handle.write(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(training.torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        training.atomic_save({"epoch": 2}, path)
    assert torch.load(path, weights_only=True) == {"epoch": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_top_k_prunes_previous_best(tiny_training, monkeypatch):
    monkeypatch.setattr(training, "START_SAVE_EPOCH", 1)
    monkeypatch.setattr(training, "KEEP_TOP_K", 1)
    metrics = iter([(0.5, 0.5, 1.0), (0.9, 0.1, 0.5)])
    monkeypatch.setattr(training, "evaluate", lambda *args: next(metrics))
    path = run_training(tiny_training, "topk", 2)
    best_files = sorted(path.parent.glob("best_*.pth"))
    assert len(best_files) == 1
    assert torch.load(best_files[0], weights_only=True)["epoch"] == 2


def test_periodic_checkpoint_saved(tiny_training, monkeypatch):
    monkeypatch.setattr(training, "START_SAVE_EPOCH", 1)
    path = run_training(tiny_training, "periodic", 5)
    assert torch.load(path, weights_only=True)["epoch"] == 5
    assert (path.parent / "epoch_5.pth").exists()
