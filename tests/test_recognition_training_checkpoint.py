"""验证整轮续训与连续训练等价，以及不兼容权重的明确拒绝。"""
import random

import numpy as np
import pytest
import torch
from torch import nn

from recognition import train as training
from recognition.utils import training_checkpoint as checkpoints
from recognition.utils import training_run
from recognition.utils.checkpoint import extract_state_dict


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
    monkeypatch.setattr(training, "TRAIN_ANNOTATION", str(annotation))
    monkeypatch.setattr(training, "VALID_ANNOTATION", str(annotation))
    monkeypatch.setattr(training, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(training, "CRNN", TinyRecognizer)
    monkeypatch.setattr(training, "create_dataloader", lambda **kwargs: TinyLoader())
    # 固定指标让调度器在第 22 轮降学习率，验证其等待计数也被恢复。
    monkeypatch.setattr(training, "evaluate", lambda *args: (0.5, 0.5, 1.0))
    return tmp_path


def run_training(root, name, epochs, resume=None):
    argv = ["--epochs", str(epochs), "--checkpoint-dir", str(root / name),
            "--log-dir", str(root / f"{name}_logs")]
    if resume is not None:
        argv += ["--resume", str(resume)]
    training.train(argv)
    return root / name / "last.pth"


def test_resume_matches_continuous_training(tiny_training):
    root = tiny_training
    uninterrupted = run_training(root, "continuous", 23)
    first = run_training(root, "first", 10)
    resumed = run_training(root, "resumed", 23, first)
    expected = torch.load(uninterrupted, weights_only=True)
    actual = checkpoints.load_training_checkpoint(resumed, expected["config"], ["甲", "乙"])
    assert actual["epoch"] == 23
    assert actual["global_step"] == 46
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert actual["optimizer"]["param_groups"][0]["lr"] == pytest.approx(training.LEARNING_RATE * 0.8)
    for key, value in extract_state_dict(expected).items():
        torch.testing.assert_close(extract_state_dict(actual)[key], value, rtol=0, atol=0)
    for parameter, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual["optimizer"]["state"][parameter][key], value, rtol=0, atol=0)
    assert torch.load(first, weights_only=True)["epoch"] == 10


def test_last_saved_before_best_stage_and_contract_checks(tiny_training):
    root = tiny_training
    path = run_training(root, "first", 1)
    assert list(path.parent.glob("*.pth")) == [path]
    config = torch.load(path, weights_only=True)["config"]
    with pytest.raises(ValueError, match="Dictionary content/order"):
        checkpoints.load_training_checkpoint(path, config, ["乙", "甲"])
    annotation = root / "train.txt"
    annotation.write_text('image.png, "乙"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="annotations"):
        run_training(root, "changed_data", 2, path)


def test_legacy_weights_need_matching_dictionary(tmp_path):
    model = nn.Linear(2, 3)
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="Not a complete training checkpoint"):
        checkpoints.load_training_checkpoint(path, {}, ["甲", "乙"])
    with pytest.raises(ValueError, match="require --weights-keys"):
        checkpoints.load_initial_weights(model, path, ["甲", "乙"])
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n乙\n", encoding="utf-8")
    target = nn.Linear(2, 3)
    checkpoints.load_initial_weights(target, path, ["甲", "乙"], keys)
    torch.testing.assert_close(target.weight, model.weight)
    with pytest.raises(ValueError, match="Dictionary content/order"):
        checkpoints.load_initial_weights(target, path, ["乙", "甲"], keys)
    with pytest.raises(ValueError, match="Model shape differs"):
        checkpoints.load_initial_weights(nn.Linear(2, 4), path, ["甲", "乙"], keys)


def test_atomic_save_failure_preserves_previous_file(monkeypatch, tmp_path):
    path = tmp_path / "last.pth"
    checkpoints.atomic_save({"epoch": 1}, path)

    def fail_save(payload, handle):
        handle.write(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(checkpoints.torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        checkpoints.atomic_save({"epoch": 2}, path)
    assert torch.load(path, weights_only=True) == {"epoch": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_top_k_resume_preserves_source_files(tiny_training, monkeypatch):
    monkeypatch.setattr(training, "START_SAVE_EPOCH", 1)
    monkeypatch.setattr(training, "KEEP_TOP_K", 1)
    root = tiny_training
    source = run_training(root, "source", 1)
    old_best = next(source.parent.glob("best_*.pth"))
    monkeypatch.setattr(training, "evaluate", lambda *args: (0.9, 0.1, 0.5))
    path = run_training(root, "resumed", 2, source)
    payload = torch.load(path, weights_only=True)
    assert payload["best_checkpoints"][0]["epoch"] == 2
    assert old_best.exists()
    best_path = next(path.parent.glob("best_*.pth"))
    assert torch.load(best_path, weights_only=True)["epoch"] == 2
    target = TinyRecognizer(32, 3, 4)
    checkpoints.load_initial_weights(target, best_path, ["甲", "乙"])


@pytest.mark.parametrize("argv", [
    ["--resume", "a", "--weights", "b"], ["--resume", "a", "--lr", "0.001"],
    ["--weights-keys", "keys.txt"], ["--epochs", "0"], ["--lr", "-1"], ["--lr", "nan"],
])
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        training_run.parse_args(argv, epochs=680, learning_rate=3e-4, checkpoint_dir="unused")


def test_output_directory_rejects_existing_files(tmp_path):
    path = tmp_path / "last.pth"
    path.write_text("keep")
    args = training_run.parse_args(["--checkpoint-dir", str(tmp_path)],
                                   epochs=680, learning_rate=3e-4, checkpoint_dir="unused")
    with pytest.raises(ValueError, match="Output directory must be empty"):
        training_run.prepare_run_directories(args, "unused")
    assert path.read_text() == "keep"


def test_completed_target_does_not_create_new_run(tiny_training):
    path = run_training(tiny_training, "first", 1)
    run_training(tiny_training, "noop", 1, path)
    assert not (tiny_training / "noop").exists()
    assert not (tiny_training / "noop_logs").exists()


def test_periodic_checkpoint_supports_resume(tiny_training, monkeypatch):
    monkeypatch.setattr(training, "START_SAVE_EPOCH", 1)
    root = tiny_training
    run_training(root, "first", 5)
    path = run_training(root, "resumed", 6, root / "first" / "epoch_5.pth")
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["epoch"] == 6
    assert checkpoint["global_step"] == 12


def test_weights_start_a_new_training_schedule(tiny_training):
    root = tiny_training
    source = run_training(root, "first", 2)
    training.train([
        "--weights", str(source), "--epochs", "1", "--lr", "0.001",
        "--checkpoint-dir", str(root / "fresh"), "--log-dir", str(root / "fresh_logs"),
    ])
    checkpoint = torch.load(root / "fresh" / "last.pth", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 2
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 0.001


def test_defaults_follow_training_script_constants(tiny_training, monkeypatch):
    root = tiny_training
    monkeypatch.setattr(training, "EPOCHS", 1)
    monkeypatch.setattr(training, "LEARNING_RATE", 0.002)
    monkeypatch.setattr(training, "CHECKPOINT_DIR", str(root / "checkpoints"))
    training.train(["--log-dir", str(root / "logs")])
    path, = (root / "checkpoints").glob("ocr_exp_*/last.pth")
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 0.002
