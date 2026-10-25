"""验证训练 DataLoader 按配置包含或跳过验证集。"""
from recognition import training_data


def test_create_training_loaders_includes_validation(monkeypatch, tmp_path):
    validation = tmp_path / "val.txt"
    validation.write_text("", encoding="utf-8")
    calls = []
    train_sentinel, valid_sentinel = object(), object()

    def fake_dataloader(**kwargs):
        calls.append(kwargs)
        return train_sentinel if kwargs["is_train"] else valid_sentinel

    monkeypatch.setattr(training_data, "TRAIN_ANNOTATION", "train.txt")
    monkeypatch.setattr(training_data, "VALID_ANNOTATION", str(validation))
    monkeypatch.setattr(training_data, "create_dataloader", fake_dataloader)

    train_loader, valid_loader = training_data.create_training_loaders()

    assert train_loader is train_sentinel
    assert valid_loader is valid_sentinel
    assert [call["is_train"] for call in calls] == [True, False]
    assert calls[0]["annotation_path"] == "train.txt"
    assert calls[1]["annotation_path"] == str(validation)


def test_create_training_loaders_skips_missing_validation(monkeypatch, tmp_path):
    calls = []
    train_sentinel = object()

    def fake_dataloader(**kwargs):
        calls.append(kwargs)
        return train_sentinel

    monkeypatch.setattr(training_data, "TRAIN_ANNOTATION", "train.txt")
    monkeypatch.setattr(training_data, "VALID_ANNOTATION", str(tmp_path / "missing.txt"))
    monkeypatch.setattr(training_data, "create_dataloader", fake_dataloader)

    train_loader, valid_loader = training_data.create_training_loaders()

    assert train_loader is train_sentinel
    assert valid_loader is None
    assert len(calls) == 1
