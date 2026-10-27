"""验证部署产物只含运行白名单、哈希可核对且能在隔离目录运行。"""

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_recognition_bundle import build_bundle, verify_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundle_layout_manifest_and_overwrite_guard(tmp_path, tiny_recognition_model):
    output = tmp_path / "bundle"

    manifest = build_bundle(
        tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
        output, verify=False,
    )

    # 运行代码、模型和字典必须齐全，训练数据与合成工具不能进入产物。
    assert (output / "recognition" / "runtime" / "engine.py").is_file()
    assert (output / "recognition" / "components" / "preprocessing.py").is_file()
    assert (output / "recognition" / "models" / "model.py").is_file()
    assert (output / manifest["model"]["file"]).is_file()
    assert (output / "keys.txt").is_file()
    assert (output / "example.py").is_file()
    assert not (output / "recognition" / "data").exists()
    assert not (output / "recognition" / "checkpoints").exists()
    assert not (output / "recognition" / "utils" / "synthesis").exists()

    # manifest 必须与磁盘内容一致，部署方可以据此核对模型和字典。
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert _sha256(output / manifest["model"]["file"]) == manifest["model"]["sha256"]
    assert _sha256(output / "keys.txt") == manifest["keys"]["sha256"]
    assert manifest["model"]["num_layers"] == "2"
    assert manifest["model"]["num_channels"] == "3"
    assert manifest["keys"]["num_classes"] == 7

    for name in (
        "requirements.txt", "requirements-onnx.txt",
        "requirements-openvino.txt", "requirements-pytorch.txt",
    ):
        assert (output / name).is_file()
    assert "torch" in (output / "requirements.txt").read_text(encoding="utf-8")
    assert "onnxruntime" in (output / "requirements-onnx.txt").read_text(
        encoding="utf-8"
    )
    assert "-r requirements.txt" in (output / "requirements-onnx.txt").read_text(
        encoding="utf-8"
    )

    # 未加 --force 不覆盖；加了 --force 只能覆盖上一次构建产物。
    with pytest.raises(FileExistsError, match="输出目录已存在"):
        build_bundle(tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
                     output, verify=False)
    rebuilt = build_bundle(
        tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
        output, force=True, verify=False,
    )
    assert rebuilt["model"]["sha256"] == manifest["model"]["sha256"]

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("不要删除", encoding="utf-8")
    with pytest.raises(FileExistsError, match="拒绝覆盖非构建产物目录"):
        build_bundle(tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
                     foreign, force=True, verify=False)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "不要删除"

    # 空目录可以安全复用，构建后成为正常产物。
    empty = tmp_path / "empty"
    empty.mkdir()
    build_bundle(tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
                 empty, force=True, verify=False)
    assert (empty / "manifest.json").is_file()

    not_a_directory = tmp_path / "not_a_directory"
    not_a_directory.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="输出路径不是目录"):
        build_bundle(tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
                     not_a_directory, force=True, verify=False)


def test_bundle_runs_without_repository_on_pythonpath(tmp_path, tiny_recognition_model):
    output = tmp_path / "bundle"
    build_bundle(
        tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
        output, verify=False,
    )

    # 自检脚本会确认 recognition 来自 bundle，并在临时工作目录完成一次推理。
    assert isinstance(verify_bundle(output), str)


def test_bundle_copies_openvino_bin_companion(tmp_path, tiny_recognition_model):
    # OpenVINO IR 由 .xml 和同名 .bin 组成，部署时不能只拷 .xml。
    model = tmp_path / "model.xml"
    model.write_bytes(b"<net/>")
    (tmp_path / "model.bin").write_bytes(b"weights")
    output = tmp_path / "bundle"

    build_bundle(model, tiny_recognition_model["keys"], output, verify=False)

    assert (output / "models" / "model.bin").read_bytes() == b"weights"
