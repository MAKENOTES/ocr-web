"""验证 recognition.runtime 的公开入口、顺序保持和字典校验。"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from recognition.components.converter import LabelConverter
from recognition.components.dataset import InferenceDataset, dynamic_padding_collate_fn
from recognition.runtime import RecognitionEngine, collect_image_paths
from recognition.runtime.backends import (
    OpenVINOBackend,
    load_backend,
    normalize_output_to_tbc,
)

# 自包含运行包目录；下面的测试会扫描它，或把它整体拷贝到临时项目后导入。
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "recognition" / "runtime"


def _write_images(directory: Path) -> list[Path]:
    """写入三张不同宽度的图片，用于覆盖排序与动态补边。"""
    paths = []
    for index, width in enumerate((64, 240, 96)):
        path = directory / f"image_{index}.png"
        image = Image.new("RGB", (width, 32), (250, 248, 242))
        image.putpixel((width // 2, 16), (20, 20, 20))
        image.save(path)
        paths.append(path)
    return paths


def _expected_texts(model, keys_path: Path, paths: list[Path]) -> list[str]:
    """按单图共享预处理 + PyTorch 前向计算期望文本。"""
    converter = LabelConverter(str(keys_path))
    dataset = InferenceDataset([str(path) for path in paths], img_height=32, num_channels=3)
    texts = []
    for index in range(len(dataset)):
        images, _, widths = dynamic_padding_collate_fn([dataset[index]])
        with torch.inference_mode():
            preds = model(images, img_widths=widths)
        texts.append(converter.decode(preds, layout="TBC")[0])
    return texts


def test_pytorch_backend_matches_shared_pipeline(tiny_recognition_model, tmp_path):
    images = _write_images(tmp_path)
    engine = RecognitionEngine(
        tiny_recognition_model["checkpoint"], tiny_recognition_model["keys"],
        backend="pytorch", num_threads=1,
    )

    report = engine.recognize(images, batch_size=1)

    assert [result.text for result in report] == _expected_texts(
        tiny_recognition_model["model"], tiny_recognition_model["keys"], images
    )


def test_onnx_backend_matches_pytorch_and_keeps_input_order(tiny_recognition_model, tmp_path):
    images = _write_images(tmp_path)
    engine = RecognitionEngine(
        tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
        backend="onnx", num_threads=1,
    )

    report = engine.recognize(images, batch_size=2)

    assert [Path(result.path).name for result in report] == [path.name for path in images]
    assert [result.text for result in report] == _expected_texts(
        tiny_recognition_model["model"], tiny_recognition_model["keys"], images
    )


def test_sort_by_width_changes_only_batching_order(tiny_recognition_model, tmp_path):
    images = _write_images(tmp_path)
    engine = RecognitionEngine(
        tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
        backend="onnx", num_threads=1,
    )

    unordered = engine.recognize(images, batch_size=1)
    sorted_report = engine.recognize(images, batch_size=2, sort_by_width=True)

    assert [Path(result.path).name for result in sorted_report] == [path.name for path in images]
    assert [result.text for result in sorted_report] == [result.text for result in unordered]


def test_batch_size_does_not_change_text(tiny_recognition_model, tmp_path):
    images = _write_images(tmp_path)
    engine = RecognitionEngine(
        tiny_recognition_model["onnx"], tiny_recognition_model["keys"],
        backend="onnx", num_threads=1,
    )

    single = engine.recognize(images, batch_size=1)
    packed = engine.recognize(images, batch_size=3)

    assert [result.text for result in packed] == [result.text for result in single]


def test_keys_hash_mismatch_is_rejected(tiny_recognition_model, tmp_path):
    other_keys = tmp_path / "other_keys.txt"
    other_keys.write_text("天地玄黄\n", encoding="utf-8")

    with pytest.raises(ValueError, match="字典与导出模型不一致"):
        RecognitionEngine(
            tiny_recognition_model["onnx"], other_keys, backend="onnx", num_threads=1
        )


def test_missing_model_or_keys_reports_concrete_path(tiny_recognition_model, tmp_path):
    with pytest.raises(FileNotFoundError, match="模型文件不存在"):
        RecognitionEngine(tmp_path / "missing.onnx", tiny_recognition_model["keys"])
    with pytest.raises(FileNotFoundError, match="字典文件不存在"):
        RecognitionEngine(tiny_recognition_model["onnx"], tmp_path / "missing.txt")


def test_empty_image_list_is_rejected(tiny_recognition_model):
    engine = RecognitionEngine(
        tiny_recognition_model["checkpoint"], tiny_recognition_model["keys"],
        backend="pytorch", num_threads=1,
    )

    with pytest.raises(ValueError, match="image_paths 不能为空"):
        engine.recognize([])


def test_collect_image_paths_handles_directory_and_file(tmp_path):
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (16, 32)).save(tmp_path / "a.png")
    Image.new("RGB", (16, 32)).save(tmp_path / "B.JPG")
    Image.new("RGB", (16, 32)).save(tmp_path / "sub" / "nested.png")
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")

    paths = collect_image_paths(tmp_path)
    assert [Path(path).name for path in paths] == ["B.JPG", "a.png"]
    assert collect_image_paths(tmp_path / "a.png") == [str(tmp_path / "a.png")]

    with pytest.raises(FileNotFoundError, match="未找到图像文件"):
        collect_image_paths(tmp_path / "missing")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="目录中没有受支持的图片"):
        collect_image_paths(empty_dir)


def test_normalize_output_to_tbc_accepts_both_layouts():
    tbc = np.zeros((5, 2, 7), dtype=np.float32)
    btc = np.transpose(tbc, (1, 0, 2))

    assert normalize_output_to_tbc(tbc, batch_size=2) is tbc
    np.testing.assert_array_equal(normalize_output_to_tbc(btc, batch_size=2), tbc)

    with pytest.raises(ValueError, match="无法从输出 shape"):
        normalize_output_to_tbc(np.zeros((5, 3, 7), dtype=np.float32), batch_size=2)


def test_legacy_entry_points_keep_cli_defaults():
    import recognition_inference
    import recognition_inference_layer

    # recognition_inference 仍被 benchmark/量化直接导入，必须继续转出后端符号。
    assert recognition_inference.load_backend is load_backend
    assert recognition_inference.OpenVINOBackend is OpenVINOBackend

    # layer 入口只承诺 CLI 兼容，后端类统一从 runtime 导入。
    assert recognition_inference.DEFAULT_MODEL_PATH == "best_recognition.onnx"
    assert recognition_inference_layer.DEFAULT_MODEL_PATH == "best_recognition_layer.onnx"
    assert callable(recognition_inference_layer.main)


def test_cli_writes_results_in_input_order(tiny_recognition_model, tmp_path):
    from recognition.runtime.cli import main

    images = _write_images(tmp_path)
    output_file = tmp_path / "result.txt"
    main([
        "--model_path", str(tiny_recognition_model["onnx"]),
        "--keys_path", str(tiny_recognition_model["keys"]),
        "--image_path", str(tmp_path),
        "--output_file", str(output_file),
        "--backend", "onnx",
        "--batch_size", "2",
        "--sort_by_width",
    ])

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[0] for line in lines] == [path.name for path in images]


def test_cli_requires_explicit_paths_when_no_defaults():
    from recognition.runtime.cli import main

    with pytest.raises(SystemExit):
        main(["--model_path", "model.onnx"])


def test_inference_dataset_applies_explicit_source_augment(tmp_path):
    """鲁棒性评估会传入确定性扰动，InferenceDataset 必须保留该参数。"""
    from recognition.runtime.dataset import InferenceDataset as RuntimeInferenceDataset

    image_path = tmp_path / "source.png"
    Image.new("RGB", (120, 32), (240, 238, 232)).save(image_path)

    def fill_black(image):
        return Image.new("RGB", image.size, (0, 0, 0))

    plain = RuntimeInferenceDataset([image_path], 32, 3)[0][0]
    perturbed = RuntimeInferenceDataset(
        [image_path], 32, 3, source_augment=fill_black
    )[0][0]

    assert not torch.equal(plain, perturbed)


def test_runtime_folder_has_no_repository_imports():
    """runtime 只能依赖第三方库和自身模块，不能引用训练或仓库其他代码。"""
    offenders = []
    for path in sorted(RUNTIME_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    (path.name, alias.name)
                    for alias in node.names
                    if alias.name == "recognition" or alias.name.startswith("recognition.")
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and (
                    module == "recognition" or module.startswith("recognition.")
                ):
                    offenders.append((path.name, module))

    assert offenders == []


def test_runtime_folder_can_be_copied_and_renamed(tiny_recognition_model, tmp_path):
    """只拷贝 runtime 目录并改名后仍能导入和推理，证明可嵌入其他项目。"""
    standalone = tmp_path / "standalone"
    package = standalone / "ocr_runtime"
    shutil.copytree(RUNTIME_DIR, package, ignore=shutil.ignore_patterns("__pycache__"))

    image_path = tmp_path / "check.png"
    Image.new("RGB", (160, 32), (245, 243, 238)).save(image_path)

    script = (
        "import sys\n"
        "from ocr_runtime import RecognitionEngine\n"
        "engine = RecognitionEngine(sys.argv[1], sys.argv[2], backend='onnx')\n"
        "report = engine.recognize([sys.argv[3]], batch_size=1)\n"
        "print(len(report))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(standalone)
    completed = subprocess.run(
        [
            sys.executable, "-c", script,
            str(tiny_recognition_model["onnx"]),
            str(tiny_recognition_model["keys"]),
            str(image_path),
        ],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "1"
