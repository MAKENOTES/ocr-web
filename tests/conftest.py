"""识别 runtime 与部署产物测试共享的小模型。

单独导出一次并复用，避免每个测试都重复导出 ONNX；模型结构、字典和导出参数
与 tests/test_recognition_export_onnx_layer.py 保持同一套契约。
"""

import pytest

# 测试字典字符；总类别数 = 字符数 + blank/UNK。
KEYS_TEXT = "甲乙丙丁戊"
# 测试模型层数；覆盖多层 GRU 的逐层导出路径。
TEST_NUM_LAYERS = 2
# 导出样例张量宽度（像素）；需不小于模型最小宽度与右侧保护区之和。
EXPORT_WIDTH = 96


@pytest.fixture(scope="session")
def tiny_recognition_model(tmp_path_factory):
    """返回小型 CRNN 的 PyTorch 权重、导出的 ONNX 和配套字典。"""
    # 只在请求该 fixture 时导入 torch/onnx，避免拖慢或影响检测等无关测试。
    import torch

    from recognition.models.model import CRNN
    from recognition_export_onnx_layer import export_onnx

    directory = tmp_path_factory.mktemp("recognition_runtime_fixture")
    keys_path = directory / "keys.txt"
    keys_path.write_text(
        "".join(f"{char}\n" for char in KEYS_TEXT), encoding="utf-8"
    )

    torch.manual_seed(20261027)
    model = CRNN(32, 3, len(KEYS_TEXT) + 2, num_layers=TEST_NUM_LAYERS).eval()
    checkpoint_path = directory / "model.pth"
    torch.save({"state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)

    onnx_path = directory / "model.onnx"
    export_onnx(
        str(checkpoint_path), str(onnx_path), str(keys_path),
        img_width=EXPORT_WIDTH, num_threads=1,
    )
    return {
        "model": model,
        "checkpoint": checkpoint_path,
        "onnx": onnx_path,
        "keys": keys_path,
    }
