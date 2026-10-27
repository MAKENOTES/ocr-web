"""验证逐层 BiGRU 导出：ONNX 结构、数值一致性和 OpenVINO 长度修正。"""

import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from recognition.models.model import CRNN
from recognition.runtime.backends import OpenVINOBackend
from recognition_export_onnx_layer import export_onnx

EXPORT_WIDTH = 96  # 导出样例张量宽度，必须不小于 MIN_IMAGE_WIDTH + CNN_RIGHT_CONTEXT


def _write_keys(path, chars: str = "甲乙丙丁戊") -> int:
    """写入测试字典，返回含 blank/UNK 的类别数。"""
    path.write_text("".join(f"{char}\n" for char in chars), encoding="utf-8")
    return len(chars) + 2


def _write_checkpoint(tmp_path, num_layers: int, n_class: int):
    """保存指定层数的小模型权重；GRU bias 取非零值以便暴露 padding 污染。

    使用 CRNN 的默认隐藏维度，与导出入口按默认值构建的模型保持一致。
    """
    torch.manual_seed(20260908)
    model = CRNN(32, 3, n_class, num_layers=num_layers).eval()
    with torch.no_grad():
        for name, parameter in model.rnn.named_parameters():
            if "bias" in name:
                parameter.uniform_(-0.3, 0.3)

    checkpoint = tmp_path / f"model_l{num_layers}.pth"
    torch.save({"state_dict": model.state_dict(), "epoch": 1}, checkpoint)
    return model, checkpoint


@pytest.mark.parametrize("num_layers", [1, 2])
def test_export_is_layer_split_and_matches_pytorch(tmp_path, num_layers):
    keys = tmp_path / "keys.txt"
    n_class = _write_keys(keys)
    model, checkpoint = _write_checkpoint(tmp_path, num_layers, n_class)
    output = tmp_path / f"layer{num_layers}.onnx"

    # 导出内部会比较 PyTorch 与 ONNX，并在结构或数值不符时抛错。
    export_onnx(str(checkpoint), str(output), str(keys),
                img_width=EXPORT_WIDTH, num_threads=1)

    graph = onnx.load(str(output))
    props = {prop.key: prop.value for prop in graph.metadata_props}
    assert props["num_layers"] == str(num_layers)
    assert props["num_channels"] == "3"
    assert props["keys_sha256"] == hashlib.sha256(keys.read_bytes()).hexdigest()

    # 多层必须拆成每层一个 GRU 节点：ONNX GRU 的 W 只描述单层，
    # 且第 2 层起的输入维度是 2*hidden，与首层不同。
    gru_nodes = [node for node in graph.graph.node if node.op_type == "GRU"]
    assert len(gru_nodes) == num_layers
    for node in gru_nodes:
        assert len(node.input) > 4 and node.input[4], "每个 GRU 节点都必须带 sequence_lens"

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    # 覆盖张量宽度除以 4 的全部余数，防止池化按导出样例的奇偶性固化。
    for tensor_width, widths in [(96, [64]), (97, [65]), (98, [66]), (99, [67]),
                                 (129, [48, 64, 97])]:
        images = torch.randn(len(widths), 3, 32, tensor_width)
        img_widths = torch.tensor(widths)
        with torch.inference_mode():
            expected = model(images, img_widths).numpy()
        actual = session.run(
            None, {"input": images.numpy(), "img_widths": img_widths.numpy()},
        )[0]
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        # padding 时间步固定为 blank；多层串联不能破坏尾部约定。
        steps = model._calc_cnn_output_width(widths[0])
        assert actual[steps:, 0].argmax(1).max() == 0


@pytest.mark.parametrize("num_layers", [1, 2])
def test_openvino_backend_keeps_batch_consistency(tmp_path, num_layers):
    keys = tmp_path / "keys.txt"
    n_class = _write_keys(keys)
    model, checkpoint = _write_checkpoint(tmp_path, num_layers, n_class)
    output = tmp_path / f"layer{num_layers}.onnx"
    export_onnx(str(checkpoint), str(output), str(keys),
                img_width=EXPORT_WIDTH, num_threads=1)

    backend = OpenVINOBackend(str(output), num_threads=1)
    assert backend.has_img_widths

    # 混合宽度 batch 在同一画布上补边；前两张的有效宽度是 48，第三张是 97。
    # padding 区域填入随机内容，任一方向泄漏 padding 都会改变输出。
    single = torch.randn(1, 3, 32, 129)
    mixed = torch.randn(3, 3, 32, 129)
    mixed[2] = single[0]
    img_widths = torch.tensor([48, 48, 97])
    with torch.inference_mode():
        expected = model(mixed, img_widths)
        single_expected = model(single, torch.tensor([97]))

    actual = backend.infer(mixed, img_widths)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    # 同一张图片在单图与混合宽度 batch 中的有效输出必须一致；输出为 TBC，按 batch 维取。
    torch.testing.assert_close(actual[:, 2:3], single_expected, rtol=1e-4, atol=1e-5)
