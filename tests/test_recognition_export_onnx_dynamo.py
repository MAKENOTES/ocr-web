import hashlib

import numpy as np
import onnx
import pytest
import torch

from recognition.models.model import CRNN
from recognition_export_onnx_dynamo import DynamoExportWrapper, create_session, export_model
from recognition_inference import OpenVINOBackend


@pytest.mark.parametrize("channels,output_type,trace_width", [(3, "log_probs", 256), (1, "logits", 257)])
def test_dynamo_export_preserves_dynamic_shapes_and_source_training(
    tmp_path, recwarn, channels, output_type, trace_width,
):
    torch.manual_seed(20260909)
    # dynamo 导出包装器只支持单层 BiGRU；多层模型由 recognition_export_onnx_layer.py 导出。
    model = CRNN(32, channels, 7, nh=32, num_layers=1).train()
    with torch.no_grad():
        for name, parameter in model.rnn.named_parameters():
            if "bias" in name:
                parameter.uniform_(-0.3, 0.3)
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n乙\n丙\n丁\n戊\n", encoding="utf-8")
    path = tmp_path / "dynamo.onnx"
    assert export_model(model, path, keys, trace_width, output_type) < 1e-3
    assert model.training and model.cnn.stem[1].training and model.rnn.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
    assert not any("legacy TorchScript" in str(w.message) for w in recwarn)
    assert not any("axis name" in str(w.message) for w in recwarn)

    graph = onnx.load(path)
    assert {p.key: p.value for p in graph.metadata_props}["exporter"] == "torch.export"
    assert {p.key: p.value for p in graph.metadata_props}["keys_sha256"] == hashlib.sha256(keys.read_bytes()).hexdigest()
    assert not path.with_suffix(".onnx.data").exists()
    assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    assert graph.graph.input[0].type.tensor_type.shape.dim[3].dim_param

    session = create_session(path, 1)
    reference = DynamoExportWrapper(model, output_type)
    # 覆盖输入宽度除以 4 的全部余数；防止池化按 trace 的奇偶性固化。
    for tensor_width, widths in [(80, [48]), (81, [49]), (82, [50]), (83, [51]),
                                  (225, [193, 48, 97, 48])]:
        images = torch.rand(len(widths), channels, 32, tensor_width)
        img_widths = torch.tensor(widths)
        with torch.inference_mode():
            expected = reference(images, img_widths).numpy()
        actual = session.run(None, {"input": images.numpy(), "img_widths": img_widths.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    # 新图须能被现有 OpenVINO 后端读取，包括反向 GRU 的长度修正。
    backend = OpenVINOBackend(str(path))
    actual = backend.infer(images, img_widths)
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-4, atol=1e-5)

    # 导出之后源模型仍可训练，双向 GRU 的所有参数均能收到有限梯度。
    model(images, img_widths)[0, :, 1].neg().mean().backward()
    for parameter in model.rnn.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_failed_dynamo_validation_preserves_existing_output(tmp_path, monkeypatch):
    model = CRNN(32, 3, 7, nh=32, num_layers=1).eval()
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n乙\n丙\n丁\n戊\n", encoding="utf-8")
    output = tmp_path / "existing.onnx"
    output.write_bytes(b"previous validated model")

    def fail_validation(*args):
        raise RuntimeError("numerical validation failed")

    monkeypatch.setattr("recognition_export_onnx_dynamo.validate_outputs", fail_validation)
    with pytest.raises(RuntimeError, match="numerical validation failed"):
        export_model(model, output, keys)
    assert output.read_bytes() == b"previous validated model"
