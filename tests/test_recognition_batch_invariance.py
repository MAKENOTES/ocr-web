import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from recognition.components.dataset import (
    CNN_RIGHT_CONTEXT,
    dynamic_padding_collate_fn,
)
from recognition.components.converter import LabelConverter
from recognition.models.model import CRNN
from recognition_export_onnx import OnnxExportWrapper
from recognition_inference import OpenVINOBackend, normalize_output_to_tbc


def _collate(*images: torch.Tensor):
    return dynamic_padding_collate_fn(
        [(image, f"sample-{index}") for index, image in enumerate(images)]
    )


def _make_model(num_layers: int = 1) -> CRNN:
    """构造测试模型；默认单层以匹配 recognition_export_onnx.py 的导出包装器。"""
    torch.manual_seed(20260902)
    model = CRNN(img_height=32, nc=3, n_class=19, num_layers=num_layers)

    # 非零 bias 能稳定复现旧实现中反向 GRU 穿过 padding 后污染有效状态的问题。
    with torch.no_grad():
        for name, parameter in model.rnn.named_parameters():
            if "bias" in name:
                parameter.uniform_(-0.3, 0.3)
    return model


@pytest.mark.parametrize("num_layers", [1, 2])
def test_crnn_output_is_invariant_to_mixed_width_batch_padding(num_layers):
    """短图的有效输出不能因同批出现长图而变化。"""
    model = _make_model(num_layers)
    model.eval()

    short = torch.rand(3, 32, 64)
    long = torch.rand(3, 32, 256)

    single_images, _, single_widths = _collate(short)
    mixed_images, _, mixed_widths = _collate(short, long)

    assert single_images.shape[-1] == 64 + CNN_RIGHT_CONTEXT
    assert mixed_images.shape[-1] == 256 + CNN_RIGHT_CONTEXT
    assert single_widths.tolist() == [64]
    assert mixed_widths.tolist() == [64, 256]

    with torch.no_grad():
        single_output = model(single_images, img_widths=single_widths)[:, 0]
        mixed_output = model(mixed_images, img_widths=mixed_widths)[:, 0]

    valid_steps = model._calc_cnn_output_width(64)
    torch.testing.assert_close(
        single_output[:valid_steps],
        mixed_output[:valid_steps],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("num_layers", [1, 2])
def test_length_aware_crnn_supports_training_backward_pass(num_layers):
    """PackedSequence 路径必须可用于正式训练和反向传播。"""
    model = _make_model(num_layers)
    model.train()

    short = torch.rand(3, 32, 64)
    long = torch.rand(3, 32, 128)
    images, _, widths = _collate(short, long)

    output = model(images, img_widths=widths)
    loss = -output[0, :, 1].mean()
    loss.backward()

    assert model.rnn.weight_ih_l0.grad is not None
    assert torch.isfinite(model.rnn.weight_ih_l0.grad).all()


@pytest.mark.parametrize("output_type", ["log_probs", "logits"])
def test_exported_onnx_uses_sequence_lengths_and_is_batch_invariant(
    tmp_path, capfd, recwarn, output_type,
):
    """ONNX 的 GRU 也必须跳过 padding，不能只修复 PyTorch。"""
    # 旧导出包装器只支持单层；多层导出见 test_recognition_export_onnx_layer.py。
    model = _make_model(num_layers=1)
    model.eval()
    wrapper = OnnxExportWrapper(model, output_type=output_type).eval()
    original_state = {key: value.clone() for key, value in model.state_dict().items()}

    short = torch.rand(3, 32, 64)
    long = torch.rand(3, 32, 256)
    single_images, _, single_widths = _collate(short)
    mixed_images, _, mixed_widths = _collate(short, long)

    model_path = tmp_path / "recognition_batch_invariant.onnx"
    torch.onnx.export(
        wrapper,
        (mixed_images, mixed_widths.long()),
        str(model_path),
        opset_version=14,
        dynamo=False,
        input_names=["input", "img_widths"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "width"},
            "img_widths": {0: "batch_size"},
            "output": {0: "seq_len", 1: "batch_size"},
        },
    )

    # C++ 形状推断警告写入 stderr；Python GRU/trace 警告由 recwarn 捕获。
    export_stderr = capfd.readouterr().err
    assert "PackPadded" not in export_stderr
    assert "PadPacked" not in export_stderr
    assert not any("batch_size other than 1" in str(w.message) for w in recwarn)
    assert not any(issubclass(w.category, torch.jit.TracerWarning) for w in recwarn)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)

    graph = onnx.load(str(model_path)).graph
    gru_nodes = [node for node in graph.node if node.op_type == "GRU"]
    assert len(gru_nodes) == 1
    assert len(gru_nodes[0].input) > 4
    assert gru_nodes[0].input[4]  # ONNX GRU 的 sequence_lens 输入
    assert all(node.domain in ("", "ai.onnx") for node in graph.node)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    single_output = session.run(
        None,
        {
            "input": single_images.numpy(),
            "img_widths": single_widths.numpy().astype(np.int64),
        },
    )[0][:, 0]
    mixed_output = session.run(
        None,
        {
            "input": mixed_images.numpy(),
            "img_widths": mixed_widths.numpy().astype(np.int64),
        },
    )[0][:, 0]

    valid_steps = model._calc_cnn_output_width(64)
    np.testing.assert_allclose(
        single_output[:valid_steps],
        mixed_output[:valid_steps],
        rtol=1e-5,
        atol=1e-6,
    )

    with torch.no_grad():
        pytorch_mixed = model(mixed_images, img_widths=mixed_widths).numpy()
    onnx_mixed = session.run(
        None,
        {
            "input": mixed_images.numpy(),
            "img_widths": mixed_widths.numpy().astype(np.int64),
        },
    )[0]
    if output_type == "logits":
        onnx_mixed = torch.from_numpy(onnx_mixed).log_softmax(2).numpy()
    np.testing.assert_allclose(pytorch_mixed, onnx_mixed, rtol=1e-5, atol=1e-6)

    # 改变 batch、奇数宽度、长度顺序及重复长度，检查动态 h0 和门权重转换。
    for valid_widths in ([13], [97, 13, 53], [17, 161, 64, 17]):
        images, _, widths = _collate(*[torch.rand(3, 32, width) for width in valid_widths])
        actual = session.run(None, {
            "input": images.numpy(), "img_widths": widths.numpy().astype(np.int64),
        })[0]
        if output_type == "logits":
            actual = torch.from_numpy(actual).log_softmax(2).numpy()
        with torch.no_grad():
            expected = model(images, img_widths=widths).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_openvino_backend_finds_renamed_lengths_input_and_is_batch_invariant(tmp_path):
    model = _make_model(num_layers=1)
    model.eval()
    wrapper = OnnxExportWrapper(model).eval()

    short = torch.rand(3, 32, 64)
    long = torch.rand(3, 32, 256)
    single_images, _, single_widths = _collate(short)
    mixed_images, _, mixed_widths = _collate(short, long)

    model_path = tmp_path / "recognition_openvino.onnx"
    torch.onnx.export(
        wrapper,
        (mixed_images, mixed_widths.long()),
        str(model_path),
        opset_version=14,
        dynamo=False,
        input_names=["input", "img_widths"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "width"},
            "img_widths": {0: "batch_size"},
            "output": {0: "seq_len", 1: "batch_size"},
        },
    )

    backend = OpenVINOBackend(str(model_path))
    assert backend.has_img_widths

    single_output = backend.infer(single_images, single_widths)[:, 0]
    mixed_output = backend.infer(mixed_images, mixed_widths)[:, 0]
    valid_steps = model._calc_cnn_output_width(64)
    # OpenVINO 的图融合会带来小量浮点差异；门禁同时限制数值漂移并要求
    # CTC argmax token 完全一致。
    torch.testing.assert_close(
        single_output[:valid_steps],
        mixed_output[:valid_steps],
        rtol=2e-3,
        atol=1e-2,
    )
    assert torch.equal(
        single_output[:valid_steps].argmax(dim=1),
        mixed_output[:valid_steps].argmax(dim=1),
    )
    # 跨 batch 一致不代表正确：两个结果都可能错误地遍历右侧 padding。
    # 必须同时对齐 PyTorch 的有效输出，才能发现 OpenVINO 反向 GRU 长度问题。
    with torch.inference_mode():
        reference = model(mixed_images, mixed_widths)
    actual = backend.infer(mixed_images, mixed_widths)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


def test_output_layout_normalization_uses_known_batch_dimension():
    # T < B 时不能再用维度大小关系猜测 layout。
    tbc = np.arange(4 * 8 * 3).reshape(4, 8, 3)
    btc = np.transpose(tbc, (1, 0, 2))

    np.testing.assert_array_equal(normalize_output_to_tbc(tbc, batch_size=8), tbc)
    np.testing.assert_array_equal(normalize_output_to_tbc(btc, batch_size=8), tbc)


def test_ctc_decode_uses_explicit_layout_when_time_is_shorter_than_batch(tmp_path):
    keys_path = tmp_path / "keys.txt"
    keys_path.write_text("a\nb\n", encoding="utf-8")
    converter = LabelConverter(str(keys_path))

    torch.manual_seed(3)
    tbc = torch.randn(3, 8, converter.num_classes)
    btc = tbc.transpose(0, 1)

    assert converter.decode(tbc, layout="TBC") == converter.decode(
        btc,
        layout="BTC",
    )
