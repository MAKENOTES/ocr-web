import numpy as np
import onnx
import onnxruntime as ort
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


def _make_model() -> CRNN:
    torch.manual_seed(20260902)
    model = CRNN(img_height=32, nc=3, n_class=19)

    # 非零 bias 能稳定复现旧实现中反向 GRU 穿过 padding 后污染有效状态的问题。
    with torch.no_grad():
        for name, parameter in model.rnn.named_parameters():
            if "bias" in name:
                parameter.uniform_(-0.3, 0.3)
    return model


def test_crnn_output_is_invariant_to_mixed_width_batch_padding():
    """短图的有效输出不能因同批出现长图而变化。"""
    model = _make_model()
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


def test_length_aware_crnn_supports_training_backward_pass():
    """PackedSequence 路径必须可用于正式训练和反向传播。"""
    model = _make_model()
    model.train()

    short = torch.rand(3, 32, 64)
    long = torch.rand(3, 32, 128)
    images, _, widths = _collate(short, long)

    output = model(images, img_widths=widths)
    loss = -output[0, :, 1].mean()
    loss.backward()

    assert model.rnn.weight_ih_l0.grad is not None
    assert torch.isfinite(model.rnn.weight_ih_l0.grad).all()


def test_exported_onnx_uses_sequence_lengths_and_is_batch_invariant(tmp_path):
    """ONNX 的 GRU 也必须跳过 padding，不能只修复 PyTorch。"""
    model = _make_model()
    model.eval()
    wrapper = OnnxExportWrapper(model).eval()

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

    graph = onnx.load(str(model_path)).graph
    gru_nodes = [node for node in graph.node if node.op_type == "GRU"]
    assert len(gru_nodes) == 1
    assert len(gru_nodes[0].input) > 4
    assert gru_nodes[0].input[4]  # ONNX GRU 的 sequence_lens 输入

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
    np.testing.assert_allclose(pytorch_mixed, onnx_mixed, rtol=1e-5, atol=1e-6)


def test_openvino_backend_finds_renamed_lengths_input_and_is_batch_invariant(tmp_path):
    model = _make_model()
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
