"""使用 torch.export / dynamo=True 独立导出 CRNN（需要 PyTorch >= 2.9）。

默认输出 best_recognition_dynamo.onnx，不覆盖旧导出器的默认产物。
训练继续使用原有模型；仅导出包装器通过 torch.onnx.ops.symbolic 写入标准 GRU。
"""

import argparse
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from recognition.components.converter import LabelConverter
from recognition.components.dataset import CNN_RIGHT_CONTEXT, dynamic_padding_collate_fn
from recognition.components.preprocessing import MIN_IMAGE_WIDTH
from recognition.models.model import CRNN
from recognition.utils.checkpoint import extract_state_dict, infer_num_channels

MAX_EXPORT_BATCH = 256
MAX_EXPORT_WIDTH = 16384


class DynamoExportWrapper(torch.nn.Module):
    """模型的独立推理快照；不会改变源模型的参数、设备或 train/eval 状态。"""

    def __init__(self, model: CRNN, output_type: str = "log_probs"):
        super().__init__()
        if output_type not in ("log_probs", "logits"):
            raise ValueError(f"未知 output_type: {output_type}")
        self.model = copy.deepcopy(model).cpu().eval()
        self.output_type = output_type
        rnn = self.model.rnn
        if (rnn.num_layers != 1 or not rnn.bidirectional
                or not rnn.bias or rnn.batch_first):
            raise ValueError("新版 ONNX 导出需要单层、带 bias、batch_first=False 的双向 GRU")

        def reorder(weight: torch.Tensor) -> torch.Tensor:
            # PyTorch reset/update/new → ONNX update/reset/new。
            reset, update, new = weight.detach().chunk(3, dim=0)
            return torch.cat((update, reset, new), dim=0)

        directions = [
            [reorder(getattr(rnn, f"{name}_l0{suffix}"))
             for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            for suffix in ("", "_reverse")
        ]
        self.register_buffer("gru_w", torch.stack([d[0] for d in directions]))
        self.register_buffer("gru_r", torch.stack([d[1] for d in directions]))
        self.register_buffer("gru_b", torch.stack([torch.cat(d[2:]) for d in directions]))
        self.eval()

    def forward(self, images: torch.Tensor, img_widths: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            torch._check(img_widths.shape[0] == images.shape[0])
            cnn = self.model.cnn
            conv = cnn.stem(images)
            # PyTorch 2.9 的 MaxPool capture 会按样例宽度奇偶性添加约束。
            # 显式输出标准 MaxPool，仍采用原有 kernel=2/stride=2/floor 语义。
            conv = torch.onnx.ops.symbolic(
                "::MaxPool", (conv,), {"kernel_shape": [2, 2], "strides": [2, 2]},
                dtype=conv.dtype,
                shape=(conv.shape[0], conv.shape[1], conv.shape[2] // 2, conv.shape[3] // 2),
                version=12,
            )
            conv = cnn.block3(cnn.block2(cnn.block1(conv)))
        else:
            conv = self.model.cnn(images)
        conv = conv.squeeze(2).permute(2, 0, 1)
        seq_len, batch, _ = conv.shape
        # 与 MobileFeatureExtractor 的 stride=2 卷积、stride=2 池化一致。
        lengths = ((img_widths.long() + 1) // 2) // 2
        lengths = lengths.clamp(min=1, max=seq_len)
        hidden_size = self.model.rnn.hidden_size
        h0 = conv.new_zeros((2, batch, hidden_size))

        if torch.onnx.is_in_onnx_export():
            # 官方 FX symbolic 接口：标准 ONNX 算子，无自定义 runtime 插件。
            # GRU 的第二个输出 Y_h 不被 CRNN 消费，因此只声明输出 Y。
            output = torch.onnx.ops.symbolic(
                "::GRU", (conv, self.gru_w, self.gru_r, self.gru_b,
                          lengths.to(torch.int32), h0),
                {"hidden_size": hidden_size, "direction": "bidirectional",
                 "linear_before_reset": 1},
                dtype=conv.dtype, shape=(seq_len, 2, batch, hidden_size), version=14,
            )
            # 避免 PyTorch reshape 按 batch=1 与 batch>1 的 stride 特化。
            output = torch.onnx.ops.symbolic(
                "::Transpose", (output,), {"perm": [0, 2, 1, 3]}, dtype=conv.dtype,
                shape=(seq_len, batch, 2, hidden_size), version=13,
            )
            output = torch.onnx.ops.symbolic(
                "::Reshape", (output, torch.tensor([0, 0, -1], device=images.device)),
                {"allowzero": 0}, dtype=conv.dtype, shape=(seq_len, batch, 2 * hidden_size),
                version=14,
            )
        else:
            packed = pack_padded_sequence(conv, lengths.cpu(), enforce_sorted=False)
            packed_output, _ = self.model.rnn(packed, h0)
            output, _ = pad_packed_sequence(packed_output, total_length=seq_len)

        logits = self.model.embedding(output)
        padding = torch.arange(seq_len, device=images.device)[:, None] >= lengths[None, :]
        classes = torch.arange(logits.shape[2], device=images.device)
        blank_logits = torch.where(classes == 0, 0.0, -1e9).to(logits.dtype)
        logits = torch.where(padding[:, :, None], blank_logits, logits)
        return logits if self.output_type == "logits" else logits.log_softmax(2)


def create_session(path: str | Path, num_threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def validate_outputs(model: CRNN, session: ort.InferenceSession, output_type: str,
                     num_channels: int, img_width: int) -> float:
    """用原始 CRNN 校验真实 collate 路径、混合长度及跨 batch 一致性。"""
    generator = torch.Generator().manual_seed(20260909)
    short_width = max(MIN_IMAGE_WIDTH, 17)
    short = torch.rand(num_channels, 32, short_width, generator=generator)
    longest = min(img_width + 81, MAX_EXPORT_WIDTH - CNN_RIGHT_CONTEXT)
    cases = ([short_width], [short_width, longest],
             [97, short_width, 53], [longest, short_width, 64, short_width])
    single_output = None
    max_error = 0.0
    for widths in cases:
        samples = [
            (short.clone() if width == short_width else
             torch.rand(num_channels, 32, width, generator=generator), str(index))
            for index, width in enumerate(widths)
        ]
        images, _, img_widths = dynamic_padding_collate_fn(samples)
        with torch.inference_mode():
            expected = model(images, img_widths=img_widths).numpy()
        actual = session.run(None, {
            "input": images.numpy(), "img_widths": img_widths.numpy().astype(np.int64),
        })[0]
        if output_type == "logits":
            actual = torch.from_numpy(actual).log_softmax(2).numpy()
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise RuntimeError(f"ONNX 输出形状或数值无效: widths={widths}, shape={actual.shape}")
        error = float(np.max(np.abs(expected - actual)))
        if error >= 1e-3:
            raise RuntimeError(f"PyTorch/ONNX 输出不一致: widths={widths}, max_err={error:.2e}")
        max_error = max(max_error, error)
        steps = model._calc_cnn_output_width(short_width)
        if single_output is None:
            single_output = actual[:steps, 0]
        for index, width in enumerate(widths):
            if width == short_width:
                np.testing.assert_allclose(actual[:steps, index], single_output, rtol=0, atol=1e-3)
        print(f"    batch={len(widths)}, tensor_width={images.shape[-1]}, "
              f"widths={widths}: 最大误差 {error:.2e} ✓")
    return max_error


def export_model(model: CRNN, output_path: str | Path, keys_path: str | Path,
                 img_width: int = 256, output_type: str = "log_probs",
                 num_threads: int = 1) -> float:
    """导出并验证，通过后才将独立 ONNX 文件写入目标路径。"""
    output_path, keys_path = Path(output_path), Path(keys_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"输出路径必须使用 .onnx 扩展名: {output_path}")
    if not MIN_IMAGE_WIDTH + CNN_RIGHT_CONTEXT <= img_width <= MAX_EXPORT_WIDTH:
        raise ValueError(f"img_width 必须在 {MIN_IMAGE_WIDTH + CNN_RIGHT_CONTEXT}..{MAX_EXPORT_WIDTH} 之间")
    if num_threads < 1:
        raise ValueError("num_threads 必须大于 0")
    keys_hash = hashlib.sha256(keys_path.read_bytes()).hexdigest()
    wrapper = DynamoExportWrapper(model, output_type)
    channels = wrapper.model.cnn.stem[0].in_channels
    generator = torch.Generator().manual_seed(20260909)
    images = torch.randn(2, channels, 32, img_width, generator=generator)
    max_width = img_width - CNN_RIGHT_CONTEXT
    widths = torch.tensor([max(MIN_IMAGE_WIDTH, max_width // 2), max_width])
    # 为 torch.export 的卷积索引范围证明提供有限上界，避免 2.9 的无界约束错误。
    batch_dim = torch.export.Dim("batch_size", min=1, max=MAX_EXPORT_BATCH)
    width_dim = torch.export.Dim("width", min=MIN_IMAGE_WIDTH + CNN_RIGHT_CONTEXT,
                                 max=MAX_EXPORT_WIDTH)
    print("[导出] torch.export / dynamo=True, opset=18, 动态 batch/width")
    # 不启用 fallback；导出失败时明确报错，不能静默切回旧版。
    program = torch.onnx.export(
        wrapper, (images, widths), dynamo=True, fallback=False, opset_version=18,
        input_names=["input", "img_widths"], output_names=["output"],
        # widths 的 batch 由 forward 中的等长约束关联，避免重复命名同一轴的提示。
        dynamic_shapes=({0: batch_dim, 3: width_dim}, {0: torch.export.Dim.AUTO}),
        optimize=True, external_data=False,
    )
    if program.exported_program is None:
        raise RuntimeError("导出结果缺少 torch.export.ExportedProgram")
    graph = program.model_proto
    onnx.helper.set_model_props(graph, {
        "output_type": output_type, "output_layout": "TBC", "keys_sha256": keys_hash,
        "num_channels": str(channels), "cnn_right_context": str(CNN_RIGHT_CONTEXT),
        "exporter": "torch.export", "torch_version": torch.__version__,
    })
    onnx.checker.check_model(graph, full_check=True)
    gru_nodes = [node for node in graph.graph.node if node.op_type == "GRU"]
    if len(gru_nodes) != 1 or len(gru_nodes[0].input) < 6 or not all(gru_nodes[0].input[4:6]):
        raise RuntimeError("ONNX GRU 必须包含 sequence_lens 和动态 initial_h")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dynamo-export-", dir=output_path.parent) as directory:
        candidate = Path(directory) / "model.onnx"
        onnx.save(graph, candidate)
        session = create_session(candidate, num_threads)
        print("[验证] 原始 PyTorch CRNN 与 ONNX Runtime")
        max_error = validate_outputs(wrapper.model, session, output_type, channels, img_width)
        os.replace(candidate, output_path)
    print(f"✓ 导出完成: {output_path} ({output_path.stat().st_size / 1024 ** 2:.1f} MB)")
    return max_error


def export_onnx(model_path: str, output_path: str, keys_path: str,
                img_width: int = 256, output_type: str = "log_probs",
                num_threads: int = 1) -> None:
    if num_threads < 1:
        raise ValueError("num_threads 必须大于 0")
    torch.set_num_threads(num_threads)
    print(f"[加载] {model_path}")
    converter = LabelConverter(keys_path)
    state = extract_state_dict(torch.load(model_path, map_location="cpu"))
    model = CRNN(32, infer_num_channels(state), converter.num_classes,
                 nh=state["rnn.weight_hh_l0"].shape[1])
    model.load_state_dict(state)
    model.eval()
    export_model(model, output_path, keys_path, img_width, output_type, num_threads)

    session = create_session(output_path, num_threads)
    images = torch.zeros(1, model.cnn.stem[0].in_channels, 32, img_width)
    widths = torch.tensor([img_width - CNN_RIGHT_CONTEXT])
    feeds = {"input": images.numpy(), "img_widths": widths.numpy()}
    with torch.inference_mode():
        for _ in range(5):
            model(images, img_widths=widths)
            session.run(None, feeds)
        start = time.perf_counter()
        for _ in range(20):
            model(images, img_widths=widths)
        pytorch_ms = (time.perf_counter() - start) * 1000 / 20
        start = time.perf_counter()
        for _ in range(20):
            session.run(None, feeds)
        onnx_ms = (time.perf_counter() - start) * 1000 / 20
    print(f"[CPU] PyTorch: {pytorch_ms:.1f} ms/张, ONNX Runtime: {onnx_ms:.1f} ms/张")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用新版 torch.export 导出动态 CRNN ONNX")
    parser.add_argument("--model_path", default="recognition/checkpoints/1788913363best_cer_0.0067_epoch_573.pth")
    parser.add_argument("--output_path", default="best_recognition_dynamo.onnx")
    parser.add_argument("--keys_path", default="recognition/keys.txt")
    parser.add_argument("--img_width", type=int, default=256, help="导出样例宽度；动态张量宽度支持 80..16384")
    parser.add_argument("--output_type", choices=("log_probs", "logits"), default="log_probs")
    parser.add_argument("--num_threads", type=int, default=1)
    export_onnx(**vars(parser.parse_args()))
