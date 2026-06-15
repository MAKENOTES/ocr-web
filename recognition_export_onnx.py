"""
recognition_export_onnx.py — 将训练好的 CRNN PyTorch 模型导出为 ONNX 格式 (修复版)
=====================================================================
修复问题:
  1. GRU 的 h0 在 batch_size=1 时被 trace 为常量，导致 batch>1 推理时
     hidden state 形状不匹配，反向 GRU 输出错乱 → 字符重复（如 "乙乙方"）
  2. ONNX 不支持 img_widths 的 Python for-loop 动态计算，
     改用纯 tensor 算术使 padding mask 可导出

解决方案:
  用 OnnxExportWrapper 包装原始模型:
  - h0 改为由输入 batch 维度动态构建（expand 而非 zeros(b,...)）
  - img_widths 作为第二个输入传入 ONNX，padding mask 全部用 tensor 运算
  - CNN 宽度下采样公式用 torch 算术替代 Python int 循环

用法:
    python recognition_export_onnx.py

依赖:
    pip install onnx onnxruntime onnxsim
"""

import torch
import torch.nn.functional as F
import os
import numpy as np

# ---- 路径修正 ----
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.models.model import CRNN
from recognition.components.converter import LabelConverter


class OnnxExportWrapper(torch.nn.Module):
    """
    ONNX 导出专用包装器。

    解决两个核心问题:
    1. h0 动态化: 用 expand 从 [num_layers*2, 1, hidden] 广播到任意 batch，
       避免 trace 时 batch 维度被烧死为常量。
    2. img_widths 支持: 将 _calc_cnn_output_width 的 Python 循环
       替换为纯 tensor 算术，使 padding mask 对 ONNX 可见。
    """

    def __init__(self, crnn_model: CRNN):
        super().__init__()
        self.cnn = crnn_model.cnn
        self.cnn_dropout = crnn_model.cnn_dropout
        self.rnn = crnn_model.rnn
        self.rnn_dropout = crnn_model.rnn_dropout
        self.embedding = crnn_model.embedding

        # 预计算 h0 模板: [num_layers * num_directions, 1, hidden_size]
        num_layers = self.rnn.num_layers
        num_directions = 2 if self.rnn.bidirectional else 1
        hidden_size = self.rnn.hidden_size
        # 注册为 buffer（不参与梯度，但会随 .to(device) 移动）
        self.register_buffer(
            'h0_template',
            torch.zeros(num_layers * num_directions, 1, hidden_size)
        )

    @staticmethod
    def _calc_cnn_width_tensor(pixel_widths: torch.Tensor) -> torch.Tensor:
        """
        纯 tensor 运算版本的 CNN 宽度下采样计算。
        对应 MobileFeatureExtractor: stem(s=2,p=1,k=3) → pool(k=2,s=2)

        stem: w = floor((W + 2*1 - 3) / 2) + 1 = floor((W - 1) / 2) + 1
        pool: w = floor((w_stem - 2) / 2) + 1
        """
        # stem: Conv2d(k=3, s=2, p=1)
        w = (pixel_widths + 2 * 1 - 3) // 2 + 1
        # pool: MaxPool2d(k=2, s=2)
        w = (w - 2) // 2 + 1
        return w

    def forward(self, input: torch.Tensor, img_widths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input:      [B, C, 32, W] 图像张量（含 padding）
            img_widths: [B] 每张图片 padding 前的真实像素宽度 (int32/int64)
        Returns:
            output:     [SeqLen, B, NumClasses] log_softmax
        """
        # ---- CNN ----
        conv = self.cnn(input)           # [B, 256, 1, W/4]
        conv = conv.squeeze(2)           # [B, 256, W/4]
        conv = conv.permute(2, 0, 1)     # [SeqLen, B, 256]
        conv = self.cnn_dropout(conv)

        b = conv.size(1)
        seq_len = conv.size(0)

        # ---- h0: 动态 batch（关键修复） ----
        # expand 不分配新内存，ONNX 能正确处理动态 batch 维
        h0 = self.h0_template.expand(-1, b, -1).contiguous()

        # ---- Padding Mask（纯 tensor 运算，ONNX 兼容） ----
        real_seq_lens = self._calc_cnn_width_tensor(img_widths.long())
        real_seq_lens = real_seq_lens.clamp(min=1, max=seq_len)

        # mask: [SeqLen, B]  True = padding 位置
        time_indices = torch.arange(seq_len, device=conv.device).unsqueeze(1)  # [SeqLen, 1]
        padding_mask = time_indices >= real_seq_lens.unsqueeze(0)               # [SeqLen, B]

        # 将 padding 时间步的 CNN 特征置零（阻止反向 GRU 传播噪声）
        conv = conv.masked_fill(padding_mask.unsqueeze(2), 0.0)

        # ---- GRU ----
        rnn_out, _ = self.rnn(conv, h0)
        rnn_out = self.rnn_dropout(rnn_out)
        output = self.embedding(rnn_out)  # [SeqLen, B, n_class]

        # 屏蔽 padding 区域的 logits
        # 将 padding 位置所有类别设为 -1e9，blank (idx=0) 设为 0
        mask_expanded = padding_mask.unsqueeze(2)  # [SeqLen, B, 1]
        blank_score = torch.zeros_like(output)
        blank_score[:, :, 0] = 0.0  # blank 得分 = 0
        neg_inf = torch.full_like(output, -1e9)
        neg_inf[:, :, 0] = 0.0  # blank 位保持 0
        output = torch.where(mask_expanded, neg_inf, output)

        return F.log_softmax(output, dim=2)


def export_onnx(model_path: str, output_path: str, keys_path: str,
                img_height: int = 32, img_width: int = 256):
    """
    将 PyTorch CRNN 模型导出为 ONNX 格式。
    """
    print("=" * 60)
    print("CRNN 模型 → ONNX 导出工具 (修复版)")
    print("=" * 60)

    device = torch.device("cpu")

    # 1. 加载模型
    print(f"\n[1/5] 加载 PyTorch 模型: {model_path}")
    converter = LabelConverter(keys_path)
    num_classes = converter.num_classes

    model = CRNN(img_height=img_height, nc=1, n_class=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print(f"    类别数: {num_classes}")

    # 2. 包装模型
    print(f"[2/5] 包装模型 (OnnxExportWrapper)")
    wrapper = OnnxExportWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    # 3. 准备虚拟输入（使用 batch_size=2 以确保动态 batch 正确导出）
    export_batch = 2
    print(f"[3/5] 准备虚拟输入 ({export_batch}, 1, {img_height}, {img_width})")
    dummy_input = torch.randn(export_batch, 1, img_height, img_width, device=device)
    dummy_widths = torch.full((export_batch,), img_width, dtype=torch.long, device=device)

    # 4. 导出 ONNX
    print(f"[4/5] 导出 ONNX 到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_widths),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input", "img_widths"],
        output_names=["output"],
        dynamic_axes={
            "input":      {0: "batch_size", 3: "width"},
            "img_widths": {0: "batch_size"},
            "output":     {0: "seq_len", 1: "batch_size"},
        },
        verbose=False,
        dynamo=False,
    )
    print(f"    ✓ 基础 ONNX 导出完成")

    # 5. onnxsim 简化
    print(f"[5/5] 尝试使用 onnx-simplifier 进行图优化...")
    try:
        import onnx
        from onnxsim import simplify

        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)

        model_simplified, check = simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output_path)
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"    ✓ 简化成功！模型大小: {size_mb:.1f} MB")
        else:
            print("    ⚠ 简化校验失败，保留原始 ONNX")
    except ImportError:
        print("    ⚠ onnxsim 未安装，跳过简化步骤")
        print("    → 安装命令: pip install onnxsim")

    # ---- 验证：对比 PyTorch 与 ONNX 在不同 batch_size 下的输出 ----
    print(f"\n[验证] 对比 PyTorch 与 ONNX 输出...")
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(output_path, sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])

        for test_batch in [1, 4]:
            test_input = torch.randn(test_batch, 1, img_height, img_width, device=device)
            test_widths = torch.full((test_batch,), img_width, dtype=torch.long)

            # PyTorch (原始模型，带 img_widths)
            with torch.no_grad():
                pt_output = model(test_input, img_widths=test_widths).numpy()

            # ONNX Runtime
            ort_inputs = {
                "input": test_input.numpy(),
                "img_widths": test_widths.numpy(),
            }
            ort_output = ort_session.run(None, ort_inputs)[0]

            max_err = np.max(np.abs(pt_output - ort_output))
            status = "✓" if max_err < 1e-3 else "⚠"
            print(f"    batch={test_batch}: 最大误差 {max_err:.2e} {status}")

    except ImportError:
        print("    ⚠ onnxruntime 未安装，跳过验证")

    # ---- Benchmark ----
    print(f"\n[Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...")
    try:
        import onnxruntime as ort
        import time

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        ort_session = ort.InferenceSession(output_path, sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])

        bench_input = torch.randn(1, 1, img_height, img_width, device=device)
        bench_widths = torch.full((1,), img_width, dtype=torch.long)
        ort_inputs = {
            "input": bench_input.numpy(),
            "img_widths": bench_widths.numpy(),
        }

        # PyTorch
        for _ in range(5):
            with torch.no_grad():
                model(bench_input)
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                model(bench_input)
        pt_time = (time.perf_counter() - t0) / 20 * 1000

        # ONNX
        for _ in range(5):
            ort_session.run(None, ort_inputs)
        t0 = time.perf_counter()
        for _ in range(20):
            ort_session.run(None, ort_inputs)
        ort_time = (time.perf_counter() - t0) / 20 * 1000

        print(f"    PyTorch (CPU):    {pt_time:.1f} ms/张")
        print(f"    ONNX Runtime:     {ort_time:.1f} ms/张")
        print(f"    加速比:           {pt_time/ort_time:.2f}x")

    except Exception as e:
        print(f"    Benchmark 失败: {e}")

    print(f"\n{'='*60}")
    print(f"✓ 导出完成: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    model_path = "best_recognition.pth"
    output_path = "best_recognition.onnx"
    keys_path = "recognition/keys.txt"
    img_height = 32
    img_width = 256

    export_onnx(model_path, output_path, keys_path, img_height, img_width)