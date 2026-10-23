"""
export_onnx.py — 将训练好的 EAST PyTorch 模型导出为 ONNX 格式
=====================================================================
第一步：导出 ONNX（无损，推理速度 2-3x 提升）

用法:
    python export_onnx.py \
        --model_path detection/checkpoints/best_model.pth \
        --output_path detection/checkpoints/east_model.onnx \
        --input_size 256x768

依赖:
    pip install onnx onnxruntime onnxsim onnxscript
"""

import torch
import argparse
import os
import numpy as np

# ---- 路径修正：让脚本在任意位置都能找到模块 ----
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.models.east_model import EASTModel


INPUT_SIZE = (256, 768)  # (height, width)，与 detection/train_resnet.py 一致


def normalize_input_size(input_size):
    """统一返回 (高, 宽)，兼容 int、二元序列和 ``256x768`` 字符串。"""
    if isinstance(input_size, str):
        parts = input_size.lower().replace("×", "x").split("x")
        if len(parts) == 1:
            input_size = int(parts[0])
        elif len(parts) == 2:
            input_size = tuple(int(part) for part in parts)
        else:
            raise ValueError(f"无效输入尺寸: {input_size!r}")
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        input_h, input_w = int(input_size[0]), int(input_size[1])
        if input_h > 0 and input_w > 0:
            return input_h, input_w
    raise ValueError(f"input_size 必须是正整数或 (高, 宽)，实际为: {input_size!r}")


def export_onnx(model_path: str, output_path: str, input_size=INPUT_SIZE):
    """
    将 PyTorch EAST 模型导出为 ONNX 格式，并可选地使用 onnx-simplifier 优化。

    Args:
        model_path:  .pth 权重文件路径
        output_path: 输出的 .onnx 文件路径
        input_size:  推理输入尺寸，int 表示正方形，tuple 表示 (高, 宽)
    """
    print("=" * 60)
    print("EAST 模型 → ONNX 导出工具")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载 PyTorch 模型: {model_path}")
    device = torch.device("cpu")
    model = EASTModel(pretrained=False)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    model.to(device)

    # 2. 准备虚拟输入
    input_h, input_w = normalize_input_size(input_size)
    print(f"[2/4] 准备虚拟输入 (1, 3, {input_h}, {input_w})")
    dummy_input = torch.randn(1, 3, input_h, input_w, device=device)

    # 3. 导出 ONNX
    print(f"[3/4] 导出 ONNX 到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ---------- 优先使用新 dynamo 导出器 ----------
    try:
        batch_dim = torch.export.Dim("batch_size", min=1, max=32)
        dynamic_shapes = ({0: batch_dim},)  # 对应 forward(self, x) 的第一个参数 x

        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            input_names=["input"],
            output_names=["score_map", "geo_map"],
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            verbose=False,
        )
        print(f"    ✓ 基础 ONNX 导出完成 (dynamo 导出器)")

    except Exception as e:
        # ---------- 回退到传统 TorchScript 导出器 ----------
        print(f"    ⚠ dynamo 导出失败 ({e})，回退到传统导出器...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=18,
            input_names=["input"],
            output_names=["score_map", "geo_map"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "score_map": {0: "batch_size"},
                "geo_map": {0: "batch_size"},
            },
            verbose=False,
            dynamo=False,
        )
        print(f"    ✓ 基础 ONNX 导出完成 (传统导出器)")

    # 4. 使用 onnxsim 简化模型（可选但强烈推荐）
    print(f"[4/4] 尝试使用 onnx-simplifier 进行图优化...")
    try:
        import onnx
        from onnxsim import simplify

        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)

        model_simplified, check = simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output_path)
            orig_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"    ✓ 简化成功！模型大小: {orig_size:.1f} MB")
        else:
            print("    ⚠ 简化校验失败，保留原始 ONNX")
    except ImportError:
        print("    ⚠ onnxsim 未安装，跳过简化步骤")
        print("    → 安装命令: pip install onnxsim")

    # 5. 验证 ONNX 输出与 PyTorch 输出一致性
    print(f"\n[验证] 对比 PyTorch 与 ONNX 输出...")
    try:
        import onnxruntime as ort
        import onnx

        # PyTorch 推理
        with torch.no_grad():
            pt_score, pt_geo = model(dummy_input)
        pt_score_np = pt_score.numpy()
        pt_geo_np = pt_geo.numpy()

        # ONNX Runtime 推理
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(output_path, sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])
        ort_inputs = {"input": dummy_input.numpy()}
        ort_score, ort_geo = ort_session.run(None, ort_inputs)

        # 比较最大绝对误差
        score_err = np.max(np.abs(pt_score_np - ort_score))
        geo_err = np.max(np.abs(pt_geo_np - ort_geo))
        print(f"    score_map 最大误差: {score_err:.2e}")
        print(f"    geo_map   最大误差: {geo_err:.2e}")

        if score_err < 1e-4 and geo_err < 1e-4:
            print("    ✓ 输出一致，导出成功！")
        else:
            print("    ⚠ 误差偏大，请检查模型结构是否包含不支持的算子")

    except ImportError:
        print("    ⚠ onnxruntime 未安装，跳过验证")
        print("    → 安装命令: pip install onnxruntime")

    # 6. 速度对比 benchmark
    print(f"\n[Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...")
    try:
        import onnxruntime as ort
        import time

        # ONNX Runtime session（开启所有优化）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        ort_session = ort.InferenceSession(output_path, sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])
        ort_inputs = {"input": dummy_input.numpy()}

        # PyTorch 预热
        for _ in range(5):
            with torch.no_grad():
                model(dummy_input)

        # PyTorch 计时
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                model(dummy_input)
        pt_time = (time.perf_counter() - t0) / 20 * 1000

        # ONNX 预热
        for _ in range(5):
            ort_session.run(None, ort_inputs)

        # ONNX 计时
        t0 = time.perf_counter()
        for _ in range(20):
            ort_session.run(None, ort_inputs)
        ort_time = (time.perf_counter() - t0) / 20 * 1000

        speedup = pt_time / ort_time
        print(f"    PyTorch (CPU):    {pt_time:.1f} ms/张")
        print(f"    ONNX Runtime:     {ort_time:.1f} ms/张")
        print(f"    加速比:           {speedup:.2f}x")

    except Exception as e:
        print(f"    Benchmark 失败: {e}")

    print(f"\n{'='*60}")
    print(f"✓ 导出完成: {output_path}")
    print(f"  下一步: 运行 quantize_int8.py 进一步提速 3-4x")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 EAST PyTorch 模型导出为 ONNX")
    parser.add_argument("--model_path", default="detection/checkpoints/detection_epoch_460.pth")
    parser.add_argument("--output_path", default="best_detection.onnx")
    parser.add_argument(
        "--input_size",
        type=normalize_input_size,
        default=INPUT_SIZE,
        help="模型输入尺寸：正方形可写 640，矩形按高x宽写 256x768",
    )
    args = parser.parse_args()
    export_onnx(args.model_path, args.output_path, args.input_size)
