"""导出大图 EAST ONNX：优先 Dynamo、可选图简化，并校验 CPU 数值和动态 batch。"""

import argparse
import json
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from detection_larger.components.onnx_runtime import CPU_THREADS, session_options
from detection_larger.models.east_model import ONNX_CONFIG_KEY, load_checkpoint


BENCHMARK_ENABLED = True  # 导出校验后比较 CPU batch=1 前向速度，不影响数值校验要求。
BENCHMARK_WARMUP = 5  # 每个后端正式测速前的预热次数。
BENCHMARK_RUNS = 20  # 每个后端计入平均前向耗时的运行次数。
OPSET_VERSION = 18  # 两种导出器统一使用的 ONNX 算子集版本。
MAX_BATCH_SIZE = 32  # Dynamo 动态 batch 的最大图片张数；高宽固定为检查点尺寸。
SIMPLIFY_MODEL = True  # 启用时尝试 onnxsim，缺少依赖或简化失败则保留原图。
VALIDATION_RTOL = 1e-4  # PyTorch 与 ONNX 输出比较的相对误差容限。
VALIDATION_ATOL = 1e-5  # PyTorch 与 ONNX 输出比较的绝对误差容限。


def export_graph(model, sample, output):
    """优先使用 Dynamo；失败时显式回退到 TorchScript，保留已有大图接口名称。"""
    names = {"input_names": ["image"], "output_names": ["score", "geometry"]}
    try:
        torch.onnx.export(
            model, (sample,), str(output), **names, opset_version=OPSET_VERSION,
            dynamo=True, external_data=False,
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)},),
        )
        print("Dynamo 导出完成")
    except Exception as error:
        print(f"Dynamo 导出失败，回退到 TorchScript: {error}")
        torch.onnx.export(
            model, sample, str(output), **names, opset_version=OPSET_VERSION,
            dynamo=False, external_data=False,
            dynamic_axes={name: {0: "batch"} for name in ("image", "score", "geometry")},
        )


def simplify_graph(graph):
    """简化是可选优化；失败不影响原始模型后续的强制数值校验。"""
    if not SIMPLIFY_MODEL:
        return graph
    try:
        import onnx
        from onnxsim import simplify

        simplified, valid = simplify(graph)
        if valid:
            onnx.checker.check_model(simplified)
            print("ONNX 图简化完成")
            return simplified
        print("ONNX 图简化校验失败，保留原图")
    except Exception as error:
        print(f"跳过 ONNX 图简化: {error}")
    return graph


def benchmark_model(model, session, sample):
    """以相同线程预算比较 CPU 单图前向；计时不含模型加载与图片预处理。"""
    if BENCHMARK_WARMUP < 0 or BENCHMARK_RUNS < 1:
        raise ValueError("Benchmark warmup must be nonnegative and runs must be positive")
    previous_threads = torch.get_num_threads()
    batch = sample[:1]
    inputs = {"image": batch.numpy()}
    try:
        torch.set_num_threads(CPU_THREADS)
        with torch.inference_mode():
            for _ in range(BENCHMARK_WARMUP):
                model(batch)
            started = time.perf_counter()
            for _ in range(BENCHMARK_RUNS):
                model(batch)
            pytorch_ms = (time.perf_counter() - started) * 1000 / BENCHMARK_RUNS
        for _ in range(BENCHMARK_WARMUP):
            session.run(["score", "geometry"], inputs)
        started = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            session.run(["score", "geometry"], inputs)
        onnx_ms = (time.perf_counter() - started) * 1000 / BENCHMARK_RUNS
    finally:
        torch.set_num_threads(previous_threads)
    result = {"pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms, "speedup": pytorch_ms / onnx_ms}
    print(f"CPU {CPU_THREADS} 线程，预热 {BENCHMARK_WARMUP} 次，计时 {BENCHMARK_RUNS} 次: "
          f"PyTorch {pytorch_ms:.1f} ms/张, ONNX Runtime {onnx_ms:.1f} ms/张, "
          f"加速比 {result['speedup']:.2f}x")
    return result


def export_model(checkpoint_path, output):
    """按检查点尺寸导出并验证 batch 1、2；全部通过后才生成目标文件，拒绝覆盖。"""
    import onnx
    import onnxruntime as ort

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"ONNX output already exists: {output}")
    model, checkpoint = load_checkpoint(checkpoint_path)
    height, width = checkpoint["config"]["input_size"]
    # batch=2 避免 Dynamo 将示例中的单张 batch 特化为固定 1。
    sample = torch.rand(2, 3, height, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="onnx-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model.onnx"
        export_graph(model, sample, staged)
        graph = simplify_graph(onnx.load(str(staged)))
        # 图优化之后再写配置，确保旧推理入口仍能还原预处理契约。
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        properties[ONNX_CONFIG_KEY] = json.dumps(checkpoint["config"])
        onnx.helper.set_model_props(graph, properties)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(staged))

        session = ort.InferenceSession(str(staged), sess_options=session_options(), providers=["CPUExecutionProvider"])
        for batch_size in (1, 2):
            batch = sample[:batch_size]
            with torch.inference_mode():
                expected = model(batch)
            actual = session.run(["score", "geometry"], {"image": batch.numpy()})
            for name, reference, observed in zip(("score", "geometry"), expected, actual):
                np.testing.assert_allclose(
                    observed, reference.numpy(), rtol=VALIDATION_RTOL, atol=VALIDATION_ATOL,
                    err_msg=f"ONNX validation failed: {name}, batch={batch_size}",
                )
                error = np.max(np.abs(observed - reference.numpy()))
                print(f"batch={batch_size}, {name} 最大绝对误差: {error:.2e}")
        if BENCHMARK_ENABLED:
            benchmark_model(model, session, sample)
        # 独占创建防止校验期间同名目标出现时被覆盖；输出不依赖外部权重文件。
        with output.open("xb") as destination, staged.open("rb") as source:
            shutil.copyfileobj(source, destination)
    print(f"ONNX 文件大小: {output.stat().st_size / (1024 ** 2):.1f} MiB; 下一步: .venv/bin/python detection_larger_inference_onnx.py")
    return checkpoint["config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="含 config 和 model 的大图 .pth 检查点")
    parser.add_argument("--output", type=Path, default=Path("detection_larger/artifacts/best_detection.onnx"), help="新的 ONNX 路径，拒绝覆盖")
    args = parser.parse_args()
    config = export_model(args.checkpoint, args.output)
    print(f"导出及数值校验通过: {args.output}; {config}")


if __name__ == "__main__":
    main()
