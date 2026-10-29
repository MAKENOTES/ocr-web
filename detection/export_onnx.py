"""导出统一检测 ONNX：优先 Dynamo、可选图简化，并校验 batch 与动态高宽。"""
import json
import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.components.onnx_runtime import CPU_THREADS, session_options
from detection.utils import ONNX_CONFIG_KEY, checkpoint_config, load_checkpoint


CHECKPOINT_PATH = Path("detection/checkpoints/best_unified.pth")  # 待导出的统一检测检查点，必须包含 config 和 model。
OUTPUT_PATH = Path("best_detection_unified.onnx")  # 导出 ONNX 文件路径；目标已存在时拒绝覆盖。
INPUT_SIZE = 640  # 导出输入尺寸上限，单位 px；也可设为 (高, 宽)，各边至少 64 且为 32 的倍数。
DYNAMIC_SPATIAL = True  # 启用时导出动态高宽，推理可按图片宽高比选择尺寸；关闭则固定为 INPUT_SIZE。
BENCHMARK_ENABLED = True  # 启用时额外比较 CPU batch=1 前向速度；数值校验始终执行。
BENCHMARK_WARMUP = 5  # 每个后端正式测速前的预热次数。
BENCHMARK_RUNS = 20  # 每个后端计入平均前向耗时的运行次数。
OPSET_VERSION = 18  # 两种导出器统一使用的 ONNX 算子集版本。
MAX_BATCH_SIZE = 32  # 动态 batch 的最大图片张数。
MAX_SPATIAL_SIZE = 4096  # 动态高宽的上限，单位 px；只用于导出维度声明，不限制实际推理。
SIMPLIFY_MODEL = True  # 启用时尝试 onnxsim，缺少依赖或简化失败则保留原图。
VALIDATION_RTOL = 1e-4  # PyTorch 与 ONNX 输出比较的相对误差容限。
VALIDATION_ATOL = 1e-5  # PyTorch 与 ONNX 输出比较的绝对误差容限。


def export_graph(model, sample, output, dynamic_spatial):
    """优先使用 Dynamo；失败时回退到 TorchScript，保留统一接口名称。"""
    names = {"input_names": ["image"], "output_names": ["score", "geometry"]}
    dynamic_axes = ({0: torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)}
                    | ({2: torch.export.Dim("height", min=64, max=MAX_SPATIAL_SIZE),
                        3: torch.export.Dim("width", min=64, max=MAX_SPATIAL_SIZE)}
                       if dynamic_spatial else {}))
    try:
        torch.onnx.export(model, (sample,), str(output), **names, opset_version=OPSET_VERSION,
                          dynamo=True, external_data=False, dynamic_shapes=(dynamic_axes,))
        print("Dynamo 导出完成")
    except Exception as error:
        print(f"Dynamo 导出失败，回退到 TorchScript: {error}")
        axes = {"image": {0: "batch"}, "score": {0: "batch"}, "geometry": {0: "batch"}}
        if dynamic_spatial:
            for value in axes.values():
                value.update({2: "height", 3: "width"})
        torch.onnx.export(model, sample, str(output), **names, opset_version=OPSET_VERSION,
                          dynamo=False, external_data=False, dynamic_axes=axes)


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
    print(f"开始 CPU 测速：每个后端预热 {BENCHMARK_WARMUP} 次、计时 {BENCHMARK_RUNS} 次", flush=True)
    previous_threads = torch.get_num_threads()
    batch = sample[:1]
    inputs = {"image": batch.numpy()}
    try:
        torch.set_num_threads(CPU_THREADS)
        print("PyTorch CPU 预热及测速中...", flush=True)
        with torch.inference_mode():
            for _ in range(BENCHMARK_WARMUP):
                model(batch)
            started = time.perf_counter()
            for _ in range(BENCHMARK_RUNS):
                model(batch)
            pytorch_ms = (time.perf_counter() - started) * 1000 / BENCHMARK_RUNS
        print(f"PyTorch CPU 完成：{pytorch_ms:.1f} ms/张；开始 ONNX Runtime 测速...", flush=True)
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


def validation_batches(sample, dynamic_spatial):
    """返回用于数值校验的 (batch, 输入) 列表；动态模型额外覆盖不同高宽。"""
    batches = [(2, sample), (1, sample[:1])]
    if dynamic_spatial:
        height, width = sample.shape[-2:]
        height = min(MAX_SPATIAL_SIZE, height + 32)
        width = min(MAX_SPATIAL_SIZE, width + 32)
        batches.append((1, torch.rand(1, 3, height, width)))
    return batches


def export_model(checkpoint_path, output, input_size=None, dynamic_spatial=None):
    """按配置导出并验证 PyTorch 与 ONNX；全部通过后才生成目标文件，拒绝覆盖。"""
    import onnx
    import onnxruntime as ort

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"ONNX output already exists: {output}")
    input_size = INPUT_SIZE if input_size is None else input_size
    dynamic_spatial = DYNAMIC_SPATIAL if dynamic_spatial is None else dynamic_spatial
    config = checkpoint_config(input_size, dynamic_spatial)
    height, width = config["input_size"]

    model, _ = load_checkpoint(checkpoint_path)
    print(f"导出输入尺寸：{height}×{width}；动态高宽：{dynamic_spatial}；"
          f"CPU 测速：{BENCHMARK_ENABLED}", flush=True)
    # batch=2 避免 Dynamo 将示例中的单张 batch 特化为固定 1。
    sample = torch.rand(2, 3, height, width)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="onnx-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model.onnx"
        export_graph(model, sample, staged, dynamic_spatial)
        graph = simplify_graph(onnx.load(str(staged)))
        # 元数据记录本次实际导出配置，避免沿用检查点尺寸造成后端校验错配。
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        properties[ONNX_CONFIG_KEY] = json.dumps(config)
        onnx.helper.set_model_props(graph, properties)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(staged))

        session = ort.InferenceSession(str(staged), sess_options=session_options(),
                                       providers=["CPUExecutionProvider"])
        for batch_size, batch in validation_batches(sample, dynamic_spatial):
            with torch.inference_mode():
                expected = model(batch)
            actual = session.run(["score", "geometry"], {"image": batch.numpy()})
            for name, reference, observed in zip(("score", "geometry"), expected, actual):
                np.testing.assert_allclose(
                    observed, reference.numpy(), rtol=VALIDATION_RTOL, atol=VALIDATION_ATOL,
                    err_msg=f"ONNX validation failed: {name}, batch={batch_size}, shape={tuple(batch.shape)}")
                error = np.max(np.abs(observed - reference.numpy()))
                print(f"batch={batch_size}, shape={tuple(batch.shape[2:])}, {name} 最大绝对误差: {error:.2e}")

        if BENCHMARK_ENABLED:
            benchmark_model(model, session, sample)
        else:
            print("数值校验通过，跳过 CPU 测速，正在保存 ONNX...", flush=True)
        # 独占创建防止校验期间同名目标出现时被覆盖；输出不依赖外部权重文件。
        with output.open("xb") as destination, staged.open("rb") as source:
            shutil.copyfileobj(source, destination)
    print(f"ONNX 文件大小: {output.stat().st_size / (1024 ** 2):.1f} MiB; "
          f"下一步: .venv/bin/python detection_inference_onnx.py")
    return config


def main():
    """按顶部路径常量导出，数值校验通过后保存新模型。"""
    config = export_model(CHECKPOINT_PATH, OUTPUT_PATH)
    print(f"导出及数值校验通过: {OUTPUT_PATH}; {config}")


if __name__ == "__main__":
    main()
