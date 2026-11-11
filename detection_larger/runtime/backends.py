"""大图检测 ONNX/OpenVINO 后端：会话配置、元数据校验和 CPU 前向。

runtime 由 tools/sync_detection_larger_runtime.py 生成，部署包只包含本包、入口脚本和模型。
"""
import json
from pathlib import Path

import numpy as np

from .config import ONNX_CONFIG_KEY, validate_config


CPU_THREADS = 4  # CPU 推理最多使用的工作线程数，避免大图推理占满整机。
BACKEND_CHOICES = ("auto", "onnx", "openvino")  # 可选的推理后端；auto 等价 onnx。


def session_options():
    """启用 ONNX Runtime 图优化和内存复用，使用固定线程预算。"""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = True
    options.enable_mem_reuse = True
    return options


def validate_metadata(properties, model_path):
    """读取并校验 ONNX 元数据中的大图检测配置。"""
    try:
        config = validate_config(json.loads(properties[ONNX_CONFIG_KEY]))
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid document ONNX config: {model_path}: {error}") from error
    return config


def static_spatial_size(shape):
    """返回 ONNX 输入 shape 的静态 (高, 宽)；动态维用 None 表示。"""
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D ONNX input shape: {shape!r}")
    return tuple(value if isinstance(value, int) and value > 0 else None for value in shape[2:])


class OnnxRuntimeBackend:
    """加载大图检测 ONNX，校验元数据/输入输出并执行 CPU 前向。"""

    def __init__(self, model_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), sess_options=session_options(), providers=["CPUExecutionProvider"])
        self.config = validate_metadata(
            self.session.get_modelmeta().custom_metadata_map, model_path)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "image" or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3
                or {value.name for value in outputs} != {"score", "geometry"}):
            raise ValueError(f"Incompatible document ONNX inputs/outputs: {model_path}")
        static_size = static_spatial_size(inputs[0].shape)
        if None not in static_size and static_size != tuple(self.config["input_size"]):
            raise ValueError(f"ONNX input size {static_size} does not match config "
                             f"{self.config['input_size']}: {model_path}")
        self.static_input_size = None if None in static_size else tuple(static_size)

    def infer(self, batch):
        """执行一次 NCHW 前向，返回 score/geometry 的 numpy 数组。"""
        score, geometry = self.session.run(["score", "geometry"], {"image": batch})
        return score, geometry


class OpenVINOBackend:
    """从大图检测 ONNX 编译 CPU FP32 模型，按名称取出输出。"""

    def __init__(self, model_path):
        import onnx
        import openvino as ov

        graph = onnx.load(str(model_path), load_external_data=False)
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        self.config = validate_metadata(properties, model_path)

        inputs = graph.graph.input
        if (len(inputs) != 1 or inputs[0].name != "image"
                or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or len(inputs[0].type.tensor_type.shape.dim) != 4
                or inputs[0].type.tensor_type.shape.dim[1].dim_value != 3
                or {value.name for value in graph.graph.output} != {"score", "geometry"}):
            raise ValueError(f"Incompatible document ONNX inputs/outputs: {model_path}")
        spatial = tuple(dimension.dim_value if dimension.dim_value > 0 else None
                        for dimension in inputs[0].type.tensor_type.shape.dim[2:])
        if None not in spatial and spatial != tuple(self.config["input_size"]):
            raise ValueError(f"ONNX input size {spatial} does not match config "
                             f"{self.config['input_size']}: {model_path}")
        self.static_input_size = None if None in spatial else spatial

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(model_path)), "CPU", {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": CPU_THREADS,
            "INFERENCE_PRECISION_HINT": "f32",
        })
        self.request = self.compiled.create_infer_request()

    def infer(self, batch):
        """返回独立的置信度与几何数组，避免下一次请求覆盖输出缓冲区。"""
        self.request.infer({"image": batch})
        return tuple(np.array(self.request.get_tensor(name).data, copy=True)
                     for name in ("score", "geometry"))


def load_backend(model_path, backend="auto"):
    """按显式后端加载 ONNX；auto 固定为 ONNX Runtime。"""
    if Path(model_path).suffix.lower() != ".onnx":
        raise ValueError(f"ONNX 推理入口只接受 .onnx 模型: {model_path}")
    if backend == "auto":
        backend = "onnx"
    if backend == "onnx":
        print("  ✓ 后端: ONNX Runtime")
        return OnnxRuntimeBackend(model_path)
    if backend == "openvino":
        print("  ✓ 后端: OpenVINO")
        return OpenVINOBackend(model_path)
    raise ValueError(f"未知后端: {backend}")


def exported_shape(backend):
    """返回 (是否动态高宽, 输入尺寸上限, 固定输入尺寸)。"""
    config = backend.config
    static_size = backend.static_input_size
    dynamic = static_size is None
    if bool(config["dynamic_spatial"]) != dynamic:
        raise ValueError(f"ONNX dynamic_spatial={config['dynamic_spatial']} "
                         f"does not match exported shape {static_size}")
    if dynamic:
        size = tuple(int(value) for value in config["input_size"])
        return True, size, None
    size = tuple(int(value) for value in static_size)
    return False, size, size
