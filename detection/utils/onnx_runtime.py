"""ONNX Runtime 会话配置与 OpenVINO 加载，CPU 线程预算集中在此。"""
import json

import numpy as np

from detection.utils import ONNX_CONFIG_KEY, validate_config


CPU_THREADS = 4  # CPU 推理最多使用的工作线程数；避免大图推理占满整机。


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


def validate_onnx_metadata(properties, model_path):
    """读取并校验统一检测 ONNX 的元数据配置。"""
    try:
        config = json.loads(properties[ONNX_CONFIG_KEY])
        validate_config(config)
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid unified detection ONNX config: {model_path}: {error}") from error
    return config


def static_spatial_size(shape):
    """返回 ONNX 输入 shape 的静态 (高, 宽)；动态维用 None 表示。"""
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D ONNX input shape: {shape!r}")
    return tuple(value if isinstance(value, int) and value > 0 else None for value in shape[2:])


class OpenVINOBackend:
    """从统一检测 ONNX 编译 CPU FP32 模型，按名称取出输出。"""

    def __init__(self, model_path):
        import onnx
        import openvino as ov

        graph = onnx.load(str(model_path), load_external_data=False)
        properties = {entry.key: entry.value for entry in graph.metadata_props}
        self.config = validate_onnx_metadata(properties, model_path)

        inputs = graph.graph.input
        if (len(inputs) != 1 or inputs[0].name != "image"
                or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or len(inputs[0].type.tensor_type.shape.dim) != 4
                or inputs[0].type.tensor_type.shape.dim[1].dim_value != 3
                or {value.name for value in graph.graph.output} != {"score", "geometry"}):
            raise ValueError(f"Incompatible unified detection ONNX inputs/outputs: {model_path}")
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
