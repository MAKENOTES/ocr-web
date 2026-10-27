"""识别推理后端：ONNX Runtime、OpenVINO 和 PyTorch。

所有后端接收 [B,C,32,W] 图片张量与每张图片的真实宽度 img_widths，输出统一为
[T,B,类别数]。归一化函数兼容导出模型可能使用的 [B,T,C] 布局。
"""

import hashlib
import os

import numpy as np
import torch

from .checkpoint import extract_state_dict, infer_num_channels, infer_num_layers
from .converter import LabelConverter
from .dataset import CNN_RIGHT_CONTEXT
from .model import DEFAULT_NUM_LAYERS

# 识别模型固定输入高度（像素）；训练、导出和推理共用该约束。
IMG_HEIGHT = 32
# 无法从模型输入或权重推断通道数时使用的 RGB 通道数。
DEFAULT_NUM_CHANNELS = 3
# 预热张量宽度（像素），需覆盖一次动态宽度推理并保留右侧背景保护区。
WARMUP_IMAGE_WIDTH = 256

# 扩展名到默认后端的映射；未列出的扩展名按 PyTorch checkpoint 处理。
DEFAULT_BACKEND_BY_SUFFIX = {".onnx": "onnx", ".xml": "openvino"}


def normalize_output_to_tbc(output: np.ndarray, batch_size: int) -> np.ndarray:
    """依据已知 batch 维把后端三维输出规范为 [T, B, C]。"""
    if output.ndim != 3:
        raise ValueError(f"识别后端输出必须是三维张量，当前 shape={output.shape}")
    if output.shape[1] == batch_size:
        return output
    if output.shape[0] == batch_size:
        return np.transpose(output, (1, 0, 2))
    raise ValueError(
        f"无法从输出 shape={output.shape} 定位 batch_size={batch_size} 维度"
    )


# ============================================================
# 后端加载器
# ============================================================

class ONNXBackend:
    """ONNX Runtime CPU 后端；存在 img_widths 输入时按真实长度推理。"""

    backend_name = "onnx"

    def __init__(self, model_path: str, num_threads: int = 1, allow_spinning: bool = False):
        import onnxruntime as ort

        if num_threads < 1:
            raise ValueError("num_threads 必须大于 0")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # 预处理在两次 infer 之间执行；空转会抢占这些 CPU 时间，尤其高线程数时。
        opts.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if allow_spinning else "0"
        )
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )

        # 旧版导出模型缺少 img_widths，只能整段推理；新模型应包含该输入。
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.has_img_widths = "img_widths" in self.input_names
        image_input = next(inp for inp in self.session.get_inputs() if inp.name == "input")
        channel_dim = image_input.shape[1]
        self.num_channels = channel_dim if isinstance(channel_dim, int) else DEFAULT_NUM_CHANNELS
        if self.has_img_widths:
            print("  ✓ ONNX 模型包含 img_widths 输入 (sequence_lens 已启用)")
        else:
            print("  ⚠ ONNX 模型不包含 img_widths 输入 (旧版导出，batch>1 可能不准)")

        # 层数来自导出时写入的元数据；缺省说明是未记录层数的旧产物。
        metadata = self.session.get_modelmeta().custom_metadata_map
        declared_layers = metadata.get("num_layers", "")
        if declared_layers.isdigit():
            print(f"  ✓ ONNX 模型为 {declared_layers} 层 BiGRU")

        self._warmup()
        print("  ✓ ONNX Runtime 后端加载完成")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(
            1, self.num_channels, IMG_HEIGHT, WARMUP_IMAGE_WIDTH
        ).astype(np.float32)
        dummy_widths = np.array(
            [WARMUP_IMAGE_WIDTH - CNN_RIGHT_CONTEXT], dtype=np.int64
        )
        for _ in range(n):
            feeds = {"input": dummy}
            if self.has_img_widths:
                feeds["img_widths"] = dummy_widths
            self.session.run(None, feeds)

    def infer(self, images: torch.Tensor, img_widths: torch.Tensor):
        feeds = {"input": images.numpy()}
        if self.has_img_widths:
            feeds["img_widths"] = img_widths.numpy().astype(np.int64)
        outputs = self.session.run(None, feeds)
        output = normalize_output_to_tbc(outputs[0], batch_size=images.shape[0])
        return torch.from_numpy(output)


class OpenVINOBackend:
    """OpenVINO CPU 后端；加载前修复双向 GRU 的 padding 污染。"""

    backend_name = "openvino"

    def __init__(self, model_path: str, num_threads: int = 1):
        from openvino import Core

        if num_threads < 1:
            raise ValueError("num_threads 必须大于 0")
        ie = Core()
        model = ie.read_model(model=model_path)
        from .openvino import make_gru_length_safe

        model = make_gru_length_safe(model)
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": num_threads,
            "NUM_STREAMS": 1,
            # 未量化层保持 FP32，避免不同 CPU 默认使用 BF16 带来精度变化。
            "INFERENCE_PRECISION_HINT": "f32",
        }
        self.compiled_model = ie.compile_model(model=model, device_name="CPU", config=config)
        self.infer_request = self.compiled_model.create_infer_request()

        # OpenVINO 可能在 ONNX 转换时重命名输入，因此按 rank 识别图片和宽度，
        # 推理时直接使用端口对象，不依赖字符串名称。
        inputs = list(self.compiled_model.inputs)
        self.input_names = [inp.any_name for inp in inputs]
        self.image_input = next(
            (inp for inp in inputs if len(inp.partial_shape) == 4),
            None,
        )
        self.widths_input = next(
            (inp for inp in inputs if len(inp.partial_shape) == 1),
            None,
        )
        if self.image_input is None:
            raise ValueError(f"OpenVINO 模型缺少四维图片输入: {self.input_names}")

        self.has_img_widths = self.widths_input is not None
        try:
            self.num_channels = self.image_input.partial_shape[1].get_length()
        except Exception:
            self.num_channels = DEFAULT_NUM_CHANNELS
        if self.has_img_widths:
            print(
                "  ✓ OpenVINO 模型包含长度输入 "
                f"({self.widths_input.any_name}, sequence_lens 已启用)"
            )
        else:
            print("  ⚠ OpenVINO 模型不包含 img_widths 输入 (旧版导出)")

        self._warmup()
        print("  ✓ OpenVINO 后端加载完成")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(
            1, self.num_channels, IMG_HEIGHT, WARMUP_IMAGE_WIDTH
        ).astype(np.float32)
        dummy_widths = np.array(
            [WARMUP_IMAGE_WIDTH - CNN_RIGHT_CONTEXT], dtype=np.int64
        )
        for _ in range(n):
            feeds = {self.image_input: dummy}
            if self.has_img_widths:
                feeds[self.widths_input] = dummy_widths
            self.infer_request.infer(feeds)

    def infer(self, images: torch.Tensor, img_widths: torch.Tensor):
        feeds = {self.image_input: images.numpy()}
        if self.has_img_widths:
            feeds[self.widths_input] = img_widths.numpy().astype(np.int64)

        result = self.infer_request.infer(feeds)
        output_node = self.compiled_model.outputs[0]
        out = np.array(result[output_node])

        out = normalize_output_to_tbc(out, batch_size=images.shape[0])
        return torch.from_numpy(out)


class PyTorchBackend:
    """PyTorch checkpoint 后端：按权重实际层数与通道数构建模型。"""

    backend_name = "pytorch"

    def __init__(self, model_path: str, keys_path: str, num_threads: int = 1):
        if num_threads < 1:
            raise ValueError("num_threads 必须大于 0")
        torch.set_num_threads(num_threads)
        from .model import CRNN

        converter = LabelConverter(keys_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        self.num_channels = infer_num_channels(state_dict, default=DEFAULT_NUM_CHANNELS)
        num_layers = infer_num_layers(state_dict, default=DEFAULT_NUM_LAYERS)
        self.model = CRNN(
            img_height=IMG_HEIGHT,
            nc=self.num_channels,
            n_class=converter.num_classes,
            num_layers=num_layers,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"  ✓ PyTorch 后端加载完成 (BiGRU {num_layers} 层)")

    def infer(self, images: torch.Tensor, img_widths: torch.Tensor):
        with torch.inference_mode():
            output = self.model(images, img_widths=img_widths)
        return output


def load_backend(model_path: str, backend: str, keys_path: str, num_threads: int = 1,
                 allow_spinning: bool = False):
    """按显式后端名或扩展名创建后端；ONNX 会校验字典哈希。"""
    if backend not in ("auto", "onnx", "openvino", "pytorch"):
        raise ValueError(f"未知后端: {backend}")

    extension = os.path.splitext(model_path)[1].lower()
    if backend == "auto":
        backend = DEFAULT_BACKEND_BY_SUFFIX.get(extension, "pytorch")

    print(f"\n[后端] 使用: {backend.upper()}")
    if backend == "onnx":
        result = ONNXBackend(model_path, num_threads, allow_spinning)
        metadata = result.session.get_modelmeta().custom_metadata_map
        if "keys_sha256" in metadata:
            with open(keys_path, "rb") as keys_file:
                keys_hash = hashlib.sha256(keys_file.read()).hexdigest()
            if keys_hash != metadata["keys_sha256"]:
                raise ValueError(f"字典与导出模型不一致: {keys_path}；请使用导出时的 keys 文件")
        return result
    if backend == "openvino":
        return OpenVINOBackend(model_path, num_threads)
    return PyTorchBackend(model_path, keys_path, num_threads)
