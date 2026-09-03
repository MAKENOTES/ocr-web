"""
recognition_inference.py — 多后端优化识别推理脚本 (修复版)
=====================================================================
修复: ONNX/OpenVINO 后端接收 img_widths，并统一输出为 TBC。
导出的 ONNX GRU 使用 sequence_lens，batch>1 时不再读取其他样本造成的 padding。
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# -- 动态添加项目根目录到 sys.path --
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 强制复用核心模块，保证逻辑一致
from recognition.components.converter import LabelConverter
from recognition.components.dataset import InferenceDataset, dynamic_padding_collate_fn
from recognition.utils.checkpoint import extract_state_dict, infer_num_channels

# ============================================================
# 参数配置
# ============================================================

DEFAULT_MODEL_PATH  = "best_recognition.onnx"
# DEFAULT_MODEL_PATH  = "recognition_quantized_model/int8/crnn_int8.xml"

DEFAULT_IMAGE_PATH  = "recognition/data/test_images"
DEFAULT_KEYS_PATH   = "recognition/keys.txt"

IMG_HEIGHT          = 32
DEFAULT_NUM_CHANNELS = 3


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
    """ONNX Runtime 后端 (修复版: 传入 img_widths)"""
    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

        # 检测模型是否包含 img_widths 输入
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.has_img_widths = "img_widths" in self.input_names
        image_input = next(inp for inp in self.session.get_inputs() if inp.name == "input")
        channel_dim = image_input.shape[1]
        self.num_channels = channel_dim if isinstance(channel_dim, int) else DEFAULT_NUM_CHANNELS
        if self.has_img_widths:
            print(f"  ✓ ONNX 模型包含 img_widths 输入 (sequence_lens 已启用)")
        else:
            print(f"  ⚠ ONNX 模型不包含 img_widths 输入 (旧版导出，batch>1 可能不准)")

        self._warmup()
        print(f"  ✓ ONNX Runtime 后端加载完成")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, self.num_channels, IMG_HEIGHT, 256).astype(np.float32)
        dummy_widths = np.array([256], dtype=np.int64)
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
    """OpenVINO 后端 (修复版: 传入 img_widths)"""
    def __init__(self, model_path: str):
        from openvino import Core
        ie = Core()
        model = ie.read_model(model=model_path)
        config = {"PERFORMANCE_HINT": "LATENCY"}
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
            print(f"  ⚠ OpenVINO 模型不包含 img_widths 输入 (旧版导出)")

        self._warmup()
        print(f"  ✓ OpenVINO 后端加载完成")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, self.num_channels, IMG_HEIGHT, 256).astype(np.float32)
        dummy_widths = np.array([256], dtype=np.int64)
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
    """原始 PyTorch 后端"""
    def __init__(self, model_path: str, keys_path: str):
        from recognition.models.model import CRNN
        converter = LabelConverter(keys_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        self.num_channels = infer_num_channels(state_dict, default=DEFAULT_NUM_CHANNELS)
        self.model = CRNN(
            img_height=IMG_HEIGHT,
            nc=self.num_channels,
            n_class=converter.num_classes,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"  ✓ PyTorch 后端加载完成")

    def infer(self, images: torch.Tensor, img_widths: torch.Tensor):
        with torch.no_grad():
            output = self.model(images, img_widths=img_widths)
        return output


def load_backend(model_path: str, backend: str, keys_path: str):
    ext = os.path.splitext(model_path)[1].lower()
    if backend == "auto":
        backend = "onnx" if ext == ".onnx" else "pytorch"

    print(f"\n[后端] 使用: {backend.upper()}")
    if backend == "onnx":
        return ONNXBackend(model_path)
    elif backend == "openvino":
        return OpenVINOBackend(model_path)
    elif backend == "pytorch":
        return PyTorchBackend(model_path, keys_path)
    else:
        raise ValueError(f"未知后端: {backend}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CRNN 优化推理 (修复版)")
    parser.add_argument("--model_path",  type=str,
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path",  type=str,
                        default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--keys_path",   type=str,
                        default=DEFAULT_KEYS_PATH)
    parser.add_argument("--output_file", type=str,
                        default=None)
    parser.add_argument("--backend",     type=str, choices=["auto", "onnx", "openvino", "pytorch"],
                        default="onnx")
    parser.add_argument("--batch_size",  type=int,
                        default=1, help="批量大小")
    args = parser.parse_args()

    print("=" * 60)
    print("CRNN 优化推理 (修复版 - 支持 batch>1)")
    print("=" * 60)

    # 1. 收集图片
    image_paths = []
    if os.path.isdir(args.image_path):
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
            image_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_path, ext.upper())))
    elif os.path.isfile(args.image_path):
        image_paths = [args.image_path]

    if not image_paths:
        print("✗ 未找到图像文件")
        return

    image_paths = sorted(set(image_paths))
    print(f"发现 {len(image_paths)} 张待处理图片")

    # 2. 初始化核心组件
    converter = LabelConverter(args.keys_path)
    backend = load_backend(args.model_path, args.backend, args.keys_path)

    # detection 输出已经是文字框，直接识别，不做二次裁剪或按颜色删除像素。
    dataset = InferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=backend.num_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=dynamic_padding_collate_fn
    )

    # 4. 执行推理
    results = []
    total_ms = 0.0
    t_wall_start = time.perf_counter()

    for images, paths, img_widths in loader:
        t0 = time.perf_counter()

        # 统一接口: 所有后端都接收 img_widths
        preds = backend.infer(images, img_widths)
        pred_texts = converter.decode(preds, layout="TBC")

        infer_ms = (time.perf_counter() - t0) * 1000
        total_ms += infer_ms

        for path, text in zip(paths, pred_texts):
            results.append((os.path.basename(path), text))

    t_wall = time.perf_counter() - t_wall_start

    # 5. 输出结果
    print(f"\n{'-'*60}")
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for filename, text in results:
                f.write(f"{filename}\t{text}\n")
        print(f"结果已保存至: {args.output_file}")
    else:
        print("识别结果:")
        for filename, text in results:
            print(f"  {filename}\t{text}")

    print(f"\n{'='*60}")
    print(f"推理完成")
    print(f"  图像数:         {len(image_paths)} 张")
    print(f"  纯推理耗时:     {total_ms:.1f} ms")
    print(f"  总耗时(含预处理): {t_wall*1000:.0f} ms")
    print(f"  吞吐量:         {len(image_paths)/t_wall:.2f} 张/秒")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
