"""
recognition_inference.py — 多后端优化识别推理脚本 (修复版)
=====================================================================
修复: ONNX/OpenVINO 后端现在接收 img_widths 输入，与 PyTorch 后端行为一致。
导出的 ONNX 模型包含 padding mask 逻辑，batch>1 时不再出现字符重复。
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

# ============================================================
# 参数配置
# ============================================================

DEFAULT_MODEL_PATH  = "best_recognition.onnx"
# DEFAULT_MODEL_PATH  = "recognition_quantized_model/int8/crnn_int8.xml"

DEFAULT_IMAGE_PATH  = "recognition/data/test_images1"
DEFAULT_KEYS_PATH   = "recognition/keys.txt"

IMG_HEIGHT          = 32
NUM_CHANNELS        = 1

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
        if self.has_img_widths:
            print(f"  ✓ ONNX 模型包含 img_widths 输入 (padding mask 已启用)")
        else:
            print(f"  ⚠ ONNX 模型不包含 img_widths 输入 (旧版导出，batch>1 可能不准)")

        self._warmup()
        print(f"  ✓ ONNX Runtime 后端加载完成")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, 1, IMG_HEIGHT, 256).astype(np.float32)
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
        return torch.from_numpy(outputs[0])


class OpenVINOBackend:
    """OpenVINO 后端 (修复版: 传入 img_widths)"""
    def __init__(self, model_path: str):
        from openvino import Core
        ie = Core()
        model = ie.read_model(model=model_path)
        config = {"PERFORMANCE_HINT": "LATENCY"}
        self.compiled_model = ie.compile_model(model=model, device_name="CPU", config=config)
        self.infer_request = self.compiled_model.create_infer_request()

        # 检测模型输入
        self.input_names = [inp.any_name for inp in self.compiled_model.inputs]
        self.has_img_widths = "img_widths" in self.input_names
        if self.has_img_widths:
            print(f"  ✓ OpenVINO 模型包含 img_widths 输入 (padding mask 已启用)")
        else:
            print(f"  ⚠ OpenVINO 模型不包含 img_widths 输入 (旧版导出)")

        self._warmup()
        print(f"  ✓ OpenVINO 后端加载完成")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, 1, IMG_HEIGHT, 256).astype(np.float32)
        dummy_widths = np.array([256], dtype=np.int64)
        for _ in range(n):
            feeds = {"input": dummy}
            if self.has_img_widths:
                feeds["img_widths"] = dummy_widths
            self.infer_request.infer(feeds)

    def infer(self, images: torch.Tensor, img_widths: torch.Tensor):
        feeds = {"input": images.numpy()}
        if self.has_img_widths:
            feeds["img_widths"] = img_widths.numpy().astype(np.int64)

        result = self.infer_request.infer(feeds)
        output_node = self.compiled_model.outputs[0]
        out = np.array(result[output_node])

        # OpenVINO 可能改变 layout
        if out.ndim == 3 and out.shape[0] < out.shape[1]:
            out = np.transpose(out, (1, 0, 2))
        return torch.from_numpy(out)


class PyTorchBackend:
    """原始 PyTorch 后端"""
    def __init__(self, model_path: str, keys_path: str):
        from recognition.models.model import CRNN
        converter = LabelConverter(keys_path)
        self.model = CRNN(img_height=IMG_HEIGHT, nc=NUM_CHANNELS, n_class=converter.num_classes)
        checkpoint = torch.load(model_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
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
                        default="openvino")
    parser.add_argument("--batch_size",  type=int,
                        default=32, help="批量大小")
    args = parser.parse_args()

    print("=" * 60)
    print("CRNN 优化推理 (修复版 - 支持 batch>1)")
    print("=" * 60)

    # 1. 收集图片
    image_paths = []
    if os.path.isdir(args.image_path):
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]:
            image_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_path, ext.upper())))
    elif os.path.isfile(args.image_path):
        image_paths = [args.image_path]

    if not image_paths:
        print("✗ 未找到图像文件")
        return

    print(f"发现 {len(image_paths)} 张待处理图片")

    # 2. 初始化核心组件
    converter = LabelConverter(args.keys_path)
    backend = load_backend(args.model_path, args.backend, args.keys_path)

    # 3. 复用原始 InferenceDataset
    dataset = InferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=NUM_CHANNELS
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
        pred_texts = converter.decode(preds)

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