"""识别推理命令行入口。

runtime 自身要求显式传入模型、字典和图片路径；历史脚本通过默认值保持旧命令兼容。
"""

import argparse
import os

import torch

from recognition.runtime.engine import BACKEND_CHOICES, RecognitionEngine
from recognition.runtime.preprocess import collect_image_paths


def build_parser(*, description: str, default_model_path: str | None,
                 default_image_path: str | None, default_keys_path: str | None,
                 default_backend: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_path", type=str, default=default_model_path,
                        help="模型文件：.onnx、.xml 或 .pth")
    parser.add_argument("--image_path", type=str, default=default_image_path,
                        help="单张图片或图片目录")
    parser.add_argument("--keys_path", type=str, default=default_keys_path,
                        help="与模型配套的字符字典")
    parser.add_argument("--output_file", type=str, default=None,
                        help="结果输出文件；不指定时打印到标准输出")
    parser.add_argument("--backend", type=str, choices=BACKEND_CHOICES,
                        default=default_backend)
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="CPU 推理线程数，建议实测 1/2/4")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="预处理进程数；单图用 0")
    parser.add_argument("--allow_spinning", action="store_true",
                        help="ONNX 线程空转；需实测完整流水线是否更快")
    parser.add_argument("--sort_by_width", action="store_true",
                        help="按宽高比组 batch，结果仍按原文件顺序输出")
    return parser


def main(argv=None, *, description: str = "CRNN 优化推理",
         default_model_path: str | None = None, default_image_path: str | None = None,
         default_keys_path: str | None = None, default_backend: str = "auto"):
    """解析参数并执行推理；路径为 None 时要求命令行显式提供。"""
    parser = build_parser(
        description=description,
        default_model_path=default_model_path,
        default_image_path=default_image_path,
        default_keys_path=default_keys_path,
        default_backend=default_backend,
    )
    args = parser.parse_args(argv)

    for name, value in (
        ("--model_path", args.model_path),
        ("--image_path", args.image_path),
        ("--keys_path", args.keys_path),
    ):
        if value is None:
            parser.error(f"必须指定 {name}")
    if args.batch_size < 1 or args.num_threads < 1 or args.num_workers < 0:
        parser.error("batch_size/num_threads 必须大于 0，num_workers 不能为负")

    # ONNX/OpenVINO 自己管理推理线程；预处理和解码只处理小张量，用单线程
    # 避免两个线程池竞争。PyTorchBackend 会在加载时单独设置推理线程数。
    torch.set_num_threads(1)

    print("=" * 60)
    print(f"{description} - 支持 batch>1")
    print("=" * 60)

    # 1. 收集图片
    image_paths = collect_image_paths(args.image_path)
    print(f"发现 {len(image_paths)} 张待处理图片")

    # 2. 加载模型并推理
    engine = RecognitionEngine(
        model_path=args.model_path,
        keys_path=args.keys_path,
        backend=args.backend,
        num_threads=args.num_threads,
        allow_spinning=args.allow_spinning,
    )
    report = engine.recognize(
        image_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sort_by_width=args.sort_by_width,
    )

    # 3. 输出结果；文件名即可对应输入，避免不同目录同名时路径过长。
    results = [(os.path.basename(result.path), result.text) for result in report]
    print(f"\n{'-' * 60}")
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as output:
            for filename, text in results:
                output.write(f"{filename}\t{text}\n")
        print(f"结果已保存至: {args.output_file}")
    else:
        print("识别结果:")
        for filename, text in results:
            print(f"  {filename}\t{text}")

    print(f"\n{'=' * 60}")
    print("推理完成")
    print(f"  图像数:         {len(image_paths)} 张")
    print(f"  推理+解码耗时:  {report.infer_ms:.1f} ms")
    print(f"  总耗时(含预处理): {report.wall_seconds * 1000:.0f} ms")
    print(f"  吞吐量:         {len(image_paths) / report.wall_seconds:.2f} 张/秒")
    print(f"{'=' * 60}")
