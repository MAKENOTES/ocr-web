"""大图检测推理命令行入口。

runtime 自身要求显式传入模型、图片和输出目录；仓库入口通过默认值保持旧命令兼容。
"""
import argparse
from pathlib import Path

from .backends import BACKEND_CHOICES, exported_shape, load_backend
from .engine import run_inference
from .image import IMAGE_EXTENSIONS
from .postprocess import DEFAULT_NMS_IOU_THRESHOLD, DEFAULT_SCORE_THRESHOLD


def build_parser(*, description: str, default_model_path: Path | None = None,
                 default_image_path: Path | None = None,
                 default_output_dir: Path | None = None,
                 default_backend: str = "onnx",
                 default_score_thr: float = DEFAULT_SCORE_THRESHOLD,
                 default_nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
                 default_save_crops: bool = False,
                 default_crop_dir: Path | None = None,
                 default_warmup_runs: int = 0) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", type=Path, default=default_model_path,
                        help="已导出的大图检测 .onnx 模型")
    parser.add_argument("--image_path", type=Path, default=default_image_path,
                        help="输入图片文件或目录")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir,
                        help="画框图、labels 和裁片输出目录")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default=default_backend,
                        help="auto 等价 onnx；openvino 需显式选择")
    parser.add_argument("--score_thr", type=float, default=default_score_thr,
                        help="文字置信度阈值，范围 (0, 1)")
    parser.add_argument("--nms_iou", type=float, default=default_nms_iou,
                        help="重复框抑制交并比，范围 (0, 1]")
    parser.add_argument("--save_crops", action="store_true", default=default_save_crops,
                        help="保存原图文字框裁片")
    parser.add_argument("--crop_dir", type=Path, default=default_crop_dir,
                        help="裁片输出目录；仅 --save_crops 时使用")
    parser.add_argument("--warmup_runs", type=int, default=default_warmup_runs,
                        help="正式计时前的预热次数，0 表示关闭")
    return parser


def validate_args(parser, args):
    """在加载模型前拒绝明显无效的路径和阈值，避免跑到一半才失败。"""
    for name, value in (("--model_path", args.model_path), ("--image_path", args.image_path),
                        ("--output_dir", args.output_dir)):
        if value is None:
            parser.error(f"必须指定 {name}")
    if not 0 < args.score_thr < 1:
        parser.error("--score_thr 必须在 (0, 1) 内")
    if not 0 < args.nms_iou <= 1:
        parser.error("--nms_iou 必须在 (0, 1] 内")
    if args.warmup_runs < 0:
        parser.error("--warmup_runs 必须是非负整数")
    if not args.model_path.is_file():
        parser.error(f"模型文件不存在: {args.model_path}")
    if args.model_path.suffix.lower() != ".onnx":
        parser.error(f"ONNX 推理入口只接受 .onnx 模型: {args.model_path}")
    if not args.image_path.exists():
        parser.error(f"图片路径不存在: {args.image_path}")
    if args.image_path.is_file() and args.image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f"不支持的图片扩展名: {args.image_path}")
    if args.image_path.is_dir() and not any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in args.image_path.iterdir()):
        parser.error(f"目录中没有支持的图片: {args.image_path}")


def main(argv=None, *, description: str = "大图检测 ONNX CPU 推理：ONNX Runtime 或 OpenVINO。",
         default_model_path: Path | None = None, default_image_path: Path | None = None,
         default_output_dir: Path | None = None, default_backend: str = "onnx",
         default_score_thr: float = DEFAULT_SCORE_THRESHOLD,
         default_nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
         default_save_crops: bool = False, default_crop_dir: Path | None = None,
         default_warmup_runs: int = 0,
         default_expand_ratio_w: float = 0.0, default_expand_ratio_h: float = 0.0):
    """解析参数并执行 ONNX CPU 推理；输入尺寸来自模型元数据。"""
    parser = build_parser(
        description=description, default_model_path=default_model_path,
        default_image_path=default_image_path, default_output_dir=default_output_dir,
        default_backend=default_backend, default_score_thr=default_score_thr,
        default_nms_iou=default_nms_iou, default_save_crops=default_save_crops,
        default_crop_dir=default_crop_dir, default_warmup_runs=default_warmup_runs)
    args = parser.parse_args(argv)
    validate_args(parser, args)

    print("=" * 60)
    print("大图检测 ONNX CPU 推理")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  输入: {args.image_path}")
    print(f"  输出: {args.output_dir}")
    print(f"  后端: {args.backend}")
    print(f"  阈值: score={args.score_thr}, nms_iou={args.nms_iou}")
    print(f"  裁片: {args.crop_dir if args.save_crops else '关闭'}")
    print(f"  预热: {args.warmup_runs} 次")
    print()

    try:
        backend = load_backend(args.model_path, args.backend)
        dynamic, max_input_size, fixed_input_size = exported_shape(backend)
    except (ImportError, ValueError) as error:
        parser.error(str(error))
        return

    if dynamic:
        print(f"  输入尺寸上限: {max_input_size[1]}×{max_input_size[0]} (宽×高)，动态高宽")
    else:
        print(f"  输入尺寸: {fixed_input_size[1]}×{fixed_input_size[0]} (宽×高)，固定高宽")
    print()
    run_inference(
        backend, args.image_path, args.output_dir,
        dynamic=dynamic, max_input_size=max_input_size,
        fixed_input_size=fixed_input_size, threshold=args.score_thr,
        iou_threshold=args.nms_iou, save_crops=args.save_crops,
        crop_dir=args.crop_dir, warmup_runs=args.warmup_runs,
        expand_ratio_w=default_expand_ratio_w, expand_ratio_h=default_expand_ratio_h)
