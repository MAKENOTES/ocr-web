"""小图检测 ONNX CPU 推理入口：只保留默认常量，实现由 detection_small/runtime 提供。

输入尺寸由 ONNX 元数据决定：默认的固定高宽导出始终使用导出尺寸，只有
dynamic_spatial=True 的模型才按原图宽高比选择尺寸。部署时使用
tools/build_detection_small_bundle.py 生成包含 runtime、本入口和 .onnx 模型的
独立目录，runtime 快照由 tools/sync_detection_small_runtime.py 从仓库源文件生成。
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_small.runtime.cli import main as _cli_main


MODEL_PATH = Path("best_detection_small.onnx")  # 已导出的小图检测 ONNX；本入口只接受 .onnx。
BACKEND = "openvino"  # auto 固定使用 ONNX Runtime；openvino 需显式选择。
WARMUP_RUNS = 3  # 正式计时前的模型前向预热次数，0 表示关闭。
EXPAND_BOX = False  # 启用时扩展最终框；默认关闭，避免相邻文字混入裁片。
EXPAND_RATIO_W = 0.10  # 左右各扩展框高的比例，仅 EXPAND_BOX 启用时使用。
EXPAND_RATIO_H = 0.10  # 上下各扩展框高的比例，仅 EXPAND_BOX 启用时使用。
IMAGE_PATH = Path("detection_small/data/test/4")  # 输入图片或目录。
OUTPUT_DIR = Path("detection_small/data/test_results/onnx/4")  # 画框图与标签目录，重复运行覆盖同名结果。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，范围 (0, 1)。
NMS_IOU_THRESHOLD = 0.3  # 重复框抑制的交并比阈值，范围 (0, 1]。
SAVE_CROPS = False  # 启用时按原图文字框保存 PNG 裁片。
CROP_DIR = Path("detection_small/data/crops/onnx")  # 裁片平铺目录，覆盖同名裁片并清理该图多余旧序号。


def main(argv=None):
    """按顶部常量执行 ONNX CPU 推理；命令行参数可覆盖本次输入。"""
    return _cli_main(
        argv,
        description="小图检测 ONNX CPU 推理：ONNX Runtime 或 OpenVINO。",
        default_model_path=MODEL_PATH,
        default_image_path=IMAGE_PATH,
        default_output_dir=OUTPUT_DIR,
        default_backend=BACKEND,
        default_score_thr=SCORE_THRESHOLD,
        default_nms_iou=NMS_IOU_THRESHOLD,
        default_save_crops=SAVE_CROPS,
        default_crop_dir=CROP_DIR,
        default_warmup_runs=WARMUP_RUNS,
        default_expand_ratio_w=EXPAND_RATIO_W if EXPAND_BOX else 0.0,
        default_expand_ratio_h=EXPAND_RATIO_H if EXPAND_BOX else 0.0,
    )


if __name__ == "__main__":
    main()
