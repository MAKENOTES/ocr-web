"""统一检测 CPU 部署入口：ONNX Runtime、OpenVINO 或 PyTorch 批量推理。"""
from pathlib import Path

from detection.inference import run_inference


MODEL_PATH = Path("best_detection_unified.onnx")  # 统一检测 ONNX；也可改为 .pth 并使用 auto/pytorch。
BACKEND = "auto"  # auto 按后缀选择 ONNX Runtime 或 PyTorch；openvino 需显式选择。
WARMUP_RUNS = 3  # 正式计时前的模型前向预热次数，0 表示关闭。
EXPAND_BOX = False  # 启用时扩展最终框；默认关闭，避免表格相邻单元格混入裁片。
EXPAND_RATIO_W = 0.10  # 左右各扩展框高的比例，仅 EXPAND_BOX 启用时使用。
EXPAND_RATIO_H = 0.10  # 上下各扩展框高的比例，仅 EXPAND_BOX 启用时使用。
IMAGE_PATH = Path("detection/data/test_images")  # 输入图片或目录，不直接读取 PDF。
OUTPUT_DIR = Path("detection/data/test_results/onnx")  # 画框图与标签目录，重复运行覆盖同名结果。
SCORE_THRESHOLD = 0.7  # 文字置信度阈值，范围 (0, 1)。
NMS_IOU_THRESHOLD = 0.3  # 重复框抑制的交并比阈值，范围 (0, 1]。
SAVE_CROPS = False  # 启用时按原图文字框保存 PNG 裁片。
CROP_DIR = Path("detection/data/crops/onnx")  # 裁片平铺目录，覆盖同名裁片并清理该图多余旧序号。


def main():
    """读取顶部常量执行 CPU 推理；输入尺寸由 inference.py 的常量控制。"""
    run_inference(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR, "cpu",
                  threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD,
                  save_crops=SAVE_CROPS, crop_dir=CROP_DIR, backend=BACKEND,
                  warmup_runs=WARMUP_RUNS,
                  expand_ratio_w=EXPAND_RATIO_W if EXPAND_BOX else 0.0,
                  expand_ratio_h=EXPAND_RATIO_H if EXPAND_BOX else 0.0)


if __name__ == "__main__":
    main()
