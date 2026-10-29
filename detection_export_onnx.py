"""统一检测 ONNX 导出入口：按顶部常量读出检查点并生成可部署模型。"""
from pathlib import Path

from detection.export_onnx import export_model


CHECKPOINT_PATH = Path("detection/checkpoints/best_unified.pth")  # 待导出的统一检测检查点，必须包含 config 和 model。
OUTPUT_PATH = Path("best_detection_unified.onnx")  # 导出 ONNX 文件路径；目标已存在时拒绝覆盖。


def main():
    """按顶部路径常量导出，数值校验通过后保存新模型。"""
    config = export_model(CHECKPOINT_PATH, OUTPUT_PATH)
    print(f"导出及数值校验通过: {OUTPUT_PATH}; {config}")


if __name__ == "__main__":
    main()
