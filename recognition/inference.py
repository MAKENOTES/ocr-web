"""按顶部常量执行 PyTorch 识别，共享 RGB 加载、去章及动态补边。"""

import os
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# -- 动态添加项目根目录到 sys.path --
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
# converter: 用于字符解码
from recognition.components.converter import LabelConverter
# model: 定义网络结构
from recognition.models.model import CRNN
# dataset: 复用训练侧的 PNG/RGB 预处理与动态 Padding
from recognition.components.dataset import InferenceDataset, dynamic_padding_collate_fn
from recognition.utils.checkpoint import extract_state_dict, infer_num_channels

# 配置区域 (常量)
# 1. 核心路径
MODEL_PATH = os.path.join(project_root, "recognition/checkpoints/best_cer_0.004061_epoch_603.pth")  # 训练好的模型路径
KEYS_PATH = os.path.join(project_root, "recognition/keys.txt")  # 字典路径 (必须与训练一致)
INPUT_PATH = os.path.join(project_root, "recognition/data/test_images")  # 待推理的图片文件夹 (或单张图片路径)
OUTPUT_FILE = None  # 结果保存路径 (设为 None 则只打印)

# 2. 模型参数
IMG_HEIGHT = 32  # 固定高度
DEFAULT_NUM_CHANNELS = 3  # 新训练使用 RGB；旧 checkpoint 会自动识别为 1 通道

# 3. 推理参数
BATCH_SIZE = 64  # 批量大小 (显存允许可调大，例如 64, 128)
NUM_WORKERS = 2  # 数据加载子进程数；0 表示主进程加载。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有可用 CUDA 时使用 GPU，否则 CPU。


def load_network():
    """
    加载模型和字典
    """
    if not os.path.exists(KEYS_PATH):
        raise FileNotFoundError(f"字典文件未找到: {KEYS_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

    print(f"正在初始化网络 (Device: {DEVICE})...")

    # 1. 加载字典 (获取类别数)
    converter = LabelConverter(KEYS_PATH)

    # 2. 先读取权重，并从第一层卷积自动判断灰度/RGB。
    print(f"加载权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)
    num_channels = infer_num_channels(state_dict, default=DEFAULT_NUM_CHANNELS)

    # 3. 初始化模型结构
    model = CRNN(
        img_height=IMG_HEIGHT,
        nc=num_channels,
        n_class=converter.num_classes
    ).to(DEVICE)

    model.load_state_dict(state_dict)

    model.eval()
    print(f"输入通道: {num_channels} ({'RGB' if num_channels == 3 else '灰度'})")
    return model, converter, num_channels


def main():
    # 1. 收集图片路径
    image_paths = []
    if os.path.isdir(INPUT_PATH):
        # 支持常见图片格式
        types = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp')
        for t in types:
            image_paths.extend(glob.glob(os.path.join(INPUT_PATH, t)))
            image_paths.extend(glob.glob(os.path.join(INPUT_PATH, t.upper())))
    elif os.path.isfile(INPUT_PATH):
        image_paths = [INPUT_PATH]

    if not image_paths:
        print(f"错误: 在 {INPUT_PATH} 未找到任何图片。")
        return

    image_paths = sorted(set(image_paths))
    print(f"发现 {len(image_paths)} 张待处理图片。")

    # 2. 加载模型与转换器
    model, converter, num_channels = load_network()

    # 3. 构建 DataLoader
    # 输入为完整 detection 框；共享局部去红章与缩放，不二次裁剪正文。
    dataset = InferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=num_channels,
    )

    # 使用 dynamic_padding_collate_fn 实现 Batch 内动态宽度的对齐
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=dynamic_padding_collate_fn
    )

    # 4. 执行推理
    results = []
    print(f"开始推理 (Batch Size: {BATCH_SIZE})...")

    with torch.no_grad():
        # loader 返回补边图片、路径和不含保护区的真实宽度。
        for images, paths, img_widths in tqdm(loader, unit="batch"):
            images = images.to(DEVICE)
            img_widths = img_widths.to(DEVICE)

            # 前向传播 -> (SeqLen, Batch, NumClasses)
            preds = model(images, img_widths=img_widths)

            # 解码 -> List[str]
            # converter.decode 直接取 argmax，再合并连续重复和移除 blank。
            pred_texts = converter.decode(preds.cpu(), layout="TBC")

            # 收集结果
            for path, text in zip(paths, pred_texts):
                filename = os.path.basename(path)
                results.append((filename, text))

    # 5. 输出与保存
    print("-" * 50)
    if OUTPUT_FILE:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for filename, text in results:
                    line = f"{filename}\t{text}"
                    f.write(line + "\n")
                    # 可选: 打印到控制台以便查看
                    # print(line)
            print(f"成功: 全部结果已保存至 -> {OUTPUT_FILE}")
        except Exception as e:
            print(f"保存文件失败: {e}")
            # 如果保存失败，兜底打印到控制台
            for filename, text in results:
                print(f"{filename}\t{text}")
    else:
        # 如果没有配置输出文件，直接打印
        print("识别结果:")
        for filename, text in results:
            print(f"{filename}\t{text}")


if __name__ == "__main__":
    main()
