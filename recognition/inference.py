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

# --- 导入项目模块 ---
# converter: 用于字符解码
from recognition.components.converter import LabelConverter
# model: 定义网络结构
from recognition.models.model import CRNN
# dataset: 复用其中的推理数据集类和动态Padding整理函数
# 注意：前提是你已经按照上一步修改了 dataset.py
from recognition.components.dataset import dynamic_padding_collate_fn
from recognition.utils.white_text_crop import WhiteTextCropInferenceDataset

# ================= 配置区域 (常量) =================
# 1. 核心路径
MODEL_PATH = os.path.join(project_root, "recognition/checkpoints/1783804842best_cer_0.0030_epoch_652.pth")  # 训练好的模型路径
KEYS_PATH = os.path.join(project_root, "recognition/keys.txt")  # 字典路径 (必须与训练一致)
INPUT_PATH = os.path.join(project_root, "recognition/data/test_images1")  # 待推理的图片文件夹 (或单张图片路径)
OUTPUT_FILE = None  # 结果保存路径 (设为 None 则只打印)
CROP_DEBUG_DIR = os.path.join(project_root, "recognition/data/crops")  # 保存自动裁剪结果，便于人工核查

# 2. 模型参数 (需与训练保持一致)
IMG_HEIGHT = 32  # 固定高度
NUM_CHANNELS = 1  # 1=灰度, 3=RGB

# 3. 推理参数
BATCH_SIZE = 64  # 批量大小 (显存允许可调大，例如 64, 128)
NUM_WORKERS = 2  # 数据加载线程数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================

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

    # 2. 初始化模型结构
    model = CRNN(
        img_height=IMG_HEIGHT,
        nc=NUM_CHANNELS,
        n_class=converter.num_classes
    ).to(DEVICE)

    # 3. 加载权重
    print(f"加载权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # 兼容处理 state_dict (有的代码保存整个 checkpoint dict，有的只保存 state_dict)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, converter


def main():
    # 1. 收集图片路径
    image_paths = []
    if os.path.isdir(INPUT_PATH):
        # 支持常见图片格式
        types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
        for t in types:
            image_paths.extend(glob.glob(os.path.join(INPUT_PATH, t)))
            image_paths.extend(glob.glob(os.path.join(INPUT_PATH, t.upper())))
    elif os.path.isfile(INPUT_PATH):
        image_paths = [INPUT_PATH]

    if not image_paths:
        print(f"错误: 在 {INPUT_PATH} 未找到任何图片。")
        return

    print(f"发现 {len(image_paths)} 张待处理图片。")

    # 2. 加载模型与转换器
    model, converter = load_network()

    # 3. 构建 DataLoader
    # 推理默认使用与训练/验证一致的确定性白字裁剪，并保存裁剪图便于核查。
    dataset = WhiteTextCropInferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=NUM_CHANNELS,
        save_crops_dir=CROP_DEBUG_DIR,
    )
    print(f"已启用白字/有色背景自动裁剪，裁剪图保存至: {CROP_DEBUG_DIR}")

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
        # loader 返回: (padded_images, paths)
        for images, paths, img_widths in tqdm(loader, unit="batch"):
            images = images.to(DEVICE)
            img_widths = img_widths.to(DEVICE)

            # 前向传播 -> (SeqLen, Batch, NumClasses)
            preds = model(images, img_widths=img_widths)

            # 解码 -> List[str]
            # converter.decode 内部处理了 Softmax/Argmax 和 CTC 的去重逻辑
            pred_texts = converter.decode(preds.cpu())

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