import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.components.converter import LabelConverter
from recognition.models.model import CRNN
from recognition.components.dataset import dynamic_padding_collate_fn
from recognition.utils.white_text_crop import WhiteTextCropInferenceDataset

# ================= 配置区域 =================
MODEL_PATH = "recognition/checkpoints/1783804842best_cer_0.0030_epoch_652.pth"  # 修改为你的最佳模型路径
KEYS_PATH = "recognition/keys.txt"
VAL_ANNOTATION = "recognition/data/train_annotation.txt"  # 验证集标注文件

ERROR_OUTPUT_FILE = "recognition/error_cases.txt"  # 错误案例保存文件（可设为 None）
CROP_DEBUG_DIR = os.path.join(project_root, "recognition/data/crops/evaluate_val")

IMG_HEIGHT = 32
NUM_CHANNELS = 1
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def levenshtein_distance(s1: str, s2: str) -> int:
    """编辑距离，用于计算 CER"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def load_network():
    converter = LabelConverter(KEYS_PATH)
    model = CRNN(img_height=IMG_HEIGHT, nc=NUM_CHANNELS, n_class=converter.num_classes).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, converter


def create_evaluation_dataset(image_paths, save_crops_dir=CROP_DEBUG_DIR):
    """创建与训练、推理使用相同白字自动裁剪逻辑的评估数据集。"""
    return WhiteTextCropInferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=NUM_CHANNELS,
        save_crops_dir=save_crops_dir,
    )


def main():
    # 读取标注文件，建立 filename -> label 的映射
    filename_to_label = {}
    with open(VAL_ANNOTATION, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                path_part, label_part = line.strip().split(',', 1)
                filename = os.path.basename(path_part.strip())
                filename_to_label[filename] = label_part.strip().replace('"', '')

    print(f"加载了 {len(filename_to_label)} 条标注记录")

    # 收集图片路径（从标注文件里取，确保顺序一致）
    image_paths = []
    with open(VAL_ANNOTATION, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                path_part = line.strip().split(',', 1)[0].strip()
                image_paths.append(os.path.join(os.path.dirname(VAL_ANNOTATION), path_part))

    model, converter = load_network()

    dataset = create_evaluation_dataset(image_paths)
    print(f"已启用白字/有色背景自动裁剪，裁剪图保存至: {CROP_DEBUG_DIR}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=dynamic_padding_collate_fn)

    correct = 0
    total_char_distance = 0   # 所有样本的编辑距离之和
    total_chars = 0           # 所有样本的GT字符总数
    error_cases = []

    print("开始评估...")

    with torch.no_grad():
        for images, paths, img_widths in tqdm(loader):
            images = images.to(DEVICE)
            img_widths = img_widths.to(DEVICE)
            preds = model(images, img_widths=img_widths)
            pred_texts = converter.decode(preds.cpu())

            for path, pred in zip(paths, pred_texts):
                filename = os.path.basename(path)
                gt = filename_to_label.get(filename, "")

                pred = pred.strip()
                gt = gt.strip()

                # 字符级编辑距离（正确样本距离为0，自然参与累加）
                char_dist = levenshtein_distance(gt, pred)
                total_char_distance += char_dist
                total_chars += max(len(gt), 1)

                if char_dist == 0:
                    correct += 1
                else:
                    sample_cer = char_dist / max(len(gt), 1)
                    error_cases.append({
                        'filename': filename,
                        'gt': gt,
                        'pred': pred,
                        'cer': sample_cer
                    })

    # 统计结果
    total = len(image_paths)
    accuracy = correct / total * 100
    avg_cer = (total_char_distance / total_chars) * 100 if total_chars > 0 else 0.0

    print("\n" + "=" * 70)
    print("评估完成！")
    print(f"总样本数       : {total}")
    print(f"全串准确率     : {accuracy:.3f}%")
    print(f"平均字符错误率 : {avg_cer:.3f}%")
    print(f"错误样本数     : {len(error_cases)}")

    if error_cases:
        error_cases.sort(key=lambda x: x['cer'], reverse=True)
        print("\nTop 错误案例 (按 CER 降序):")
        print("-" * 90)
        print(f"{'CER':<7} {'文件名':<25} {'真实标签':<30} {'预测结果'}")
        print("-" * 90)
        for case in error_cases[:30]:  # 前30个最严重错误
            print(f"{case['cer'] * 100:6.2f}%  {case['filename']:<25} {case['gt']:<30} {case['pred']}")

        if ERROR_OUTPUT_FILE:
            with open(ERROR_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(f"Accuracy: {accuracy:.3f}%\nAvg CER: {avg_cer:.3f}%\n\n")
                f.write("CER\t文件名\t真实\t预测\n")
                for case in error_cases:
                    f.write(f"{case['cer'] * 100:.2f}\t{case['filename']}\t{case['gt']}\t{case['pred']}\n")
            print(f"\n详细错误案例已保存 → {ERROR_OUTPUT_FILE}")


if __name__ == "__main__":
    main()