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
from recognition.components.dataset import (
    InferenceDataset,
    dynamic_padding_collate_fn,
    resolve_image_path,
)
from recognition.utils.checkpoint import extract_state_dict, infer_num_channels
from recognition.utils.annotation import split_annotation_fields

# ================= 配置区域 =================
MODEL_PATH = "recognition/checkpoints/1788561997best_cer_0.0152_epoch_610.pth"  # 修改为你的最佳模型路径
KEYS_PATH = "recognition/keys.txt"
# VAL_ANNOTATION = "recognition/data/train_annotation.txt"
# VAL_ANNOTATION = "recognition/data/val_annotation.txt"
VAL_ANNOTATION = "recognition/data/append_annotation.txt"

ERROR_OUTPUT_FILE = "recognition/error_cases.txt"  # 错误案例保存文件（可设为 None）

IMG_HEIGHT = 32
DEFAULT_NUM_CHANNELS = 3
BATCH_SIZE = 64
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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)
    num_channels = infer_num_channels(state_dict, default=DEFAULT_NUM_CHANNELS)
    model = CRNN(img_height=IMG_HEIGHT, nc=num_channels, n_class=converter.num_classes).to(DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    return model, converter, num_channels


def create_evaluation_dataset(image_paths, num_channels=DEFAULT_NUM_CHANNELS):
    """创建与训练、推理一致的无裁剪 detection 图片数据集。"""
    return InferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=num_channels,
    )


def main():
    # 严格按标注路径读取图片，并建立 stem -> label 映射。
    filename_to_label = {}
    image_paths = []
    skipped_empty_labels = 0
    with open(VAL_ANNOTATION, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            path_part, label_part = split_annotation_fields(
                line, VAL_ANNOTATION, line_number
            )
            label = label_part.strip()
            if not label:
                skipped_empty_labels += 1
                continue

            annotated_path = os.path.join(
                os.path.dirname(VAL_ANNOTATION),
                path_part,
            )
            resolved_path = resolve_image_path(annotated_path)
            image_id = os.path.splitext(os.path.basename(resolved_path))[0]
            filename_to_label[image_id] = label
            image_paths.append(resolved_path)

    print(f"加载了 {len(filename_to_label)} 条标注记录")
    if skipped_empty_labels:
        print(f"已跳过 {skipped_empty_labels} 条空标签样本")

    model, converter, num_channels = load_network()

    dataset = create_evaluation_dataset(image_paths, num_channels=num_channels)
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
            pred_texts = converter.decode(preds.cpu(), layout="TBC")

            for path, pred in zip(paths, pred_texts):
                filename = os.path.basename(path)
                image_id = os.path.splitext(filename)[0]
                gt = filename_to_label.get(image_id, "")

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
