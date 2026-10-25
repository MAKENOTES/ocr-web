"""训练数据配置与 DataLoader 构建。

标注路径、输入尺寸和批处理参数集中在这里；train.py 只负责训练循环与保存。
图片读取、去章和缩放见 components/preprocessing.py，随机增强见
components/augmentation.py，数据集与补边见 components/dataset.py。
"""

import os

from recognition.components.dataset import create_dataloader

# 1. 标注路径
# 按顺序合并 train 与八份合成标注；每条路径相对于所属标注文件。
TRAIN_ANNOTATION = (
    "recognition/data/train_annotation.txt",
    "recognition/data/synthesis1_annotation.txt",
    "recognition/data/synthesis2_annotation.txt",
    "recognition/data/synthesis3_annotation.txt",
    "recognition/data/synthesis_red_annotation.txt",
    "recognition/data/synthesis_lines_annotation.txt",
    "recognition/data/synthesis_confusable_annotation.txt",
    "recognition/data/synthesis_degraded_annotation.txt",
    "recognition/data/synthesis_edge_annotation.txt",
)
VALID_ANNOTATION = "recognition/data/val_annotation.txt"  # 独立验证标注；缺文件时当前流程仅训练，不保存最佳 CER 模型。

# 2. 输入与批处理参数
IMG_HEIGHT = 32  # 输入图片高度，单位 px；当前 CRNN 只支持 32。
IMG_WIDTH = 1800  # 缩放宽度上限 1800 px；训练和训练期验证超过此值会横向压缩。
NUM_CHANNELS = 3  # 保留白/红/黑及彩色 UI 文本的色彩对比。
BATCH_SIZE = 64  # 每批图片数，末批不足 64 张仍保留。
NUM_WORKERS = 4  # 数据加载子进程数；0 表示主进程加载。


def create_training_loaders():
    """创建训练 DataLoader 和可选验证 DataLoader。

    验证标注缺失时返回 ``(train_loader, None)``，训练继续但不保存最佳 CER 模型。
    """
    print("正在加载训练集...")
    train_loader = create_dataloader(
        annotation_path=TRAIN_ANNOTATION,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_channels=NUM_CHANNELS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_train=True,
    )

    if not VALID_ANNOTATION or not os.path.exists(VALID_ANNOTATION):
        print("⚠️ 警告: 未找到验证集标注文件，将无法评估 Accuracy 和进行最佳模型保存。")
        return train_loader, None

    print("正在加载验证集...")
    valid_loader = create_dataloader(
        annotation_path=VALID_ANNOTATION,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,  # 训练期验证沿用相同宽度上限，超宽图片会横向压缩。
        num_channels=NUM_CHANNELS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_train=False,
    )
    return train_loader, valid_loader
