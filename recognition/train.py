"""训练 RGB CRNN；超参数由顶部常量控制，续训与保存由 TrainingRun 管理。"""

import os
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# -- 动态添加项目根目录到 sys.path ---
# 确保能找到 recognition 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.components.converter import LabelConverter
from recognition.components.dataset import create_dataloader
from recognition.models.model import CRNN
from recognition.utils.training_run import TrainingRun, build_training_config, parse_args
import Levenshtein

# 配置区域
# 1. 路径配置
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
KEYS_PATH = "recognition/keys.txt"  # 当前权重绑定的字符字典，内容和顺序决定类别编号。
CHECKPOINT_DIR = "recognition/checkpoints/"  # 固定检查点根目录，TrainingRun 在其下新建本次运行目录。
LOG_DIR = "recognition/runs"  # 固定 TensorBoard 根目录，每次运行独立建目录。

# 2. 数据与模型参数
IMG_HEIGHT = 32  # 输入图片高度，单位 px；当前 CRNN 只支持 32。
IMG_WIDTH = 1800  # 缩放宽度上限 1800 px；训练和训练期验证超过此值会横向压缩。
NUM_CHANNELS = 3     # 保留白/红/黑及彩色 UI 文本的色彩对比

# 3. 训练超参数
BATCH_SIZE = 64  # 每批图片数，末批不足 64 张仍保留。
EPOCHS = 680  # 目标总轮数 680，续训也使用该总数。
LEARNING_RATE = 3e-4  # 新运行的 AdamW 初始学习率；完整续训恢复检查点中的学习率。
NUM_WORKERS = 4  # 数据加载子进程数；0 表示主进程加载。
RANDOM_SEED = 42  # 新运行的 Python、NumPy、PyTorch 随机种子。

# 4. 保存与验证策略
SAVE_INTERVAL = 5  # 第 START_SAVE_EPOCH 轮起，每 5 轮保存周期备份；last.pth 每轮保存。
START_VALID_EPOCH = 1    # 从第1轮开始验证，学习率调度优先使用验证集 CER
START_SAVE_EPOCH = 100    # 周期备份和 Top-K 从第100轮开始；last.pth 每轮保存
KEEP_TOP_K = 3           # 保留保存阶段内验证集 CER 最低的 3 个最佳模型

# 5. 硬件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有可用 CUDA 时使用 GPU，否则 CPU。


def evaluate(model, dataloader, converter, criterion):
    """
    验证函数：计算整条文本准确率、字符错误率 CER 和平均 CTC Loss。
    CER 用于学习率调度和最佳模型筛选；Accuracy 与 Loss 仅展示和记录。
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0
    batch_count = 0
    total_char_distance = 0  # 累计字符编辑距离。
    total_chars = 0  # 累计去首尾空白后的真值字符数。

    with torch.no_grad():
        for images, labels, img_widths in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            img_widths = img_widths.to(DEVICE)

            # 1. 前向传播（传入真实宽度，mask padding 区域）
            preds = model(images, img_widths=img_widths)  # Shape: [SeqLen, Batch, NumClasses]

            # 2. 计算验证 CTC Loss（仅展示和记录，不用于学习率调度）
            text, length = converter.encode(labels)
            text = text.to(DEVICE)
            length = length.to(DEVICE)

            # 用模型的统一公式计算各图片有效时间步（而非统一用 batch 最大值）。
            preds_size = model._calc_cnn_output_widths(img_widths).clamp(min=1).int()

            loss = criterion(preds, text, preds_size, length)
            # 处理 NaN 或 Inf
            if torch.isfinite(loss):
                total_loss += loss.item()
                batch_count += 1

            # 3. 计算 Accuracy
            # 解码预测结果
            decoded_preds = converter.decode(preds.cpu(), layout="TBC")

            total_samples += len(labels)
            for pred, gt in zip(decoded_preds, labels):
                # 移除首尾空白符进行对比
                p, g = pred.strip(), gt.strip()
                if p == g:
                    correct_predictions += 1
                # 计算编辑距离 (CER)
                total_char_distance += Levenshtein.distance(p, g)
                total_chars += len(g)

    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    cer = total_char_distance / total_chars if total_chars > 0 else 1.0  # 无真值字符时使用失败口径。

    return accuracy, cer, avg_loss

def log_gradient_norms(model, writer, global_step):
    total_norm = 0.0
    max_grad = 0.0
    param_count = 0

    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2)
            total_norm += grad_norm.item() ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())
            param_count += 1

    total_norm = total_norm ** 0.5

    writer.add_scalar("Grad/Total_L2_Norm", total_norm, global_step)
    writer.add_scalar("Grad/Max_Abs_Grad", max_grad, global_step)


def train(argv=None):
    args = parse_args(argv if argv is not None else [])
    if EPOCHS < 1 or not math.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0:
        raise ValueError("EPOCHS and LEARNING_RATE must be positive")
    print(f"当前使用的计算设备: {DEVICE}")
    print(f"输入预处理: 完整 detection 框 / {'RGB' if NUM_CHANNELS == 3 else '灰度'} / 不裁剪")
    print("序列处理: CNN 右侧背景保护区 / 长度感知 BiGRU / batch padding 不参与识别")

    # 1. 初始化 Converter
    converter = LabelConverter(KEYS_PATH)
    config = build_training_config(
        train_annotation=TRAIN_ANNOTATION, valid_annotation=VALID_ANNOTATION,
        img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_channels=NUM_CHANNELS,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, start_valid_epoch=START_VALID_EPOCH,
        start_save_epoch=START_SAVE_EPOCH, save_interval=SAVE_INTERVAL, keep_top_k=KEEP_TOP_K,
    )
    run = TrainingRun(args, config, converter.keys, CHECKPOINT_DIR,
                      epochs=EPOCHS, seed=RANDOM_SEED, log_root=LOG_DIR)
    if run.complete:
        return
    num_classes = converter.num_classes
    print(f"字典加载完成，类别数量: {num_classes} (含 blank/unk)")

    # 2. 初始化模型
    model = CRNN(
        img_height=IMG_HEIGHT,
        nc=NUM_CHANNELS,
        n_class=num_classes,
    ).to(DEVICE)

    # 3. 损失函数
    # 无法对齐的 CTC 无穷损失置零；不替代有效时间步与标签长度的校验。
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)

    # 4. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 学习率调度器：验证集 CER 越小越好；未执行验证时回退到训练平均 Loss。
    # 连续 21 轮未达到改善阈值时学习率乘以 0.8，最低为 1e-6。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-6
    )
    run.restore(model, optimizer, scheduler)

    # 5. 数据加载器
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

    valid_loader = None
    if VALID_ANNOTATION and os.path.exists(VALID_ANNOTATION):
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
    else:
        print("⚠️ 警告: 未找到验证集标注文件，将无法评估 Accuracy 和进行最佳模型保存。")

    writer = run.open_writer()
    global_step = run.global_step
    try:
        run.restore_random_state(optimizer)
        for epoch in range(run.start_epoch, EPOCHS + 1):
            # 训练阶段
            model.train()
            total_train_loss = 0
            train_batch_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", unit="batch")

            for i, (images, labels, img_widths) in enumerate(pbar):
                images = images.to(DEVICE)
                img_widths = img_widths.to(DEVICE)

                # 1. Forward（真实宽度控制 PackedSequence，BiGRU 不读取 batch padding）
                preds = model(images, img_widths=img_widths)  # [SeqLen, Batch, NumClasses]

                # 2. Encode Labels
                text, length = converter.encode(labels)
                text = text.to(DEVICE)
                length = length.to(DEVICE)

                # 与模型 PackedSequence 使用完全相同的有效长度公式。
                preds_size = model._calc_cnn_output_widths(img_widths).clamp(min=1).int()

                # 3. Compute Loss
                loss = criterion(preds, text, preds_size, length)

                # 4. Backward
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸  max_norm 由5降到2  比较稳定，常见推荐值
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                # 🔍 梯度监控（每 N step 记录一次，防止 TB 太大）
                if global_step % 50 == 0:
                    log_gradient_norms(model, writer, global_step)

                optimizer.step()

                # Logging
                if torch.isfinite(loss):
                    current_loss = loss.item()
                    total_train_loss += current_loss
                    train_batch_count += 1

                    pbar.set_postfix(loss=f"{current_loss:.4f}", lr=optimizer.param_groups[0]['lr'])
                    writer.add_scalar('Train/Batch_Loss', current_loss, global_step)

                global_step += 1

            train_avg_loss = total_train_loss / train_batch_count if train_batch_count > 0 else float('inf')
            print(f"Epoch {epoch} | Train Loss: {train_avg_loss:.4f}")
            writer.add_scalar('Train/Epoch_Loss', train_avg_loss, epoch)

            # 验证与调度阶段
            # 默认用训练平均 Loss 调度；本轮执行验证后改用验证集 CER。
            scheduler_metric = train_avg_loss

            if valid_loader and epoch >= START_VALID_EPOCH:
                val_acc, val_cer, val_loss = evaluate(model, valid_loader, converter, criterion)

                print(f"Epoch {epoch} | Valid Acc: {val_acc:.4f} | Valid CER: {val_cer:.4f} | Valid Loss: {val_loss:.4f}")
                writer.add_scalar('Validation/Accuracy', val_acc, epoch)
                writer.add_scalar('Validation/CER', val_cer, epoch)
                writer.add_scalar('Validation/Loss', val_loss, epoch)

                scheduler_metric = val_cer

            # 本轮结束时更新学习率，从下一轮训练开始生效。
            scheduler.step(scheduler_metric)

            writer.add_scalar("Train/Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            run.save_epoch(model, optimizer, scheduler, epoch, global_step,
                           scheduler_metric, valid_loader is not None)
            writer.flush()

    finally:
        writer.close()
    print("训练完成。")

if __name__ == "__main__":
    train(sys.argv[1:])
