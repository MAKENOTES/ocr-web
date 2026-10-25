"""训练 RGB CRNN；训练循环、保存策略与输出目录由本文件控制。

数据路径、输入尺寸和 DataLoader 构建见 training_data.py；确定性预处理见
components/preprocessing.py，随机增强见 components/augmentation.py。"""

import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import Levenshtein
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -- 动态添加项目根目录到 sys.path ---
# 确保能找到 recognition 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.components.converter import LabelConverter
from recognition.models.model import CRNN
from recognition.training_data import IMG_HEIGHT, NUM_CHANNELS, create_training_loaders

# 配置区域
# 1. 路径配置
KEYS_PATH = "recognition/keys.txt"  # 当前权重绑定的字符字典，内容和顺序决定类别编号。
CHECKPOINT_DIR = "recognition/checkpoints/"  # 固定检查点根目录，每次运行在其下新建独立目录。
LOG_DIR = "recognition/runs"  # 固定 TensorBoard 根目录，每次运行独立建目录。

# 2. 训练超参数
EPOCHS = 680  # 目标总轮数 680。
LEARNING_RATE = 3e-4  # 新运行的 AdamW 初始学习率。
RANDOM_SEED = 42  # 新运行的 Python、NumPy、PyTorch 随机种子。

# 3. 保存与验证策略
SAVE_INTERVAL = 5  # 第 START_SAVE_EPOCH 轮起，每 5 轮保存周期备份；last.pth 每轮保存。
START_VALID_EPOCH = 1    # 从第1轮开始验证，学习率调度优先使用验证集 CER
START_SAVE_EPOCH = 100    # 周期备份和 Top-K 从第100轮开始；last.pth 每轮保存
KEEP_TOP_K = 3           # 保留保存阶段内验证集 CER 最低的 3 个最佳模型

# 4. 硬件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有可用 CUDA 时使用 GPU，否则 CPU。


def prepare_run_directories():
    """新建本次运行的检查点和 TensorBoard 目录。"""
    run_name = f"ocr_exp_{time.time_ns()}"
    checkpoint_dir = (Path(CHECKPOINT_DIR) / run_name).resolve()
    log_dir = (Path(LOG_DIR) / run_name).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    return checkpoint_dir, log_dir


def atomic_save(payload, destination):
    """先写同目录临时文件，再替换目标，避免中断损坏已有检查点。"""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_epoch(model, keys, checkpoint_dir, best_checkpoints, epoch, metric, has_validation):
    """保存本轮检查点，并维护本次运行的 Top-K 最佳模型。"""
    best_path = None
    removed = None
    if epoch >= START_SAVE_EPOCH and has_validation and np.isfinite(metric):
        if len(best_checkpoints) < KEEP_TOP_K or metric < best_checkpoints[-1]["metric"]:
            best_path = checkpoint_dir / f"best_cer_{metric:.6f}_epoch_{epoch}.pth"
            best_checkpoints.append({"path": str(best_path), "metric": metric, "epoch": epoch})
            best_checkpoints.sort(key=lambda item: item["metric"])
            if len(best_checkpoints) > KEEP_TOP_K:
                removed = Path(best_checkpoints.pop()["path"])

    payload = {"state_dict": model.state_dict(), "epoch": epoch, "keys": list(keys)}
    if best_path is not None:
        atomic_save(payload, best_path)
        print(f"  [保存] 最佳模型: {best_path}")
    if epoch >= START_SAVE_EPOCH and epoch % SAVE_INTERVAL == 0:
        atomic_save(payload, checkpoint_dir / f"epoch_{epoch}.pth")
    atomic_save(payload, checkpoint_dir / "last.pth")
    if removed is not None:
        removed.unlink(missing_ok=True)


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

    for p in model.parameters():
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2)
            total_norm += grad_norm.item() ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())

    total_norm = total_norm ** 0.5

    writer.add_scalar("Grad/Total_L2_Norm", total_norm, global_step)
    writer.add_scalar("Grad/Max_Abs_Grad", max_grad, global_step)


def train():
    if EPOCHS < 1 or not math.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0:
        raise ValueError("EPOCHS and LEARNING_RATE must be positive")
    print(f"当前使用的计算设备: {DEVICE}")
    print(f"输入预处理: 完整 detection 框 / {'RGB' if NUM_CHANNELS == 3 else '灰度'} / 不裁剪")
    print("序列处理: CNN 右侧背景保护区 / 长度感知 BiGRU / batch padding 不参与识别")

    # 1. 初始化 Converter
    converter = LabelConverter(KEYS_PATH)
    num_classes = converter.num_classes
    print(f"字典加载完成，类别数量: {num_classes} (含 blank/unk)")

    # 2. 初始化模型；固定随机种子后创建参数，保证同配置运行可复现。
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
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

    # 5. 数据加载器
    train_loader, valid_loader = create_training_loaders()

    # 6. 本次运行的输出目录与日志
    checkpoint_dir, log_dir = prepare_run_directories()
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"检查点目录: {checkpoint_dir}")
    print(f"TensorBoard 目录: {log_dir}")
    print(f"准备开始训练 - Epoch 1 至 {EPOCHS}")
    print(f"  - 验证开始: Epoch {START_VALID_EPOCH}")
    print(f"  - 每轮保存 last.pth；周期备份 / Top-K 开始: Epoch {START_SAVE_EPOCH}")

    best_checkpoints = []
    global_step = 0
    try:
        for epoch in range(1, EPOCHS + 1):
            # 训练阶段
            model.train()
            total_train_loss = 0
            train_batch_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", unit="batch")

            for images, labels, img_widths in pbar:
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
            save_epoch(model, converter.keys, checkpoint_dir, best_checkpoints, epoch,
                       scheduler_metric, valid_loader is not None)
            writer.flush()

    finally:
        writer.close()
    print("训练完成。")

if __name__ == "__main__":
    train()
