import os
import sys
import time
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
# ------------------------------------

from recognition.components.converter import LabelConverter
from recognition.components.dataset import create_dataloader
from recognition.models.model import CRNN
import Levenshtein

# ================= 配置区域 =================
# 1. 路径配置
TRAIN_ANNOTATION = (
    "recognition/data/train_annotation.txt",
    # "recognition/data/append_annotation.txt",
    # "recognition/data/bench1_annotation.txt",
    # "recognition/data/bench2_annotation.txt",
    "recognition/data/synthesis_annotation.txt",
)
VALID_ANNOTATION = "recognition/data/val_annotation.txt"
KEYS_PATH = "recognition/keys.txt"
CHECKPOINT_DIR = "recognition/checkpoints/"

# 2. 数据与模型参数
IMG_HEIGHT = 32
IMG_WIDTH = 1800     # 当前数据等比缩放后的最大宽度约 1733，留少量余量
NUM_CHANNELS = 3     # 保留白/红/黑及彩色 UI 文本的色彩对比

# 3. 训练超参数
BATCH_SIZE = 64     # 适当调整
EPOCHS = 680
LEARNING_RATE = 3e-4
NUM_WORKERS = 4

# 4. 保存与验证策略
SAVE_INTERVAL = 5        # 每隔多少轮强制保存一次
START_VALID_EPOCH = 1    # 【建议】从第1轮就开始验证，以便 Scheduler 正常工作
START_SAVE_EPOCH = 100    # 从第100轮才开始保存模型 (跳过前期不稳定阶段)
KEEP_TOP_K = 3           # 只保留 Loss 最小的 3 个最佳模型

# 5. 硬件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, converter, criterion):
    """
    验证函数：计算 准确率(Accuracy) 和 验证集损失(Loss)
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0
    batch_count = 0
    total_char_distance = 0  # 新增
    total_chars = 0  # 新增

    with torch.no_grad():
        for images, labels, img_widths in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            img_widths = img_widths.to(DEVICE)

            # 1. 前向传播（传入真实宽度，mask padding 区域）
            preds = model(images, img_widths=img_widths)  # Shape: [SeqLen, Batch, NumClasses]

            # 2. 计算验证 Loss (用于 Scheduler)
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
    cer = total_char_distance / total_chars if total_chars > 0 else 1.0  # 新增

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


def train():
    print(f"当前使用的计算设备: {DEVICE}")
    print(f"输入预处理: 完整 detection 框 / {'RGB' if NUM_CHANNELS == 3 else '灰度'} / 不裁剪")
    print("序列处理: CNN 右侧背景保护区 / 长度感知 BiGRU / batch padding 不参与识别")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # TensorBoard 日志
    writer = SummaryWriter(log_dir=f"recognition/runs/ocr_exp_{int(time.time())}")

    # 1. 初始化 Converter
    converter = LabelConverter(KEYS_PATH)
    num_classes = converter.num_classes
    print(f"字典加载完成，类别数量: {num_classes} (含 blank/unk)")

    # 2. 初始化模型
    # 【注意】请确保 model.py 中的 nn.Linear 输出维度已修正为 n_class
    model = CRNN(
        img_height=IMG_HEIGHT,
        nc=NUM_CHANNELS,
        n_class=num_classes,
    ).to(DEVICE)

    # 3. 损失函数
    # zero_infinity=True 可以防止长文本导致的 Loss 无穷大导致梯度爆炸
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)

    # 4. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 学习率调度器：监控 'min' 模式（Loss越小越好）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-6
    )

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
            img_width=IMG_WIDTH,  # 验证集不需要太宽，除非有长图
            num_channels=NUM_CHANNELS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            is_train=False,
        )
    else:
        print("⚠️ 警告: 未找到验证集标注文件，将无法评估 Accuracy 和进行最佳模型保存。")

    print(f"准备开始训练 - 总 Epochs: {EPOCHS}")
    print(f"  - 验证开始: Epoch {START_VALID_EPOCH}")
    print(f"  - 保存开始: Epoch {START_SAVE_EPOCH}")

    global_step = 0
    # 用于存储最佳模型记录: [{'path': str, 'loss': float, 'epoch': int}, ...]
    best_checkpoints = []

    for epoch in range(1, EPOCHS + 1):
        # ================= 训练阶段 =================
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

        # ================= 验证与调度阶段 =================
        # 决定使用哪个 Loss 来更新 Scheduler
        scheduler_metric = train_avg_loss

        if valid_loader and epoch >= START_VALID_EPOCH:
            val_acc, val_cer, val_loss = evaluate(model, valid_loader, converter, criterion)

            print(f"Epoch {epoch} | Valid Acc: {val_acc:.4f} | Valid CER: {val_cer:.4f} | Valid Loss: {val_loss:.4f}")
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)
            writer.add_scalar('Validation/CER', val_cer, epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)

            scheduler_metric = val_cer

        # 更新学习率
        scheduler.step(scheduler_metric)

        # ================= 保存模型阶段 =================
        if epoch >= START_SAVE_EPOCH:

            # 策略 1: 周期性保存 (Backup)
            if epoch % SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [保存] 周期性备份: {ckpt_path}")

            # 策略 2: Top-K 最佳模型 (基于 Valid Loss)
            if valid_loader:
                # 判断是否应该保存为“最佳模型”
                should_save_best = False

                if len(best_checkpoints) < KEEP_TOP_K:
                    should_save_best = True
                elif scheduler_metric < best_checkpoints[-1]['metric']:
                    # 当前 Loss 小于列表中最差的一个
                    should_save_best = True

                if should_save_best:
                    best_name = f"{int(time.time())}best_cer_{scheduler_metric:.4f}_epoch_{epoch}.pth"
                    best_path = os.path.join(CHECKPOINT_DIR, best_name)

                    torch.save(model.state_dict(), best_path)
                    print(f"  [保存] 🌟 发现新最佳模型 ({scheduler_metric:.4f}) -> {best_path}")

                    # 更新列表
                    best_checkpoints.append({'path': best_path, 'metric': scheduler_metric, 'epoch': epoch})
                    # 按 metric 从小到大排序
                    best_checkpoints.sort(key=lambda x: x['metric'])

                    # 移除多余的旧模型
                    if len(best_checkpoints) > KEEP_TOP_K:
                        worst_ckpt = best_checkpoints.pop() # 弹出最后一个(Loss最大的)
                        if os.path.exists(worst_ckpt['path']):
                            try:
                                os.remove(worst_ckpt['path'])
                                print(f"  [清理] 已移除旧模型: {worst_ckpt['path']}")
                            except OSError as e:
                                print(f"  [警告] 移除文件失败: {e}")

    # 训练结束
    writer.close()
    print("训练完成。")

if __name__ == "__main__":
    train()
