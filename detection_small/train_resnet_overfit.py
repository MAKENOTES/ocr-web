import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from components.dataset import OCRDataset
from components.transforms import get_transforms
from models.east_model import EASTModel
from components.loss import EASTLoss
from torch.optim.lr_scheduler import OneCycleLR

# --- 超参数 ---
LEARNING_RATE =  1e-4  #1e-5 1e-4 3e-5
BATCH_SIZE = 4 # 根据你的 GPU 显存调整
EPOCHS = 680
INPUT_SIZE = 640
TRAIN_IMAGE_DIR = "detection_small/data/train_images"
TRAIN_LABEL_DIR = "detection_small/data/train_labels"
VAL_IMAGE_DIR = "detection_small/data/val_images"
VAL_LABEL_DIR = "detection_small/data/val_labels"

# --- 模型保存策略优化 ---
MAX_KEEP_CHECKPOINTS = 20       # 常规检查点：保留最近的 5 个 (用于断点续训)
BEST_MODELS_TO_KEEP = 3        # 最佳检查点：保留验证集 Loss 最低的 Top 3
VAL_INTERVAL = 10              # 前期：每 10 个 epoch 验证一次
START_DENSE_VAL_EPOCH = 50     # 策略切换点：从第 50 epoch 开始每个 epoch 都验证

# --- 过拟合早停参数 ---
OVERFIT_WINDOW = 30            # 用最近 30 次验证记录计算趋势
OVERFIT_PATIENCE = 15          # val_loss 趋势连续 15 次为上升才触发
OVERFIT_GAP_THRESHOLD = 2.0    # gap 扩大倍数阈值：近期 gap 是基准的 2 倍以上


class OverfitEarlyStopping:
    """
    基于过拟合检测的早停机制。

    核心思想：不再因为 val_loss "不下降" 就停止（那只是平台期），
    而是检测真正的过拟合信号 ——
      1. val_loss 的滑动趋势在上升（线性回归斜率 > 0）
      2. 同时 train_loss 的滑动趋势在下降（斜率 < 0）
      3. train/val gap 相比历史基准显著扩大

    三个条件同时满足并持续 patience 轮，才触发早停。
    这样平台期震荡、偶发尖刺都不会误杀训练。
    """

    def __init__(self, window=30, patience=15, gap_threshold=2.0):
        """
        :param window:        用多少次验证记录来计算趋势斜率
        :param patience:      过拟合信号需要连续出现几次才触发停止
        :param gap_threshold: gap 相对于基准期的倍数阈值
        """
        self.window = window
        self.patience = patience
        self.gap_threshold = gap_threshold

        # 历史记录
        self.train_history = []     # 每次验证时的 train_loss
        self.val_history = []       # 每次验证时的 val_loss
        self.gap_history = []       # val_loss - train_loss
        self.epoch_history = []     # 对应的 epoch 号

        # 最佳模型跟踪（仍然保留，用于推荐最佳 checkpoint）
        self.best_loss = None
        self.best_epoch = 0

        # 过拟合计数器
        self.overfit_counter = 0
        self.should_stop = False

        # 基准 gap（用前 window 次验证的平均 gap 作为基准）
        self.baseline_gap = None

    def _linear_slope(self, values):
        """
        对一组数值做简单线性回归，返回斜率。
        斜率 > 0 表示上升趋势，< 0 表示下降趋势。
        """
        n = len(values)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        # 最小二乘法: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        sx = x.sum()
        sy = y.sum()
        sxy = (x * y).sum()
        sx2 = (x * x).sum()
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sxy - sx * sy) / denom

    def __call__(self, val_loss, train_loss, epoch):
        """
        每次验证后调用。
        :return: True 表示检测到过拟合应停止训练, False 表示继续
        """
        self.train_history.append(train_loss)
        self.val_history.append(val_loss)
        self.gap_history.append(val_loss - train_loss)
        self.epoch_history.append(epoch)

        # 更新最佳 val_loss（用于推荐 checkpoint，不影响停止逻辑）
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch

        n = len(self.val_history)

        # 建立基准 gap（取前 window 次验证的均值）
        if self.baseline_gap is None and n >= self.window:
            self.baseline_gap = np.mean(self.gap_history[:self.window])
            print(f"  [OverfitStop] 基准 gap 已建立: {self.baseline_gap:.5f} "
                  f"(基于前 {self.window} 次验证)")

        # 数据不足 window 时，不做判断
        if n < self.window:
            return False

        # ---- 计算三个过拟合信号 ----

        recent_val = self.val_history[-self.window:]
        recent_train = self.train_history[-self.window:]
        recent_gap = self.gap_history[-self.window:]

        # 信号 1: val_loss 趋势上升
        val_slope = self._linear_slope(recent_val)
        val_rising = val_slope > 0

        # 信号 2: train_loss 趋势下降
        train_slope = self._linear_slope(recent_train)
        train_falling = train_slope < 0

        # 信号 3: gap 显著扩大（相比基准）
        current_avg_gap = np.mean(recent_gap)
        if self.baseline_gap is not None and self.baseline_gap > 0:
            gap_ratio = current_avg_gap / self.baseline_gap
        else:
            # 基准 gap 接近 0 或为负（val < train），用绝对值判断
            gap_ratio = 0.0
        gap_widening = gap_ratio > self.gap_threshold

        # ---- 判断是否过拟合 ----
        is_overfitting = val_rising and train_falling and gap_widening

        if is_overfitting:
            self.overfit_counter += 1
            print(f"  ⚠ 过拟合信号 [{self.overfit_counter}/{self.patience}] | "
                  f"val斜率: +{val_slope:.6f}, train斜率: {train_slope:.6f}, "
                  f"gap倍率: {gap_ratio:.2f}x")
        else:
            # 信号消失，重置计数器
            if self.overfit_counter > 0:
                print(f"  ✓ 过拟合信号消失，计数器重置 (之前: {self.overfit_counter})")
            self.overfit_counter = 0

        if self.overfit_counter >= self.patience:
            self.should_stop = True
            print(f"\n{'='*60}")
            print(f"★ 过拟合早停触发！连续 {self.overfit_counter} 次检测到过拟合趋势")
            print(f"  val_loss 斜率: +{val_slope:.6f} (上升)")
            print(f"  train_loss 斜率: {train_slope:.6f} (下降)")
            print(f"  gap 倍率: {gap_ratio:.2f}x (阈值: {self.gap_threshold}x)")
            print(f"  最佳 Val Loss: {self.best_loss:.5f} (Epoch {self.best_epoch})")
            print(f"  建议使用 Epoch {self.best_epoch} 的模型进行推理")
            print(f"{'='*60}\n")
            return True

        return False

    def status(self):
        """返回当前状态字符串"""
        n = len(self.val_history)
        if n < self.window:
            return (f"[OverfitStop] 数据积累中: {n}/{self.window}, "
                    f"最佳: {self.best_loss:.5f} @ Epoch {self.best_epoch}")

        recent_val = self.val_history[-self.window:]
        recent_train = self.train_history[-self.window:]
        val_slope = self._linear_slope(recent_val)
        train_slope = self._linear_slope(recent_train)

        current_gap = np.mean(self.gap_history[-self.window:])
        baseline_str = f"{self.baseline_gap:.5f}" if self.baseline_gap else "N/A"

        return (f"[OverfitStop] val斜率: {val_slope:+.6f}, train斜率: {train_slope:+.6f}, "
                f"gap: {current_gap:.5f} (基准: {baseline_str}), "
                f"信号: {self.overfit_counter}/{self.patience}, "
                f"最佳: {self.best_loss:.5f} @ Epoch {self.best_epoch}")


def validate(model, val_loader, criterion, device, writer, epoch):
    """
    在验证集上评估模型，并记录可视化结果。
    """
    model.eval() # 切换到评估模式
    val_loss = 0
    val_score_loss = 0
    val_geo_loss = 0

    # 用于可视化的样本 (只取第一个 batch)
    vis_images = None
    vis_true_score = None
    vis_pred_score = None
    vis_pred_geo = None

    with torch.no_grad(): # 不计算梯度
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            true_score_maps = batch['score_map'].to(device).unsqueeze(1)
            true_geo_maps = batch['geo_map'].to(device)

            pred_score_maps, pred_geo_maps = model(images)

            loss, score_loss, geo_loss = criterion(
                true_score_maps, pred_score_maps,
                true_geo_maps, pred_geo_maps
            )

            val_loss += loss.item()
            val_score_loss += score_loss.item()
            val_geo_loss += geo_loss.item()

            # 抓取第一个 batch 用于可视化
            if batch_idx == 0:
                vis_images = images
                vis_true_score = true_score_maps
                vis_pred_score = pred_score_maps
                vis_pred_geo = pred_geo_maps[:, 0:1, :, :]

    avg_loss = val_loss / len(val_loader)
    avg_score_loss = val_score_loss / len(val_loader)
    avg_geo_loss = val_geo_loss / len(val_loader)

    # --- TensorBoard 可视化 ---
    if writer and vis_images is not None:
        writer.add_images('Vis/Input_Images', vis_images, epoch)
        writer.add_images('Vis/True_Score_Map', vis_true_score.repeat(1, 3, 1, 1), epoch)
        writer.add_images('Vis/Pred_Score_Map', vis_pred_score.repeat(1, 3, 1, 1), epoch)
        if vis_pred_geo.max() > 0:
            vis_pred_geo = vis_pred_geo / vis_pred_geo.max()
        writer.add_images('Vis/Pred_Geo_Map_Top', vis_pred_geo.repeat(1, 3, 1, 1), epoch)

    return avg_loss, avg_score_loss, avg_geo_loss


def save_best_checkpoint(model, epoch, val_loss, best_checkpoints_list):
    """
    管理 Top-K 最佳模型的保存与删除
    """
    checkpoint_dir = "detection_small/checkpoints/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    file_name = f"best_detection_epoch_{epoch}_loss_{val_loss:.5f}.pth"
    save_path = os.path.join(checkpoint_dir, file_name)
    torch.save(model.state_dict(), save_path)

    best_checkpoints_list.append((val_loss, save_path))
    best_checkpoints_list.sort(key=lambda x: x[0])

    if len(best_checkpoints_list) > BEST_MODELS_TO_KEEP:
        worst_model = best_checkpoints_list.pop()
        worst_path = worst_model[1]
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f"移除旧的 Top-{BEST_MODELS_TO_KEEP + 1} 模型: {os.path.basename(worst_path)}")

    print(f"★ 当前 Top-{BEST_MODELS_TO_KEEP} 模型已更新 (本轮 Loss: {val_loss:.5f})")

def main():
    """
    主训练函数。
    """
    # --- TensorBoard 设置 ---
    log_dir = f"detection_small/runs/ocr_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据加载 ---
    train_transform = get_transforms(is_train=True, input_size=INPUT_SIZE)
    train_dataset = OCRDataset(image_dir=TRAIN_IMAGE_DIR, label_dir=TRAIN_LABEL_DIR, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if os.path.exists(VAL_IMAGE_DIR) and os.path.exists(VAL_LABEL_DIR):
        print("发现验证集，正在加载...")
        val_transform = get_transforms(is_train=False, input_size=INPUT_SIZE)
        val_dataset = OCRDataset(image_dir=VAL_IMAGE_DIR, label_dir=VAL_LABEL_DIR, transform=val_transform)
        if len(val_dataset) > 0:
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print(f"验证集加载完成，共 {len(val_dataset)} 张图片。")

    # --- 模型、损失函数和优化器 ---
    model = EASTModel(pretrained=True).to(device)
    # model.load_state_dict(torch.load("detection_small/checkpoints/best_detection_490_loss_0.0880.pth"))
    criterion = EASTLoss(lambda_iou = 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=3e-4,
                           steps_per_epoch=len(train_loader),
                           epochs=EPOCHS, pct_start=0.05)

    # 将模型图写入 TensorBoard (仅一次)
    if train_loader:
        first_batch = next(iter(train_loader))
        writer.add_graph(model, first_batch['image'].to(device))

    # --- 初始化过拟合早停 ---
    early_stopping = OverfitEarlyStopping(
        window=OVERFIT_WINDOW,
        patience=OVERFIT_PATIENCE,
        gap_threshold=OVERFIT_GAP_THRESHOLD
    )

    regular_checkpoints = []
    best_checkpoints = []

    # --- 训练循环 ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_score_loss = 0
        epoch_geo_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            true_score_maps = batch['score_map'].to(device).unsqueeze(1)
            true_geo_maps = batch['geo_map'].to(device)

            optimizer.zero_grad()
            pred_score_maps, pred_geo_maps = model(images)
            loss, score_loss, geo_loss = criterion(true_score_maps, pred_score_maps, true_geo_maps, pred_geo_maps)
            loss.backward()

            # --- 记录梯度信息 (每 100 个 batch 记录一次，避免日志过大) ---
            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch * len(train_loader) + batch_idx)
                        writer.add_scalar(f'Gradient_Norm/{name}', param.grad.norm(), epoch * len(train_loader) + batch_idx)

            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            loss_val, score_loss_val, geo_loss_val = loss.item(), score_loss.item(), geo_loss.item()
            # 记录每个 batch 的训练损失到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('batch/loss', loss_val, global_step)
            writer.add_scalar('batch/score_loss', score_loss_val, global_step)
            writer.add_scalar('batch/geo_loss', geo_loss_val, global_step)

            epoch_loss += loss_val
            epoch_score_loss += score_loss_val
            epoch_geo_loss += geo_loss_val

            progress_bar.set_postfix({'loss': f"{loss_val:.4f}", 'score': f"{score_loss_val:.4f}", 'geo': f"{geo_loss_val:.4f}"})

        # --- Epoch 结束统计 ---
        avg_loss = epoch_loss / len(train_loader)
        avg_score_loss = epoch_score_loss / len(train_loader)
        avg_geo_loss = epoch_geo_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f"Epoch {epoch} 训练结束: Loss: {avg_loss:.4f} (Score: {avg_score_loss:.4f}, Geo: {avg_geo_loss:.4f}) Learning: {current_lr}")

        # --- 验证步骤 ---
        should_validate = False
        if epoch >= START_DENSE_VAL_EPOCH:
            should_validate = True
        elif epoch % VAL_INTERVAL == 0:
            should_validate = True

        if val_loader and should_validate:
            val_loss, val_score, val_geo = validate(model, val_loader, criterion, device, writer, epoch)
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f} (Score: {val_score:.4f}, Geo: {val_geo:.4f})")
            writer.add_scalars('epoch/loss', {'train': avg_loss, 'validation': val_loss}, epoch)
            writer.add_scalars('epoch/score Loss', {'train': avg_score_loss, 'validation': val_score}, epoch)
            writer.add_scalars('epoch/geo Loss', {'train': avg_geo_loss, 'validation': val_geo}, epoch)

            # --- TensorBoard: 记录过拟合监控指标 ---
            writer.add_scalar('overfit/gap', val_loss - avg_loss, epoch)
            writer.add_scalars('overfit/trend', {
                'train': avg_loss,
                'val': val_loss
            }, epoch)

            # 保存最佳模型
            if len(best_checkpoints) < BEST_MODELS_TO_KEEP or val_loss < best_checkpoints[-1][0]:
                save_best_checkpoint(model, epoch, val_loss, best_checkpoints)

            # ★ 过拟合早停检查
            if early_stopping(val_loss, avg_loss, epoch):
                # 触发早停，保存最后一个 checkpoint 后退出
                checkpoint_dir = "detection_small/checkpoints/"
                final_path = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}_early_stop.pth")
                torch.save(model.state_dict(), final_path)
                print(f"已保存早停时的模型: {final_path}")
                break

            # 打印早停状态
            print(f"  {early_stopping.status()}")

        else:
            writer.add_scalars('epoch/loss_train', {'train': avg_loss}, epoch)

        # --- 常规检查点保存 ---
        checkpoint_dir = "detection_small/checkpoints/"
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        checkpoint_filename = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_filename)
        regular_checkpoints.append(checkpoint_filename)
        if len(regular_checkpoints) > MAX_KEEP_CHECKPOINTS:
            oldest = regular_checkpoints.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

    # --- 训练结束总结 ---
    print("\n" + "="*60)
    if early_stopping.should_stop:
        print(f"训练因过拟合早停在 Epoch {epoch} 结束")
    else:
        print(f"训练完成，共 {EPOCHS} 个 Epoch")
    print(f"最佳模型: Epoch {early_stopping.best_epoch}, Val Loss: {early_stopping.best_loss:.5f}")
    if best_checkpoints:
        print(f"Top-{BEST_MODELS_TO_KEEP} 模型:")
        for i, (loss, path) in enumerate(best_checkpoints):
            print(f"  {i+1}. {os.path.basename(path)} (Loss: {loss:.5f})")
    print("="*60)

    writer.close()

if __name__ == '__main__':
    if not os.path.exists(TRAIN_IMAGE_DIR) or not os.path.exists(TRAIN_LABEL_DIR):
        print(f"错误: 请确保你的数据分别在 '{TRAIN_IMAGE_DIR}' 和 '{TRAIN_LABEL_DIR}' 目录下")
    else:
        main()