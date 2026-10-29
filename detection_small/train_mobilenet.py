import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torch.utils.tensorboard import SummaryWriter

from components.dataset import OCRDataset
from components.transforms import get_transforms
from models.east_model import EASTModel
from components.loss import EASTLoss
from torch.optim.lr_scheduler import OneCycleLR

# --- 超参数 ---
# ★ 改动 1：降低学习率，MobileNetV3 通道窄，梯度传播特性不同
LEARNING_RATE = 5e-5
# ★ 改动 2：增大 batch size（MobileNetV3 更轻量，显存允许更大 batch）
BATCH_SIZE = 12
EPOCHS = 500
INPUT_SIZE = 640
TRAIN_IMAGE_DIR = "detection_small/data/train_images"
TRAIN_LABEL_DIR = "detection_small/data/train_labels"
VAL_IMAGE_DIR = "detection_small/data/val_images"
VAL_LABEL_DIR = "detection_small/data/val_labels"

# --- 模型保存策略 ---
MAX_KEEP_CHECKPOINTS = 20
BEST_MODELS_TO_KEEP = 3
VAL_INTERVAL = 10
START_DENSE_VAL_EPOCH = 50

# --- 早停参数 ---
EARLY_STOP_PATIENCE = 40
EARLY_STOP_MIN_DELTA = 1e-5

# ★ 改动：最佳模型保存标准
# 由于验证集是从训练集中抽取的，val_loss 与 train_loss 高度相关，
# 用 val_loss 选模型容易产生"虚假信心"——模型只是记住了这些样本。
# 改为用 train_loss 作为保存标准，反映模型对全部训练数据的拟合程度。
SAVE_BEST_BY = "train"  # "train" 或 "val"


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=40, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_epoch = 0
        self.gap_history = []

    def __call__(self, val_loss, train_loss, epoch):
        # ★ 改动：用 train_loss 驱动早停判断
        # 因为 val 来自 train，val_loss 不具备独立泛化评估能力
        monitor_loss = train_loss if SAVE_BEST_BY == "train" else val_loss
        gap = val_loss - train_loss
        self.gap_history.append(gap)

        if self.best_loss is None:
            self.best_loss = monitor_loss
            self.best_epoch = epoch
            return False

        if monitor_loss < self.best_loss - self.min_delta:
            self.best_loss = monitor_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n{'='*60}")
                print(f"★ 早停触发！连续 {self.counter} 次无改善 (监控: {SAVE_BEST_BY}_loss)")
                print(f"  最佳 {SAVE_BEST_BY} Loss: {self.best_loss:.5f} (Epoch {self.best_epoch})")
                print(f"  当前 {SAVE_BEST_BY} Loss: {monitor_loss:.5f} (Epoch {epoch})")
                print(f"{'='*60}\n")
                return True

        if len(self.gap_history) >= 10:
            recent_gaps = self.gap_history[-10:]
            older_gaps = self.gap_history[-20:-10] if len(self.gap_history) >= 20 else self.gap_history[:10]
            avg_recent = sum(recent_gaps) / len(recent_gaps)
            avg_older = sum(older_gaps) / len(older_gaps)
            if avg_recent > avg_older * 1.5 and avg_recent > 0.002:
                print(f"  ⚠ 过拟合警告: train/val gap 扩大中 "
                      f"(近期均值: {avg_recent:.5f}, 之前: {avg_older:.5f})")

        return False

    def status(self):
        return (f"[EarlyStopping] 耐心: {self.counter}/{self.patience}, "
                f"最佳 {SAVE_BEST_BY}: {self.best_loss:.5f} @ Epoch {self.best_epoch}")


def validate(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    val_loss = 0
    val_score_loss = 0
    val_geo_loss = 0

    vis_images = None
    vis_true_score = None
    vis_pred_score = None
    vis_pred_geo = None

    with torch.no_grad():
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

            if batch_idx == 0:
                vis_images = images
                vis_true_score = true_score_maps
                vis_pred_score = pred_score_maps
                vis_pred_geo = pred_geo_maps[:, 0:1, :, :]

    avg_loss = val_loss / len(val_loader)
    avg_score_loss = val_score_loss / len(val_loader)
    avg_geo_loss = val_geo_loss / len(val_loader)

    if writer and vis_images is not None:
        writer.add_images('Vis/Input_Images', vis_images, epoch)
        writer.add_images('Vis/True_Score_Map', vis_true_score.repeat(1, 3, 1, 1), epoch)
        writer.add_images('Vis/Pred_Score_Map', vis_pred_score.repeat(1, 3, 1, 1), epoch)
        if vis_pred_geo.max() > 0:
            vis_pred_geo = vis_pred_geo / vis_pred_geo.max()
        writer.add_images('Vis/Pred_Geo_Map_Top', vis_pred_geo.repeat(1, 3, 1, 1), epoch)

    return avg_loss, avg_score_loss, avg_geo_loss


def save_best_checkpoint(model, epoch, loss_value, best_checkpoints_list, loss_type="train"):
    """
    保存最佳模型检查点。
    :param loss_type: "train" 或 "val"，标识保存时使用的 loss 类型
    """
    checkpoint_dir = "detection_small/checkpoints/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    file_name = f"best_detection_epoch_{epoch}_{loss_type}_loss_{loss_value:.5f}.pth"
    save_path = os.path.join(checkpoint_dir, file_name)
    torch.save(model.state_dict(), save_path)

    best_checkpoints_list.append((loss_value, save_path))
    best_checkpoints_list.sort(key=lambda x: x[0])

    if len(best_checkpoints_list) > BEST_MODELS_TO_KEEP:
        worst_model = best_checkpoints_list.pop()
        worst_path = worst_model[1]
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f"移除旧的 Top-{BEST_MODELS_TO_KEEP + 1} 模型: {os.path.basename(worst_path)}")

    print(f"★ 当前 Top-{BEST_MODELS_TO_KEEP} 模型已更新 "
          f"(本轮 {loss_type} Loss: {loss_value:.5f})")

def main():
    log_dir = f"detection_small/runs/ocr_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")
    print(f"★ 最佳模型保存标准: {SAVE_BEST_BY}_loss")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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

    model = EASTModel(pretrained=True).to(device)

    # ★ 改动 3：冻结骨干网络前几个 stage
    # 延长到 25 个 epoch，让 FPN 头部 + 空洞卷积模块充分适配
    FREEZE_BACKBONE_EPOCHS = 25
    for name, param in model.named_parameters():
        if name.startswith('stage0') or name.startswith('stage1'):
            param.requires_grad = False

    criterion = EASTLoss(lambda_iou=3.0).to(device)

    # ★ 改动 4：增大 weight_decay 抑制过拟合
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=5e-4
    )
    scheduler = OneCycleLR(optimizer, max_lr=2e-4,
                           steps_per_epoch=len(train_loader),
                           epochs=EPOCHS, pct_start=0.05)

    if train_loader:
        first_batch = next(iter(train_loader))
        writer.add_graph(model, first_batch['image'].to(device))

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)

    regular_checkpoints = []
    best_checkpoints = []

    for epoch in range(1, EPOCHS + 1):
        # ★ 改动 3 续：到达指定 epoch 后解冻骨干网络
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            print(f"\n★ Epoch {epoch}: 解冻骨干网络 stage0 和 stage1")
            for name, param in model.named_parameters():
                param.requires_grad = True
            # 重建优化器以包含所有参数
            # ★ 解冻后的 warmup 延长到 0.1，给新解冻的参数更多适应时间
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
            scheduler = OneCycleLR(optimizer, max_lr=2e-4,
                                   steps_per_epoch=len(train_loader),
                                   epochs=EPOCHS - epoch + 1, pct_start=0.1)

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

            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch * len(train_loader) + batch_idx)
                        writer.add_scalar(f'Gradient_Norm/{name}', param.grad.norm(), epoch * len(train_loader) + batch_idx)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            loss_val, score_loss_val, geo_loss_val = loss.item(), score_loss.item(), geo_loss.item()
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('batch/loss', loss_val, global_step)
            writer.add_scalar('batch/score_loss', score_loss_val, global_step)
            writer.add_scalar('batch/geo_loss', geo_loss_val, global_step)

            epoch_loss += loss_val
            epoch_score_loss += score_loss_val
            epoch_geo_loss += geo_loss_val

            progress_bar.set_postfix({'loss': f"{loss_val:.4f}", 'score': f"{score_loss_val:.4f}", 'geo': f"{geo_loss_val:.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        avg_score_loss = epoch_score_loss / len(train_loader)
        avg_geo_loss = epoch_geo_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f"Epoch {epoch} 训练结束: Loss: {avg_loss:.4f} (Score: {avg_score_loss:.4f}, Geo: {avg_geo_loss:.4f}) Learning: {current_lr}")

        # ★ 改动：每个 epoch 都用 train_loss 尝试更新最佳模型
        if SAVE_BEST_BY == "train":
            if len(best_checkpoints) < BEST_MODELS_TO_KEEP or avg_loss < best_checkpoints[-1][0]:
                save_best_checkpoint(model, epoch, avg_loss, best_checkpoints, loss_type="train")

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

            # ★ 如果用 val_loss 保存，走原逻辑
            if SAVE_BEST_BY == "val":
                if len(best_checkpoints) < BEST_MODELS_TO_KEEP or val_loss < best_checkpoints[-1][0]:
                    save_best_checkpoint(model, epoch, val_loss, best_checkpoints, loss_type="val")

            # 早停仍然运行（内部根据 SAVE_BEST_BY 选择监控哪个 loss）
            if early_stopping(val_loss, avg_loss, epoch):
                checkpoint_dir = "detection_small/checkpoints/"
                final_path = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}_early_stop.pth")
                torch.save(model.state_dict(), final_path)
                print(f"已保存早停时的模型: {final_path}")
                break

            print(f"  {early_stopping.status()}")
        else:
            writer.add_scalars('epoch/loss_train', {'train': avg_loss}, epoch)
            # ★ 无验证时也用 train_loss 驱动早停（传 train_loss 作为两个参数）
            if SAVE_BEST_BY == "train" and epoch >= START_DENSE_VAL_EPOCH:
                if early_stopping(avg_loss, avg_loss, epoch):
                    checkpoint_dir = "detection_small/checkpoints/"
                    final_path = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}_early_stop.pth")
                    torch.save(model.state_dict(), final_path)
                    print(f"已保存早停时的模型: {final_path}")
                    break

        checkpoint_dir = "detection_small/checkpoints/"
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        checkpoint_filename = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_filename)
        regular_checkpoints.append(checkpoint_filename)
        if len(regular_checkpoints) > MAX_KEEP_CHECKPOINTS:
            oldest = regular_checkpoints.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

    print("\n" + "="*60)
    if early_stopping.should_stop:
        print(f"训练因早停在 Epoch {epoch} 结束")
    else:
        print(f"训练完成，共 {EPOCHS} 个 Epoch")
    print(f"最佳模型 ({SAVE_BEST_BY}_loss): Epoch {early_stopping.best_epoch}, Loss: {early_stopping.best_loss:.5f}")
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