"""按顶部配置训练小图 ResNet18 EAST，保存纯模型参数。
当前验证路径指向训练目录，独立评估前须核对并设置验证目录；不支持完整状态续训。"""

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

# 超参数
LEARNING_RATE = 1e-4  # Adam 构造学习率；OneCycleLR 随后接管，峰值在 main 中设为 3e-4。
BATCH_SIZE = 8  # 每批 8 张图片；修改尺寸或批量后需重新核对显存。
EPOCHS = 400  # 最多训练 200 轮，早停可能提前结束。
INPUT_SIZE = (256, 768)  # 模型输入 (高, 宽)，单位 px；两边为 32 的倍数。
IMAGE_EXTENSIONS = (".png",)  # 明确只训练新 PNG，不混入目录中的旧 JPG
TRAIN_IMAGE_DIR = "detection_small/data/train_images"  # 训练图片目录，路径相对于仓库根目录。
TRAIN_LABEL_DIR = "detection_small/data/train_labels"  # 训练标签目录，按图片 stem 配对。
VAL_IMAGE_DIR = "detection_small/data/train_images"  # 当前指向训练图片，仅能自评；独立验证前需改为验证目录。
VAL_LABEL_DIR = "detection_small/data/train_labels"  # 当前指向训练标签，需与独立验证图片目录一起核对。

# 模型保存策略优化
MAX_KEEP_CHECKPOINTS = 5  # 常规纯权重保留最近 5 份；不含完整续训状态。
CHECKPOINT_INTERVAL = 5          # 新数据量较大，每 5 个 epoch 保存一次常规检查点
BEST_MODELS_TO_KEEP = 3        # 最佳检查点：保留验证集 Loss 最低的 Top 3
VAL_INTERVAL = 10  # 第 80 轮前每 10 轮验证一次。
START_DENSE_VAL_EPOCH = 80  # 从第 80 轮起逐轮验证，包含该轮。
GRADIENT_LOG_INTERVAL = 1000   # 避免大数据集下 TensorBoard 直方图写入过密

# 早停参数
EARLY_STOP_PATIENCE = 30        # 连续 15 次验证没有改善就停止
EARLY_STOP_MIN_DELTA = 1e-4     # 改善幅度小于此值不算改善


class EarlyStopping:
    """
    早停机制：当验证集 loss 连续多次没有改善时，提前终止训练。

    工作原理：
    - 记录历史最佳 val_loss
    - 每次验证后检查：如果 val_loss 比最佳值改善超过 min_delta，重置计数器
    - 否则计数器 +1，达到 patience 时触发停止
    - 同时监控 train/val gap，如果 gap 持续扩大也会触发警告
    """
    def __init__(self, patience=40, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_epoch = 0
        # 用于监控过拟合趋势
        self.gap_history = []

    def __call__(self, val_loss, train_loss, epoch):
        """
        每次验证后调用。
        返回: True 表示应该停止训练, False 表示继续
        """
        # 记录 train/val gap
        gap = val_loss - train_loss
        self.gap_history.append(gap)

        if self.best_loss is None:
            # 第一次验证
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False

        if val_loss < self.best_loss - self.min_delta:
            # 有改善，重置计数器
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 没有改善
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n{'='*60}")
                print(f"★ 早停触发！连续 {self.counter} 次验证无改善")
                print(f"  最佳 Val Loss: {self.best_loss:.5f} (Epoch {self.best_epoch})")
                print(f"  当前 Val Loss: {val_loss:.5f} (Epoch {epoch})")
                print(f"  建议使用 Epoch {self.best_epoch} 的模型进行推理")
                print(f"{'='*60}\n")
                return True

        # 过拟合趋势警告（最近 10 次验证的 gap 是否在扩大）
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
        """返回当前状态字符串，用于打印"""
        return f"[EarlyStopping] 耐心: {self.counter}/{self.patience}, 最佳: {self.best_loss:.5f} @ Epoch {self.best_epoch}"


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

    # TensorBoard 可视化
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
    # TensorBoard 设置
    log_dir = f"detection_small/runs/ocr_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载
    train_transform = get_transforms(is_train=True, input_size=INPUT_SIZE)
    train_dataset = OCRDataset(
        image_dir=TRAIN_IMAGE_DIR,
        label_dir=TRAIN_LABEL_DIR,
        transform=train_transform,
        image_extensions=IMAGE_EXTENSIONS,
        require_labels=True,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"训练目录 {TRAIN_IMAGE_DIR} 中没有 PNG 图片。"
            "请先放入新的训练图片和对应标签。"
        )
    print(f"训练集加载完成，共 {len(train_dataset)} 张 PNG 图片，输入尺寸: {INPUT_SIZE}。")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if os.path.exists(VAL_IMAGE_DIR) and os.path.exists(VAL_LABEL_DIR):
        print("发现验证集，正在加载...")
        val_transform = get_transforms(is_train=False, input_size=INPUT_SIZE)
        val_dataset = OCRDataset(
            image_dir=VAL_IMAGE_DIR,
            label_dir=VAL_LABEL_DIR,
            transform=val_transform,
            image_extensions=IMAGE_EXTENSIONS,
            require_labels=True,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print(f"验证集加载完成，共 {len(val_dataset)} 张图片。")

    # 模型、损失函数和优化器
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

    # 初始化早停
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)

    regular_checkpoints = []
    best_checkpoints = []

    # 训练循环
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

            # 低频记录梯度信息，避免大数据集下日志和 I/O 过重
            if batch_idx % GRADIENT_LOG_INTERVAL == 0:
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

        # Epoch 结束统计
        avg_loss = epoch_loss / len(train_loader)
        avg_score_loss = epoch_score_loss / len(train_loader)
        avg_geo_loss = epoch_geo_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f"Epoch {epoch} 训练结束: Loss: {avg_loss:.4f} (Score: {avg_score_loss:.4f}, Geo: {avg_geo_loss:.4f}) Learning: {current_lr}")

        # 验证步骤
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

            # 保存最佳模型
            if len(best_checkpoints) < BEST_MODELS_TO_KEEP or val_loss < best_checkpoints[-1][0]:
                save_best_checkpoint(model, epoch, val_loss, best_checkpoints)

            # ★ 早停检查
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

        # 常规检查点保存
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS:
            checkpoint_dir = "detection_small/checkpoints/"
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            checkpoint_filename = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_filename)
            regular_checkpoints.append(checkpoint_filename)
            if len(regular_checkpoints) > MAX_KEEP_CHECKPOINTS:
                oldest = regular_checkpoints.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)

    # 训练结束总结
    print("\n" + "="*60)
    if early_stopping.should_stop:
        print(f"训练因早停在 Epoch {epoch} 结束")
    else:
        print(f"训练完成，共 {EPOCHS} 个 Epoch")
    if early_stopping.best_loss is not None:
        print(f"最佳模型: Epoch {early_stopping.best_epoch}, Val Loss: {early_stopping.best_loss:.5f}")
    else:
        print("未运行验证：没有可用的 PNG 验证集。")
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
