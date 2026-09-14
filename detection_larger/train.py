"""训练独立的整图 ResNet18 EAST，直接读取已准备好的训练与验证目录。"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import time
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.tensorboard import SummaryWriter

from detection_larger.components.dataset import OCRDataset
from detection_larger.components.transforms import get_transforms, normalize_input_size
from detection_larger.models.east_model import EASTModel
from detection_larger.utils import save_checkpoint
from detection_larger.components.loss import EASTLoss
from torch.optim.lr_scheduler import OneCycleLR
from detection_larger.components.training_options import parse_args, load_initial_weights

# 超参数：沿用参考项目的 Adam、OneCycleLR 和早停配置
LEARNING_RATE = 1e-4  # Adam 构造时的学习率，随后由 OneCycleLR 接管实际学习率
MAX_LEARNING_RATE = 3e-4  # OneCycleLR 的峰值学习率
BATCH_SIZE = 4  # 每批 4 张完整图片；调整输入尺寸时需重新核对显存占用
EPOCHS = 620  # 最大训练轮数；达到早停条件时提前结束
INPUT_SIZE = 640  # 整图方形输入边长，单位 px；显式传给训练预处理并写入检查点，推理尺寸由 inference.py 的 INPUT_SIZE 独立设置。
TRAIN_IMAGE_DIR = "detection_larger/data/train_images"  # 已准备好的训练图片目录，同时读取 JPG 和 PNG
TRAIN_LABEL_DIR = "detection_larger/data/train_labels"  # 训练文字框标签目录，按图片 stem 配对
VAL_IMAGE_DIR = "detection_larger/data/val_images"  # 独立验证图片目录，不与训练集混用
VAL_LABEL_DIR = "detection_larger/data/val_labels"  # 验证文字框标签目录，标签格式与训练集一致

NUM_WORKERS = 4  # 数据加载子进程数；设为 0 时在主进程加载，便于定位数据问题
PRETRAINED = True  # 无初始模型权重时加载 ImageNet 预训练参数；无缓存会下载，离线可设 False
LOG_GRAPH = True  # 是否向 TensorBoard 写入模型结构图；不影响训练损失的记录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 默认优先使用 GPU，也可手动设为 cpu 或 cuda:设备编号

MAX_KEEP_CHECKPOINTS = 5  # 常规检查点最多保留最近几份，不包含最佳模型和早停模型
CHECKPOINT_INTERVAL = 5  # 常规检查点保存间隔，单位为轮；最后一轮也保存
BEST_MODELS_TO_KEEP = 3  # 保留验证损失最低的几份模型，即 Top-K
VAL_INTERVAL = 10  # 密集验证开始前，每隔多少轮验证一次
START_DENSE_VAL_EPOCH = 50  # 从这一轮开始逐轮验证，包含该轮
GRADIENT_LOG_INTERVAL = 100  # 每轮内梯度直方图和范数的记录间隔，单位为 batch；首批也记录
EARLY_STOP_PATIENCE = 100  # 连续多少次验证未达到最小改善幅度时早停，按验证次数而非轮数计
EARLY_STOP_MIN_DELTA = 1e-5  # 验证损失下降须超过此绝对值才重置早停计数，不影响 Top-K 排名


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
                true_geo_maps, pred_geo_maps, batch["training_mask"].to(device)
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


def save_best_checkpoint(model, epoch, val_loss, best_checkpoints_list, checkpoint_dir, input_size):
    """
    管理 Top-K 最佳模型的保存与删除
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    file_name = f"best_detection_epoch_{epoch}_loss_{val_loss:.5f}.pth"
    save_path = Path(checkpoint_dir) / file_name
    save_checkpoint(model, save_path, input_size)

    best_checkpoints_list.append((val_loss, save_path))
    best_checkpoints_list.sort(key=lambda x: x[0])

    if len(best_checkpoints_list) > BEST_MODELS_TO_KEEP:
        worst_model = best_checkpoints_list.pop()
        worst_path = worst_model[1]
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f"移除旧的 Top-{BEST_MODELS_TO_KEEP + 1} 模型: {os.path.basename(worst_path)}")

    print(f"★ 当前 Top-{BEST_MODELS_TO_KEEP} 模型已更新 (本轮 Loss: {val_loss:.5f})")

def main(argv=None):
    """
    主训练函数。
    """
    args = parse_args(argv)
    if min(EPOCHS, BATCH_SIZE) <= 0 or NUM_WORKERS < 0 or not math.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0:
        raise ValueError("EPOCHS/BATCH_SIZE/LEARNING_RATE must be positive; NUM_WORKERS must be nonnegative")
    normalize_input_size(INPUT_SIZE)
    if Path(TRAIN_IMAGE_DIR).resolve() == Path(VAL_IMAGE_DIR).resolve():
        raise ValueError("Training and validation image directories must differ")
    device = torch.device(DEVICE)
    print(f"使用设备: {device}；整图输入: {INPUT_SIZE}×{INPUT_SIZE}")

    # 数据加载：只替换目录枚举，仍是一张完整原图对应一个训练样本
    train_transform = get_transforms(is_train=True, input_size=INPUT_SIZE)
    train_dataset = OCRDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    val_transform = get_transforms(is_train=False, input_size=INPUT_SIZE)
    val_dataset = OCRDataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    print(f"训练集 {len(train_dataset)} 张，验证集 {len(val_dataset)} 张；使用已准备好的独立目录。")

    # TensorBoard 与检查点分别放入本次运行的新目录
    run_name = f"ocr_experiment_{time.time_ns()}"
    checkpoint_dir = args.output or Path("detection_larger/checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("detection_larger/runs") / run_name
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard 日志: {log_dir}；检查点: {checkpoint_dir}")

    # 模型、损失函数和优化器
    model = EASTModel(pretrained=PRETRAINED and args.weights is None).to(device)
    if args.weights:
        load_initial_weights(model, args.weights)
    criterion = EASTLoss(lambda_iou = 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE,
                           steps_per_epoch=len(train_loader),
                           epochs=EPOCHS, pct_start=0.05)

    # 将模型图写入 TensorBoard (仅一次)
    if LOG_GRAPH:
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
            loss, score_loss, geo_loss = criterion(true_score_maps, pred_score_maps, true_geo_maps, pred_geo_maps, batch["training_mask"].to(device))
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
                save_best_checkpoint(model, epoch, val_loss, best_checkpoints, checkpoint_dir, INPUT_SIZE)

            # ★ 早停检查
            if early_stopping(val_loss, avg_loss, epoch):
                # 触发早停，保存最后一个 checkpoint 后退出
                final_path = checkpoint_dir / f"detection_epoch_{epoch}_early_stop.pth"
                save_checkpoint(model, final_path, INPUT_SIZE)
                print(f"已保存早停时的模型: {final_path}")
                break

            # 打印早停状态
            print(f"  {early_stopping.status()}")

        else:
            writer.add_scalars('epoch/loss_train', {'train': avg_loss}, epoch)

        # 常规检查点保存
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS:
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            checkpoint_filename = checkpoint_dir / f"detection_epoch_{epoch}.pth"
            save_checkpoint(model, checkpoint_filename, INPUT_SIZE)
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
        print("未达到验证轮次。")
    if best_checkpoints:
        print(f"Top-{BEST_MODELS_TO_KEEP} 模型:")
        for i, (loss, path) in enumerate(best_checkpoints):
            print(f"  {i+1}. {os.path.basename(path)} (Loss: {loss:.5f})")
    print("="*60)

    writer.close()

if __name__ == '__main__':
    main()
