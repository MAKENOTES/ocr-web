"""训练统一 ResNet18 EAST，覆盖 UI 小图、PDF 页面和网页表格文字。

训练使用固定输入尺寸保证批次可直接堆叠；推理入口可再按图片宽高比动态
选择尺寸。训练与验证目录必须独立，加载器严格按 stem 配对标签。
"""
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.components.augmentation import policy_weights_for_epoch
from detection.components.dataset import OCRDataset
from detection.components.eval_metrics import evaluate_model
from detection.components.loss import EASTLoss
from detection.components.transforms import get_transforms, normalize_input_size
from detection.models.east_model import EASTModel
from detection.utils import save_checkpoint


# 超参数：沿用参考项目的 Adam、OneCycleLR 和早停配置。
LEARNING_RATE = 1.5e-4  # Adam 构造学习率；batch 4→8 后按约 1.5× 缩放，随后由 OneCycleLR 接管。
MAX_LEARNING_RATE = 4.5e-4  # OneCycleLR 峰值学习率；若 batch 8 出现明显抖动可回退到 4e-4。
BATCH_SIZE = 8  # 每批 8 张完整图片；768 输入约需 9-10GB 显存，调整尺寸时重新核对显存占用。
EPOCHS = 620  # 最大训练轮数；达到早停条件时提前结束。
INPUT_SIZE = 768  # 训练方形输入边长；与当前最佳推理尺寸一致，显存不足时可回退到 640。
TRAIN_IMAGE_DIR = "detection/data/train_images"  # 已准备好的训练图片目录，同时读取 JPG 和 PNG。
TRAIN_LABEL_DIR = "detection/data/train_labels"  # 训练文字框标签目录，按图片 stem 配对。
VAL_IMAGE_DIR = "detection/data/val_images"  # 独立验证图片目录，不与训练集混用。
VAL_LABEL_DIR = "detection/data/val_labels"  # 验证文字框标签目录，标签格式与训练集一致。

CHECKPOINT_ROOT = Path("detection/checkpoints")  # 检查点根目录；每次训练在其下新建带时间戳的子目录。
RUNS_ROOT = Path("detection/runs")  # TensorBoard 日志根目录；与检查点目录分开，避免混入权重文件。

NUM_WORKERS = 8  # 数据加载子进程数；设为 0 时在主进程加载，便于定位数据问题。
PRETRAINED = True  # 是否加载 ImageNet 预训练参数；无缓存会下载，离线可设 False。
LOG_GRAPH = True  # 是否向 TensorBoard 写入模型结构图；不影响训练损失的记录。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 默认优先使用 GPU，也可手动设为 cpu 或 cuda:设备编号。
USE_AUGMENTATION_CURRICULUM = True  # 启用时按轮次从轻增强过渡到强增强；关闭时固定使用标准增强。

MAX_KEEP_CHECKPOINTS = 5  # 常规检查点最多保留最近几份，不包含最佳模型和早停模型。
CHECKPOINT_INTERVAL = 5  # 常规检查点保存间隔，单位为轮；最后一轮也保存。
BEST_MODELS_TO_KEEP = 3  # 保留验证损失最低的几份模型，即 Top-K。
BEST_F1_MODELS_TO_KEEP = 3  # 保留验证集 F1 最高的几份模型；F1 与 loss 可能不同步。
VAL_INTERVAL = 10  # 密集验证开始前，每隔多少轮验证一次。
START_DENSE_VAL_EPOCH = 50  # 从这一轮开始逐轮验证，包含该轮。
METRIC_VAL_INTERVAL = 10  # 每多少轮验证计算一次检测 F1/P/R；0 表示关闭。
METRIC_INPUT_SIZE = 768  # 训练期 F1 评估尺寸，与当前部署最佳尺寸一致；可改为 INPUT_SIZE 降低开销。
METRIC_SCORE_THRESHOLD = 0.7  # 训练期 F1 评估的置信度阈值，取值范围 (0, 1)。
METRIC_NMS_IOU_THRESHOLD = 0.3  # 训练期 F1 评估的重复框抑制阈值，取值范围 (0, 1]。
METRIC_MATCH_IOU = 0.5  # 预测框与真值匹配的 IoU 阈值。
BATCH_LOG_INTERVAL = 50  # 每多少 batch 记录一次 batch loss；0 表示关闭。
GRADIENT_LOG_INTERVAL = 500  # 每多少 batch 记录一次总梯度范数；0 表示关闭。
VIS_INTERVAL = 20  # 每多少轮写一次验证集可视化图片，减少 TensorBoard 体积。
EARLY_STOP_PATIENCE = 100  # 连续多少次验证未达到最小改善幅度时早停，按验证次数而非轮数计。
EARLY_STOP_MIN_DELTA = 1e-5  # 验证损失下降须超过此绝对值才重置早停计数，不影响 Top-K 排名。


class EarlyStopping:
    """验证损失连续多次没有改善时提前结束训练，并提示过拟合趋势。"""

    def __init__(self, patience=40, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_epoch = 0
        self.gap_history = []

    def __call__(self, val_loss, train_loss, epoch):
        """每次验证后调用；返回 True 表示应该停止训练。"""
        self.gap_history.append(val_loss - train_loss)
        if self.best_loss is None:
            self.best_loss, self.best_epoch = val_loss, epoch
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch = val_loss, epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n{'=' * 60}")
                print(f"★ 早停触发！连续 {self.counter} 次验证无改善")
                print(f"  最佳 Val Loss: {self.best_loss:.5f} (Epoch {self.best_epoch})")
                print(f"  当前 Val Loss: {val_loss:.5f} (Epoch {epoch})")
                print(f"  建议使用 Epoch {self.best_epoch} 的模型进行推理")
                print(f"{'=' * 60}\n")
                return True

        # 过拟合趋势警告（最近 10 次验证的 gap 是否在扩大）。
        if len(self.gap_history) >= 10:
            recent = self.gap_history[-10:]
            older = self.gap_history[-20:-10] if len(self.gap_history) >= 20 else self.gap_history[:10]
            average_recent = sum(recent) / len(recent)
            average_older = sum(older) / len(older)
            if average_recent > average_older * 1.5 and average_recent > 0.002:
                print(f"  ⚠ 过拟合警告: train/val gap 扩大中 "
                      f"(近期均值: {average_recent:.5f}, 之前: {average_older:.5f})")
        return False

    def status(self):
        """返回当前早停状态，用于训练日志。"""
        return (f"[EarlyStopping] 耐心: {self.counter}/{self.patience}, "
                f"最佳: {self.best_loss:.5f} @ Epoch {self.best_epoch}")


def validate(model, val_loader, criterion, device, writer, epoch):
    """在验证集上评估损失；按间隔写入输入图和预测置信度图。"""
    model.eval()
    val_loss = val_score_loss = val_geo_loss = 0.0
    vis_images = vis_pred_score = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            true_score_maps = batch["score_map"].to(device).unsqueeze(1)
            true_geo_maps = batch["geo_map"].to(device)
            pred_score_maps, pred_geo_maps = model(images)
            loss, score_loss, geo_loss = criterion(
                true_score_maps, pred_score_maps, true_geo_maps, pred_geo_maps,
                batch["training_mask"].to(device))
            val_loss += loss.item()
            val_score_loss += score_loss.item()
            val_geo_loss += geo_loss.item()

            if batch_idx == 0:
                vis_images = images
                vis_pred_score = pred_score_maps

    if writer and vis_images is not None and VIS_INTERVAL > 0 and epoch % VIS_INTERVAL == 0:
        writer.add_images("Vis/Input_Images", vis_images, epoch)
        writer.add_images("Vis/Pred_Score_Map", vis_pred_score.repeat(1, 3, 1, 1), epoch)

    count = len(val_loader)
    return val_loss / count, val_score_loss / count, val_geo_loss / count


def save_best_checkpoint(model, epoch, val_loss, best_checkpoints_list, checkpoint_dir, input_size):
    """更新 Top-K 最佳模型并删除超出的最差模型。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"best_detection_epoch_{epoch}_loss_{val_loss:.5f}.pth"
    save_checkpoint(model, save_path, input_size)

    best_checkpoints_list.append((val_loss, save_path))
    best_checkpoints_list.sort(key=lambda item: item[0])
    if len(best_checkpoints_list) > BEST_MODELS_TO_KEEP:
        _, worst_path = best_checkpoints_list.pop()
        if worst_path.exists():
            worst_path.unlink()
            print(f"移除旧的 Top-{BEST_MODELS_TO_KEEP + 1} 模型: {worst_path.name}")
    print(f"★ 当前 Top-{BEST_MODELS_TO_KEEP} 模型已更新 (本轮 Loss: {val_loss:.5f})")


def save_best_f1_checkpoint(model, epoch, f1, best_checkpoints_list, checkpoint_dir, input_size):
    """按验证集 F1 更新 Top-K 模型；F1 与 loss 可能不同步，单独保留。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"best_f1_detection_epoch_{epoch}_f1_{f1:.5f}.pth"
    save_checkpoint(model, save_path, input_size)

    best_checkpoints_list.append((f1, save_path))
    best_checkpoints_list.sort(key=lambda item: -item[0])
    if len(best_checkpoints_list) > BEST_F1_MODELS_TO_KEEP:
        _, worst_path = best_checkpoints_list.pop()
        if worst_path.exists():
            worst_path.unlink()
            print(f"移除旧的 Top-{BEST_F1_MODELS_TO_KEEP + 1} F1 模型: {worst_path.name}")
    print(f"★ 当前 Top-{BEST_F1_MODELS_TO_KEEP} F1 模型已更新 (F1: {f1:.5f})")


def main():
    """主训练函数；配置由文件顶部常量控制，检查点和日志按时间自动新建。"""
    if (min(EPOCHS, BATCH_SIZE) <= 0 or NUM_WORKERS < 0
            or not math.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0):
        raise ValueError("EPOCHS/BATCH_SIZE/LEARNING_RATE must be positive; "
                         "NUM_WORKERS must be nonnegative")
    input_size = normalize_input_size(INPUT_SIZE)
    if Path(TRAIN_IMAGE_DIR).resolve() == Path(VAL_IMAGE_DIR).resolve():
        raise ValueError("Training and validation image directories must differ")

    device = torch.device(DEVICE)
    print(f"使用设备: {device}；训练输入: {input_size[0]}×{input_size[1]}")

    # 训练集只加载一次；增强阶段切换时重建 transform 和 DataLoader。
    train_dataset = OCRDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    val_transform = get_transforms(is_train=False, input_size=input_size)
    val_dataset = OCRDataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    def stage_weights(epoch):
        """返回当前轮次的增强强度权重；关闭课程时固定标准增强。"""
        if not USE_AUGMENTATION_CURRICULUM:
            return {"standard": 1.0}
        return policy_weights_for_epoch(epoch, EPOCHS)

    def build_train_loader(stage_mix):
        """按给定增强阶段权重重建训练 transform 和 DataLoader。"""
        train_dataset.transform = get_transforms(
            is_train=True, input_size=input_size, stage_weights=stage_mix)
        return DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    initial_stage_mix = stage_weights(1)
    train_loader = build_train_loader(initial_stage_mix)
    steps_per_epoch = len(train_loader)
    print(f"训练集 {len(train_dataset)} 张，验证集 {len(val_dataset)} 张；使用独立目录。")
    print(f"数据增强: 30 种，初始阶段权重 {initial_stage_mix}；"
          f"课程开关 {USE_AUGMENTATION_CURRICULUM}。")

    # TensorBoard 与检查点分别放入本次运行的新目录。
    run_name = f"ocr_experiment_{time.time_ns()}"
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = RUNS_ROOT / run_name
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard 日志: {log_dir}；检查点: {checkpoint_dir}")

    model = EASTModel(pretrained=PRETRAINED).to(device)
    criterion = EASTLoss(lambda_iou=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE,
                           steps_per_epoch=steps_per_epoch,
                           epochs=EPOCHS, pct_start=0.05)

    if LOG_GRAPH:
        first_batch = next(iter(train_loader))
        writer.add_graph(model, first_batch["image"].to(device))

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)
    regular_checkpoints, best_checkpoints, best_f1_checkpoints = [], [], []
    best_f1 = -1.0
    current_stage_mix = initial_stage_mix

    for epoch in range(1, EPOCHS + 1):
        # 只在课程阶段切换时重建 DataLoader，避免每轮重启 worker。
        stage_mix = stage_weights(epoch)
        for stage, value in stage_mix.items():
            writer.add_scalar(f"Augmentation/{stage}", value, epoch)
        if stage_mix != current_stage_mix:
            train_loader = build_train_loader(stage_mix)
            current_stage_mix = stage_mix
            print(f"Epoch {epoch} 增强阶段切换: {stage_mix}")

        model.train()
        epoch_started = time.perf_counter()
        epoch_loss = epoch_score_loss = epoch_geo_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            true_score_maps = batch["score_map"].to(device).unsqueeze(1)
            true_geo_maps = batch["geo_map"].to(device)

            optimizer.zero_grad()
            pred_score_maps, pred_geo_maps = model(images)
            loss, score_loss, geo_loss = criterion(
                true_score_maps, pred_score_maps, true_geo_maps, pred_geo_maps,
                batch["training_mask"].to(device))
            loss.backward()

            # 只记录总梯度范数；逐参数直方图会快速膨胀事件文件。
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if GRADIENT_LOG_INTERVAL > 0 and batch_idx % GRADIENT_LOG_INTERVAL == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Gradient/total_norm", float(total_grad_norm), step)

            optimizer.step()
            scheduler.step()

            loss_value, score_value, geo_value = loss.item(), score_loss.item(), geo_loss.item()
            step = epoch * len(train_loader) + batch_idx
            if BATCH_LOG_INTERVAL > 0 and batch_idx % BATCH_LOG_INTERVAL == 0:
                writer.add_scalar("batch/loss", loss_value, step)
            epoch_loss += loss_value
            epoch_score_loss += score_value
            epoch_geo_loss += geo_value
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "score": f"{score_value:.4f}",
                                      "geo": f"{geo_value:.4f}"})

        average_loss = epoch_loss / len(train_loader)
        average_score = epoch_score_loss / len(train_loader)
        average_geo = epoch_geo_loss / len(train_loader)
        train_seconds = time.perf_counter() - epoch_started
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("Performance/train_epoch_seconds", train_seconds, epoch)
        writer.add_scalar("Performance/train_images_per_second", len(train_dataset) / train_seconds, epoch)
        print(f"Epoch {epoch} 训练结束: Loss: {average_loss:.4f} "
              f"(Score: {average_score:.4f}, Geo: {average_geo:.4f}) Learning: {current_lr}")

        should_validate = epoch >= START_DENSE_VAL_EPOCH or epoch % VAL_INTERVAL == 0
        if should_validate:
            validation_started = time.perf_counter()
            val_loss, val_score, val_geo = validate(
                model, val_loader, criterion, device, writer, epoch)
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f} "
                  f"(Score: {val_score:.4f}, Geo: {val_geo:.4f})")
            writer.add_scalars("epoch/loss", {"train": average_loss, "validation": val_loss}, epoch)
            writer.add_scalars("epoch/score Loss", {"train": average_score, "validation": val_score}, epoch)
            writer.add_scalars("epoch/geo Loss", {"train": average_geo, "validation": val_geo}, epoch)

            if METRIC_VAL_INTERVAL > 0 and epoch % METRIC_VAL_INTERVAL == 0:
                metrics = evaluate_model(
                    model, val_dataset, device, METRIC_INPUT_SIZE,
                    threshold=METRIC_SCORE_THRESHOLD,
                    nms_iou=METRIC_NMS_IOU_THRESHOLD,
                    match_iou=METRIC_MATCH_IOU)
                for group, values in metrics.items():
                    suffix = "" if group == "overall" else f"_{group}"
                    writer.add_scalar(f"Val/Precision{suffix}", values["precision"], epoch)
                    writer.add_scalar(f"Val/Recall{suffix}", values["recall"], epoch)
                    writer.add_scalar(f"Val/F1{suffix}", values["f1"], epoch)
                overall = metrics["overall"]
                print(f"Epoch {epoch} Val F1: {overall['f1']:.4f} "
                      f"(P: {overall['precision']:.4f}, R: {overall['recall']:.4f})")
                if overall["f1"] > best_f1:
                    best_f1 = overall["f1"]
                    save_best_f1_checkpoint(model, epoch, best_f1, best_f1_checkpoints,
                                            checkpoint_dir, input_size)
                writer.add_scalar("Val/best_F1", best_f1, epoch)

            writer.add_scalar("Performance/validation_seconds",
                              time.perf_counter() - validation_started, epoch)

            if len(best_checkpoints) < BEST_MODELS_TO_KEEP or val_loss < best_checkpoints[-1][0]:
                save_best_checkpoint(model, epoch, val_loss, best_checkpoints,
                                     checkpoint_dir, input_size)
            if early_stopping(val_loss, average_loss, epoch):
                final_path = checkpoint_dir / f"detection_epoch_{epoch}_early_stop.pth"
                save_checkpoint(model, final_path, input_size)
                print(f"已保存早停时的模型: {final_path}")
                break
            print(f"  {early_stopping.status()}")
        else:
            writer.add_scalars("epoch/loss", {"train": average_loss}, epoch)
            writer.add_scalars("epoch/score Loss", {"train": average_score}, epoch)
            writer.add_scalars("epoch/geo Loss", {"train": average_geo}, epoch)

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS:
            checkpoint_path = checkpoint_dir / f"detection_epoch_{epoch}.pth"
            save_checkpoint(model, checkpoint_path, input_size)
            regular_checkpoints.append(checkpoint_path)
            if len(regular_checkpoints) > MAX_KEEP_CHECKPOINTS:
                oldest = regular_checkpoints.pop(0)
                if oldest.exists():
                    oldest.unlink()

    print("\n" + "=" * 60)
    if early_stopping.should_stop:
        print(f"训练因早停在 Epoch {epoch} 结束")
    else:
        print(f"训练完成，共 {EPOCHS} 个 Epoch")
    if early_stopping.best_loss is not None:
        print(f"最佳模型: Epoch {early_stopping.best_epoch}, "
              f"Val Loss: {early_stopping.best_loss:.5f}")
    if best_checkpoints:
        print(f"Top-{BEST_MODELS_TO_KEEP} 模型:")
        for index, (loss, path) in enumerate(best_checkpoints):
            print(f"  {index + 1}. {path.name} (Loss: {loss:.5f})")
    if best_f1_checkpoints:
        print(f"Top-{BEST_F1_MODELS_TO_KEEP} F1 模型:")
        for index, (f1, path) in enumerate(best_f1_checkpoints):
            print(f"  {index + 1}. {path.name} (F1: {f1:.5f})")
    print("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()
