"""文档检测独立使用 Dice 与 Smooth L1，并屏蔽 ### 标注区域。"""
import torch
from torch import nn


class DiceLoss(nn.Module):
    """按整个批次计算文本与背景的 Dice 损失。"""

    def __init__(self, smooth=1e-5):
        """设置分母平滑量，保证空白样本的损失有限。"""
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """计算同形状置信度图与真值之间的标量损失。"""
        pred, target = pred.view(-1), target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class EASTLoss(nn.Module):
    """合并置信度损失和四方向 log1p 几何距离损失。"""

    def __init__(self, lambda_iou=5.0):
        """保留 lambda_iou 参数名兼容训练入口，其实际控制 Smooth L1 的权重。"""
        super().__init__()
        self.dice_loss = DiceLoss()
        self.lambda_iou = lambda_iou

    def forward(self, y_true_score, y_pred_score, y_true_geo, y_pred_geo, training_mask=None):
        """返回总损失、Dice 损失和几何损失；忽略区域不参与梯度计算。"""
        if training_mask is not None:
            if training_mask.ndim == 3:
                training_mask = training_mask.unsqueeze(1)
            y_true_score = y_true_score * training_mask
            y_pred_score = y_pred_score * training_mask

        score_loss = self.dice_loss(y_pred_score, y_true_score)
        mask = (y_true_score > 0).float()
        geo_loss = self.smooth_l1_geo(y_pred_geo, y_true_geo, mask)
        return score_loss + self.lambda_iou * geo_loss, score_loss, geo_loss

    def smooth_l1_geo(self, pred, target, mask):
        """仅在正样本网格计算 Smooth L1，按正样本数乘四个几何通道归一化。"""
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask_4ch = mask.unsqueeze(1).expand_as(pred)
        loss = nn.functional.smooth_l1_loss(pred * mask_4ch, target * mask_4ch, reduction='sum')
        return loss / (torch.sum(mask) * 4 + 1e-6)
