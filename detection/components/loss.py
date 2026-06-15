import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    用于分割任务的 Dice Loss。
    Dice Loss 对于处理类别不平衡问题（如文本检测中的文本/背景像素）特别有效。
    它的计算方式是 2 * |X ∩ Y| / (|X| + |Y|)，其中 X 是预测，Y 是真值。
    """
    def __init__(self, smooth=1e-5):
        """
        初始化 Dice Loss。
        :param smooth: float, 一个小的平滑常数，用于防止分母为零。
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        计算 Dice Loss。
        :param pred: 预测的分割图，张量形状为 (N, *)。
        :param target: 真值的分割图，张量形状与 pred 相同。
        :return: float, 计算出的 Dice Loss 值 (1 - Dice系数)。
        """
        # 将标签和预测张量展平
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice_coeff

class EASTLoss(nn.Module):
    """
    EAST 模型的损失函数。
    它由两部分组成：
    1. 分数图损失 (score_loss): 使用 Dice Loss 计算，用于区分文本和背景。
    2. 几何图损失 (geo_loss): 使用一个简化的 IoU 损失（基于L1距离）计算，用于回归边界框的位置。
    """
    def __init__(self, lambda_iou=5.0):
        """
        初始化 EAST 损失函数。
        :param lambda_iou: float, IoU 损失的权重。
        """
        print("lambda_iou-----------------------------------------", lambda_iou)
        super(EASTLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.lambda_iou = lambda_iou

    def forward(self, y_true_score, y_pred_score, y_true_geo, y_pred_geo):
        """
        计算总的 EAST 损失。
        :param y_true_score: 真值分数图，形状为 (批次, 1, H, W)。
        :param y_pred_score: 预测分数图，形状为 (批次, 1, H, W)。
        :param y_true_geo: 真值几何图，形状为 (批次, 5, H, W)。
        :param y_pred_geo: 预测几何图，形状为 (批次, 5, H, W)。
        :return: tuple, 包含总损失、分数图损失和几何图损失。
        """
        # 计算分数图的损失
        score_loss = self.dice_loss(y_pred_score, y_true_score)
        
        # 创建一个掩码，只在正样本区域计算几何损失
        mask = (y_true_score > 0).float()

        # y_pred_geo 已经是 ReLU 后的绝对像素距离 (N, 4, H, W)
        # y_true_geo 也是绝对像素真值 (N, 4, H, W)
        # 直接算，不需要 sigmoid * scale

        # geo_loss = self.iou_loss_real(y_pred_geo, y_true_geo, mask)
        # 在 log 空间直接用 Smooth L1
        geo_loss = self.smooth_l1_geo(y_pred_geo, y_true_geo, mask)
        
        # 总损失
        total_loss = score_loss + self.lambda_iou * geo_loss
        
        return total_loss, score_loss, geo_loss

    def smooth_l1_geo(self, pred, target, mask):
        eps = 1e-6
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        # 扩展 mask 到 4 通道
        mask_4ch = mask.unsqueeze(1).expand_as(pred)

        loss = nn.functional.smooth_l1_loss(pred * mask_4ch, target * mask_4ch, reduction='sum')
        num_pos = torch.sum(mask) * 4 + eps  # 4个通道
        return loss / num_pos

    def iou_loss(self, pred_aabb, true_aabb, mask):
        """
        一个基于 AABB 的 L1 距离的简化版 IoU 损失。
        一个更精确的版本会为每个像素计算预测边界框和真实边界框的交集与并集。
        :param pred_aabb: 预测的AABB几何图。
        :param true_aabb: 真值的AABB几何图。
        :param mask: 只在文本区域计算损失的掩码。
        :return: float, 计算出的几何损失。
        """
        # 计算距离的 L1 损失
        l1_loss = torch.abs(pred_aabb - true_aabb)
        
        # 应用掩码，只考虑正样本像素
        masked_l1 = l1_loss * mask
        
        # 按正样本像素的数量进行归一化
        num_positive_pixels = torch.sum(mask)
        if num_positive_pixels == 0:
            return torch.tensor(0.0, device=pred_aabb.device)
            
        return torch.sum(masked_l1) / num_positive_pixels

    def iou_loss_real(self, pred_aabb, true_aabb, mask):
        # --- 这里的逻辑必须加固，防止 ReLU 导致的 0 值爆炸 ---
        eps = 1e-6  # 稍微大一点的 epsilon

        # 现在的输入形状直接就是 (N, 4, H, W)
        d_top_pred = pred_aabb[:, 0]
        d_right_pred = pred_aabb[:, 1]
        d_bottom_pred = pred_aabb[:, 2]
        d_left_pred = pred_aabb[:, 3]

        d_top_true = true_aabb[:, 0]
        d_right_true = true_aabb[:, 1]
        d_bottom_true = true_aabb[:, 2]
        d_left_true = true_aabb[:, 3]

        # 计算交集 (Intersection)
        inter_w = torch.min(d_left_pred, d_left_true) + torch.min(d_right_pred, d_right_true)
        inter_h = torch.min(d_top_pred, d_top_true) + torch.min(d_bottom_pred, d_bottom_true)
        inter_area = inter_w * inter_h

        # 计算并集 (Union)
        area_pred = (d_left_pred + d_right_pred) * (d_top_pred + d_bottom_pred)
        area_true = (d_left_true + d_right_true) * (d_top_true + d_bottom_true)
        union_area = area_pred + area_true - inter_area

        # [安全锁] 防止分母为 0
        union_area = torch.clamp(union_area, min=eps)

        # 计算 IoU
        iou = (inter_area + eps) / union_area

        # [安全锁] 防止 log(0)
        iou = torch.clamp(iou, min=eps, max=1.0)

        iou_loss = -torch.log(iou)

        # 只计算 mask 区域
        if mask.dim() == 4:
            mask = mask.squeeze(1)

        iou_loss = iou_loss * mask
        num_pos = torch.sum(mask) + eps  # 防止无正样本时除零

        return torch.sum(iou_loss) / num_pos
