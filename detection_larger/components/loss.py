"""沿用原 EAST 的 Dice + lambda_iou × Smooth L1，只扩展 ### 掩码。"""
from detection.components.loss import DiceLoss, EASTLoss as BaseEASTLoss


class EASTLoss(BaseEASTLoss):
    def forward(self, y_true_score, y_pred_score, y_true_geo, y_pred_geo, training_mask=None):
        if training_mask is not None:
            if training_mask.ndim == 3:
                training_mask = training_mask.unsqueeze(1)
            y_true_score = y_true_score * training_mask
            y_pred_score = y_pred_score * training_mask
        return super().forward(y_true_score, y_pred_score, y_true_geo, y_pred_geo)
