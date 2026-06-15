import torch
import torch.nn as nn
import torchvision.models as models


class EASTModel(nn.Module):
    """
    EAST 文本检测模型，使用 MobileNetV3-Large 作为骨干网络。

    MobileNetV3-Large features 层输出通道对照：
        features[0:2]   -> /2,  16ch
        features[2:4]   -> /4,  24ch   (对应 f1)
        features[4:7]   -> /8,  40ch   (对应 f2)
        features[7:13]  -> /16, 112ch  (对应 f3)
        features[13:17] -> /32, 960ch  (对应 f4，含最后的 1x1 conv 扩展)

    ★ 架构优化（对比旧版）：
      1. FPN 通道加宽：64→128, 32→64, 24→64，对齐 ResNet50 版本的容量比例
      2. 预测头加厚：在 1x1 输出前增加 3x3 conv，从 24ch 直接预测 → 64ch 中间层
      3. 空洞卷积上下文模块同步加宽到 64ch
      4. 整体参数量仍远小于 ResNet50 版本
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # ═══════════════════════════════════════════════════
        # 骨干网络：MobileNetV3-Large
        # ═══════════════════════════════════════════════════
        mobilenet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        )
        features = mobilenet.features  # Sequential of 17 layers (index 0~16)

        # 按下采样倍率分组
        self.stage0 = features[0:2]    # /2,  out: 16ch
        self.stage1 = features[2:4]    # /4,  out: 24ch
        self.stage2 = features[4:7]    # /8,  out: 40ch
        self.stage3 = features[7:13]   # /16, out: 112ch
        self.stage4 = features[13:17]  # /32, out: 960ch

        # ═══════════════════════════════════════════════════
        # FPN 自顶向下路径（★ 通道加宽）
        # ═══════════════════════════════════════════════════
        # 上采样 /32 → /16，与 stage3 (112ch) 拼接
        self.up4 = nn.ConvTranspose2d(960, 112, kernel_size=2, stride=2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(112 + 112, 128, 3, padding=1),   # ★ 64 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 上采样 /16 → /8，与 stage2 (40ch) 拼接
        self.up3 = nn.ConvTranspose2d(128, 40, kernel_size=2, stride=2)  # ★ 64 → 128 输入
        self.fuse2 = nn.Sequential(
            nn.Conv2d(40 + 40, 64, 3, padding=1),      # ★ 32 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 上采样 /8 → /4，与 stage1 (24ch) 拼接
        self.up2 = nn.ConvTranspose2d(64, 24, kernel_size=2, stride=2)   # ★ 32 → 64 输入
        self.fuse1 = nn.Sequential(
            nn.Conv2d(24 + 24, 64, 3, padding=1),      # ★ 24 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ═══════════════════════════════════════════════════
        # ★ 空洞卷积上下文模块（同步加宽到 64ch）
        # ═══════════════════════════════════════════════════
        self.context = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),   # ★ 24 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),   # ★ 24 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ═══════════════════════════════════════════════════
        # ★ 加厚预测头：增加 3x3 中间层
        # 旧版：24ch → 1x1 → score/geo（太薄，表达力不足）
        # 新版：64ch → 3x3 → 64ch → 1x1 → score/geo
        # ═══════════════════════════════════════════════════
        self.score_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.geo_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
        )

    def forward(self, x):
        # 骨干网络前向传播，提取多尺度特征
        f0 = self.stage0(x)    # /2,  16ch
        f1 = self.stage1(f0)   # /4,  24ch
        f2 = self.stage2(f1)   # /8,  40ch
        f3 = self.stage3(f2)   # /16, 112ch
        f4 = self.stage4(f3)   # /32, 960ch

        # FPN 自顶向下合并
        y = self.up4(f4)                           # /16, 112ch
        y = self.fuse3(torch.cat([y, f3], dim=1))  # cat→224ch → 128ch  ★
        y = self.up3(y)                            # /8,  40ch
        y = self.fuse2(torch.cat([y, f2], dim=1))  # cat→80ch  → 64ch   ★
        y = self.up2(y)                            # /4,  24ch
        y = self.fuse1(torch.cat([y, f1], dim=1))  # cat→48ch  → 64ch   ★

        # 空洞卷积上下文增强（残差连接）
        y = y + self.context(y)                    # 64ch

        # ★ 加厚预测头输出
        score = torch.sigmoid(self.score_head(y))  # (N, 1, H/4, W/4)
        geo   = self.geo_head(y)                   # (N, 4, H/4, W/4)
        return score, geo