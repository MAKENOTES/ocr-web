"""统一检测的多尺度 EAST 网络，支持按配置切换骨干并接受任意 32 倍数高宽输入。"""
import torch
import torch.nn as nn

from detection.models.backbones import DEFAULT_BACKBONE, build_backbone


# 融合头三级输出通道；与 ResNet18 原实现一致，切换骨干时解码侧结构保持不变。
FUSE_CHANNELS = (128, 64, 32)


class EASTModel(nn.Module):
    def __init__(self, pretrained=True, backbone=DEFAULT_BACKBONE):
        """构建指定骨干与多尺度融合层；ResNet18 的参数名称和形状保持历史契约。"""
        super().__init__()
        layers, channels = build_backbone(backbone, pretrained)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.backbone_name = backbone
        channels1, channels2, channels3, channels4 = channels
        fuse3_channels, fuse2_channels, fuse1_channels = FUSE_CHANNELS

        # 逐级融合浅层定位信息和深层语义信息；输入通道随骨干变化，输出通道固定。
        self.up4 = nn.ConvTranspose2d(channels4, channels3, 2, stride=2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(channels3 + channels3, fuse3_channels, 3, padding=1),
            nn.BatchNorm2d(fuse3_channels), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(fuse3_channels, fuse2_channels, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(fuse2_channels + channels2, fuse2_channels, 3, padding=1),
            nn.BatchNorm2d(fuse2_channels), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(fuse2_channels, fuse1_channels, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(fuse1_channels + channels1, fuse1_channels, 3, padding=1),
            nn.BatchNorm2d(fuse1_channels), nn.ReLU())

        self.score_conv = nn.Conv2d(fuse1_channels, 1, 1)
        self.geo_conv = nn.Conv2d(fuse1_channels, 4, 1)

    def forward(self, x):
        """将 RGB 批次映射为步长 4 的置信度图及四通道 log1p 几何距离图。"""
        f0 = self.layer0(x)           # /4
        f1 = self.layer1(f0)          # /4
        f2 = self.layer2(f1)          # /8
        f3 = self.layer3(f2)          # /16
        f4 = self.layer4(f3)          # /32

        y = self.up4(f4)              # /16
        y = self.fuse3(torch.cat([y, f3], dim=1))
        y = self.up3(y)               # /8
        y = self.fuse2(torch.cat([y, f2], dim=1))
        y = self.up2(y)               # /4
        y = self.fuse1(torch.cat([y, f1], dim=1))

        return torch.sigmoid(self.score_conv(y)), self.geo_conv(y)


def create_model(pretrained=False, backbone=DEFAULT_BACKBONE):
    """按骨干名称构建评估用网络；默认 ResNet18，与历史检查点一致。"""
    return EASTModel(pretrained=pretrained, backbone=backbone)
