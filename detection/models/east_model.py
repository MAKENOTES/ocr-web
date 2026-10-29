"""统一检测的 ResNet18 EAST 网络定义，支持任意 32 倍数的高宽输入。"""
import torch
import torch.nn as nn
import torchvision.models as models


class EASTModel(nn.Module):
    def __init__(self, pretrained=True):
        """构建 ResNet18 与多尺度融合层，保持 ResNet18 参数名称和形状。"""
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 64ch,  /4
        self.layer2 = resnet.layer2  # 128ch, /8
        self.layer3 = resnet.layer3  # 256ch, /16
        self.layer4 = resnet.layer4  # 512ch, /32

        # 逐级融合浅层定位信息和深层语义信息。
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.score_conv = nn.Conv2d(32, 1, 1)
        self.geo_conv = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        """将 RGB 批次映射为步长 4 的置信度图及四通道 log1p 几何距离图。"""
        f0 = self.layer0(x)           # /4
        f1 = self.layer1(f0)          # /4,  64ch
        f2 = self.layer2(f1)          # /8,  128ch
        f3 = self.layer3(f2)          # /16, 256ch
        f4 = self.layer4(f3)          # /32, 512ch

        y = self.up4(f4)              # /16
        y = self.fuse3(torch.cat([y, f3], dim=1))
        y = self.up3(y)               # /8
        y = self.fuse2(torch.cat([y, f2], dim=1))
        y = self.up2(y)               # /4
        y = self.fuse1(torch.cat([y, f1], dim=1))

        return torch.sigmoid(self.score_conv(y)), self.geo_conv(y)


def create_model(pretrained=False):
    return EASTModel(pretrained=pretrained)
