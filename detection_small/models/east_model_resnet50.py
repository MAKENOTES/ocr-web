import torch
import torch.nn as nn
import torchvision.models as models


class EASTModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256ch, /4
        self.layer2 = resnet.layer2  # 512ch, /8
        self.layer3 = resnet.layer3  # 1024ch, /16
        self.layer4 = resnet.layer4  # 2048ch, /32

        # 融合模块：上采样 + concat + conv
        self.up4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )

        self.score_conv = nn.Conv2d(128, 1, 1)
        self.geo_conv = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        f0 = self.layer0(x)  # /4
        f1 = self.layer1(f0)  # /4,  256ch
        f2 = self.layer2(f1)  # /8,  512ch
        f3 = self.layer3(f2)  # /16, 1024ch
        f4 = self.layer4(f3)  # /32, 2048ch

        y = self.up4(f4)  # /16
        y = self.fuse3(torch.cat([y, f3], dim=1))  # /16
        y = self.up3(y)  # /8
        y = self.fuse2(torch.cat([y, f2], dim=1))  # /8
        y = self.up2(y)  # /4
        y = self.fuse1(torch.cat([y, f1], dim=1))  # /4

        score = torch.sigmoid(self.score_conv(y))
        # geo = torch.relu(self.geo_conv(y))
        geo = self.geo_conv(y)  # 去掉 ReLU！log 空间的值可以是负数
        return score, geo