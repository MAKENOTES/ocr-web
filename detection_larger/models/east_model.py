"""独立维护文档检测的 ResNet18 EAST 网络及整图检查点契约。"""
import torch

import torch.nn as nn
import torchvision.models as models
from detection_larger.components.transforms import STRIDE, normalize_input_size

MODEL_FORMAT = "document-east-resnet18-full-image-v2"  # 整图模型契约版本，保持已有文档检测权重兼容。
ONNX_CONFIG_KEY = "docdet_config"  # ONNX 元数据中的文档检测配置字段名。


class EASTModel(nn.Module):
    def __init__(self, pretrained=True):
        """构建 ResNet18 与多尺度融合层，保持已有检查点的参数名称和形状。"""
        super().__init__()
        # 主干输出四级特征，供逐级上采样融合。
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
            nn.Conv2d(256 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.score_conv = nn.Conv2d(32, 1, 1)
        self.geo_conv   = nn.Conv2d(32, 4, 1)

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

        score = torch.sigmoid(self.score_conv(y))
        geo   = self.geo_conv(y)
        return score, geo


def create_model(pretrained=False):
    return EASTModel(pretrained=pretrained)


def checkpoint_config(input_size):
    """记录调用方尺寸及预处理/几何契约；加载时校验格式，推理尺寸仍由入口常量决定。"""
    return {"format": MODEL_FORMAT, "input_size": list(normalize_input_size(input_size)),
            "stride": STRIDE, "geometry": "log1p-trbl-feature-pixels",
            "preprocess": "rgb-imagenet-center-auto-v1"}


def validate_config(config):
    if config.get("format") != MODEL_FORMAT:
        raise ValueError("Expected full-image EAST checkpoint; old tiled checkpoints require their original inference code")
    expected = checkpoint_config(config["input_size"])
    if config != expected:
        raise ValueError(f"Incompatible document model config: {config}; expected {expected}")


def save_checkpoint(model, path, input_size):
    payload = {"config": checkpoint_config(input_size), "model": model.state_dict()}
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"Expected a document detection checkpoint with config: {path}")
    validate_config(checkpoint["config"])
    model = create_model()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
