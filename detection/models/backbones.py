"""统一检测的骨干注册表：在 ResNet18 与 MobileNetV3-Large 之间切换。

骨干只负责输出 `/4`、`/8`、`/16`、`/32` 四级特征并声明各层通道数，EAST 融合头
由 east_model 按通道数构建。新增骨干时在 BACKBONE_BUILDERS 和 BACKBONE_CHANNELS
注册，并同步 detection/utils/config.py 的 MODEL_FORMATS；runtime 配置快照必须
自包含，因此不直接导入本模块，两处名称由测试保证一致。
"""
import torch.nn as nn
from torchvision import models


RESNET18 = "resnet18"  # 默认骨干；现有检查点和 ONNX 模型都基于它。
MOBILENET_V3_LARGE = "mobilenet_v3_large"  # 轻量骨干；通道更少，CPU 前向更快。
DEFAULT_BACKBONE = MOBILENET_V3_LARGE  # 新建训练模型的默认骨干；改这里切换训练骨干，推理和导出以检查点记录为准。
BACKBONE_CHOICES = (RESNET18, MOBILENET_V3_LARGE)  # 可选骨干名称，顺序仅用于报错提示。

# 各骨干 `/4`、`/8`、`/16`、`/32` 特征的通道数，融合头据此构建。
BACKBONE_CHANNELS = {
    RESNET18: (64, 128, 256, 512),
    MOBILENET_V3_LARGE: (24, 40, 112, 160),
}


def _resnet18_layers(pretrained):
    """返回 ResNet18 的 /4 输入层和 /4、/8、/16、/32 四级特征模块。"""
    resnet = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    return (nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)


def _mobilenet_v3_large_layers(pretrained):
    """按步长切分 MobileNetV3-Large，返回 /4、/8、/16、/32 四级特征。

    `features[0:4]` 已下采样到 /4；`features[12]` 输出 /16 的 112 通道特征；
    `/32` 取到 `features[15]`（160 通道），末尾 1×1 升维到 960 通道的分类前
    卷积不参与检测，避免无谓的显存和导出体积。
    """
    features = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    ).features
    return (features[0:4], nn.Identity(), features[4:7], features[7:13], features[13:16])


# 骨干名称 → (pretrained) -> (layer0, layer1, layer2, layer3, layer4)。
BACKBONE_BUILDERS = {
    RESNET18: _resnet18_layers,
    MOBILENET_V3_LARGE: _mobilenet_v3_large_layers,
}


def build_backbone(name, pretrained):
    """按名称构建骨干，返回 (layer0, layer1, layer2, layer3, layer4) 和通道数元组。"""
    if name not in BACKBONE_BUILDERS:
        raise ValueError(f"未知检测骨干: {name!r}; 可选: {BACKBONE_CHOICES}")
    return BACKBONE_BUILDERS[name](pretrained), BACKBONE_CHANNELS[name]
