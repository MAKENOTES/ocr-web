import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    """
    基于 ResNet-34 的特征提取器
    针对文本识别进行了 stride 修改：
    1. 适配 32px 高度输入 -> 输出高度为 1
    2. 适配单通道灰度输入
    3. [优化] 提升宽度分辨率，总宽度下采样倍率从 8x 降低为 4x，
       以便容纳更长的文本序列。
    """

    def __init__(self, input_channel=1, output_channel=512):
        super(ResNetFeatureExtractor, self).__init__()

        # 1. 加载预训练的 ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # 2. 适配输入通道 (RGB -> Grayscale)
        if input_channel == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 3. 截取 Layer1 - Layer4
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 4. 修改 Stride (关键优化)
        # 目标: 高度 stride=32 (变为1), 宽度 stride=4 (保留更多细节)

        # Layer 2: 原 stride=(2,2) -> 改为 (2, 1) [压高度，保宽度]
        # 这将使总宽度下采样率从 8x 变为 4x
        self.layer2[0].conv1.stride = (2, 1)
        self.layer2[0].downsample[0].stride = (2, 1)

        # Layer 3: 原 stride=(2,2) -> 改为 (2, 1)
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4: 原 stride=(2,2) -> 改为 (2, 1)
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

        self.out_channels = 512

    def forward(self, x):
        # x shape: [Batch, 1, 32, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # -> [B, 64, 16, W/2] (Pool stride=2)

        x = self.layer1(x)  # -> [B, 64, 16, W/2] (Layer1 stride=1)
        x = self.layer2(x)  # -> [B, 128, 8, W/2] (Layer2 stride=(2,1)) *Modified*
        x = self.layer3(x)  # -> [B, 256, 4, W/2] (Layer3 stride=(2,1)) *Modified*
        x = self.layer4(x)  # -> [B, 512, 2, W/2] ? Wait, let's trace carefully.

        # Correction on stride logic:
        # Input H=32
        # Conv1(s2) -> 16
        # MaxPool(s2) -> 8
        # Layer1(s1) -> 8
        # Layer2(s2,1) -> 4
        # Layer3(s2,1) -> 2
        # Layer4(s2,1) -> 1
        # Final H = 1. Perfect.

        # Width Stride:
        # Conv1(s2) * MaxPool(s2) * Layer2(s1) * Layer3(s1) * Layer4(s1) = 4.
        # Final W = InputW / 4.

        return x


class CRNN(nn.Module):
    """
    CRNN 架构: ResNet34 + BiLSTM + CTC Head
    """

    def __init__(self, img_height, nc, n_class, nh=512):
        super(CRNN, self).__init__()
        assert img_height == 32, "ResNet 版本的 CRNN 输入高度必须固定为 32"

        self.cnn = ResNetFeatureExtractor(input_channel=nc, output_channel=512)

        # CNN → LSTM 之间加 Dropout，防止CNN特征过拟合传导到LSTM
        self.cnn_dropout = nn.Dropout(p=0.2)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=nh,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=False
        )

        # LSTM → Linear 之间也加 Dropout
        self.rnn_dropout = nn.Dropout(p=0.2)

        self.embedding = nn.Linear(nh * 2, n_class)
        self._initialize_weights()

    def forward(self, input, img_widths=None):
        """
        Args:
            input: [B, C, 32, W] 图像张量（含 padding）
            img_widths: [B] 每张图片 padding 前的真实像素宽度（可选）。
                        提供时，会将 padding 区域的输出强制设为 blank，
                        防止 CTC 在 padding 区域产生错误预测。
        Returns:
            output: [SeqLen, B, NumClasses] log_softmax 后的输出
        """
        conv = self.cnn(input)

        b, c, h, w = conv.size()
        assert h == 1, f"CNN 输出高度必须为 1，当前为 {h}"

        conv = conv.squeeze(2)  # [B, 512, W]
        conv = conv.permute(2, 0, 1)  # [W, B, 512]

        conv = self.cnn_dropout(conv)  # CNN→LSTM 之间 Dropout

        rnn_out, _ = self.rnn(conv)
        rnn_out = self.rnn_dropout(rnn_out)  # LSTM→Linear 之间 Dropout
        output = self.embedding(rnn_out)  # [W, B, NumClasses]

        # ---- Attention Mask: 将 padding 区域的 logits 强制设为 blank ----
        if img_widths is not None:
            seq_len = output.size(0)
            # CNN 宽度下采样倍率为 4，计算每个样本的真实时间步数
            real_seq_lens = (img_widths.to(output.device) // 4).clamp(min=1)
            # 构建 mask: [SeqLen, Batch]，True 表示 padding 位置
            time_indices = torch.arange(seq_len, device=output.device).unsqueeze(1)  # [W, 1]
            padding_mask = time_indices >= real_seq_lens.unsqueeze(0)  # [W, B]
            # padding 位置: blank(idx=0) 的 logit 设为大值，其余设为极小值
            # 这样 softmax 后 padding 位置几乎 100% 输出 blank
            output[padding_mask] = output.new_full((output.size(2),), -1e9)
            output[padding_mask, 0] = 0.0  # blank 类别的 logit 设为 0（相对最大）

        return F.log_softmax(output, dim=2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ----------------------------------------
# 测试代码 (验证维度是否正确)
# ----------------------------------------
if __name__ == '__main__':
    # 模拟配置
    IMG_H = 32
    IMG_W = 1024  # 你之前提到的长图宽度
    NUM_CHANNELS = 1
    NUM_CLASSES = 6000 # 常用汉字数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 创建模型
    model = CRNN(img_height=IMG_H, nc=NUM_CHANNELS, n_class=NUM_CLASSES).to(device)
    model.eval()

    # 2. 创建假数据
    dummy_input = torch.randn(2, NUM_CHANNELS, IMG_H, IMG_W).to(device)

    # 3. 前向传播
    with torch.no_grad():
        output = model(dummy_input)

    print("-" * 30)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # 预期: [128, 2, 6001]

    # 4. 计算下采样倍率和时间步
    seq_len = output.shape[0]
    print(f"Sequence Length (Time Steps): {seq_len}")
    print(f"Downsampling Ratio (Width): {IMG_W / seq_len:.1f}x")

    # 5. 容量检查
    # CTC 理论公式: SeqLen >= 2 * LabelLen + 1
    max_chars = (seq_len - 1) // 2
    print(f"Max specific characters allowed: approx {max_chars}")
    print("-" * 30)

    if max_chars < 50:
         print("警告: 对于1024宽度的图片，能容纳的字符数较少。")
         print("建议: 如果你的文字非常密集，请考虑修改 ResNetFeatureExtractor 中 layer2 的 stride。")