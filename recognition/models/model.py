"""固定高度 32 的 CRNN：Mobile 风格 CNN、单层双向 GRU 和 CTC 分类头。

输入 [B,C,32,W]，输出 [T,B,类别数] 对数概率。传入真实 img_widths 时，
BiGRU 跳过补边，输出尾部固定为 blank；参数结构与历史 ResNet/BiLSTM 权重不兼容。
速度取决于字典、图片宽度、后端和硬件，不能由结构直接保证固定加速倍数。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 1. 深度可分离卷积模块

class DepthwiseSeparableConv(nn.Module):
    """逐通道空间卷积后接 1×1 通道混合、BatchNorm 和 ReLU6。
    相同输入输出通道下，卷积乘加量约为标准卷积的 1/out_ch + 1/kernel_size²。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 2. 轻量级 CNN 特征提取器

class MobileFeatureExtractor(nn.Module):
    """将 [B,nc,32,W] 转为 [B,output_channel,1,T]，默认输出通道为 256。

    stem 步长 2 后接步长 2 的池化，宽度 T=floor((W+1)/4)；
    三个后续模块只继续缩小高度，依次为 4、2、1，不再缩小宽度。"""

    def __init__(self, input_channel=1, output_channel=256):
        super().__init__()

        # 首层标准卷积将输入映射为 32 通道，同时缩小高宽。
        self.stem = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # MaxPool: 进一步下采样宽度
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 1: 32 -> 64, H/2, W不变
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=(2, 1), padding=1),
            DepthwiseSeparableConv(64, 64, stride=1, padding=1),
        )

        # Block 2: 64 -> 128, H/2, W不变
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=(2, 1), padding=1),
            DepthwiseSeparableConv(128, 128, stride=1, padding=1),
        )

        # Block 3: 128 -> 256, H/2, W不变  -> 输出 H=1
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=(2, 1), padding=1),
            DepthwiseSeparableConv(256, output_channel, stride=1, padding=1),
        )

        self.out_channels = output_channel

    def forward(self, x):
        # x: [B, nc, 32, W]
        x = self.stem(x)   # -> [B, 32, 16, W/2]
        x = self.pool(x)   # -> [B, 32,  8, W/4]
        x = self.block1(x)  # -> [B, 64,  4, W/4]
        x = self.block2(x)  # -> [B,128,  2, W/4]
        x = self.block3(x)  # -> [B,256,  1, W/4]
        return x


# 3. 长度感知 CRNN

class CRNN(nn.Module):
    """Mobile 风格 CNN 与单层 BiGRU；nh 是每方向隐藏维度，默认 256。
    调用 forward 时传入 img_widths 才启用长度感知与尾部 blank 掩码。"""

    def __init__(self, img_height, nc, n_class, nh=256):
        super(CRNN, self).__init__()
        assert img_height == 32, "CRNN 输入高度必须固定为 32"

        # CNN 特征维度
        cnn_out_channels = 256

        self.cnn = MobileFeatureExtractor(
            input_channel=nc, output_channel=cnn_out_channels
        )

        # CNN → RNN Dropout
        self.cnn_dropout = nn.Dropout(p=0.1)

        # 单层双向 GRU；两个方向的输出拼接后送入分类头。
        self.rnn = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=nh,
            num_layers=1,
            dropout=0.0,  # 单层不需要 inter-layer dropout
            bidirectional=True,
            batch_first=False,
        )

        # RNN → Linear Dropout
        self.rnn_dropout = nn.Dropout(p=0.1)

        # 分类头
        self.embedding = nn.Linear(nh * 2, n_class)

        self._initialize_weights()

    @staticmethod
    def _calc_cnn_output_width(pixel_width: int) -> int:
        """
        精确计算单张图片经过 CNN 后的时间步数（宽度方向）。
        对应 MobileFeatureExtractor 的下采样路径：
          stem (Conv k=3, s=2, p=1)  ->  floor((W + 2*1 - 3) / 2) + 1
          pool (MaxPool k=2, s=2)    ->  floor((W_stem - 2) / 2) + 1
          block1~3 宽度 stride=1，不改变宽度
        """
        # stem: Conv2d(k=3, s=2, p=1)
        w = (pixel_width + 2 * 1 - 3) // 2 + 1
        # pool: MaxPool2d(k=2, s=2)
        w = (w - 2) // 2 + 1
        return w

    @staticmethod
    def _calc_cnn_output_widths(pixel_widths: torch.Tensor) -> torch.Tensor:
        """向量化计算每张图片经过 CNN 后的有效时间步数。"""
        widths = pixel_widths.to(dtype=torch.long)
        widths = (widths + 2 * 1 - 3) // 2 + 1
        widths = (widths - 2) // 2 + 1
        return widths

    def forward(self, input, img_widths=None):
        """
        Args:
            input: [B, C, 32, W] 图像张量（含 padding）
            img_widths: [B] 每张图片 padding 前的真实像素宽度（可选）
        Returns:
            output: [SeqLen, B, NumClasses] log_softmax 后的输出
        """
        conv = self.cnn(input)  # [B, 256, 1, W/4]

        b, c, h, w = conv.size()
        assert h == 1, f"CNN 输出高度必须为 1，当前为 {h}"

        conv = conv.squeeze(2)  # [B, 256, W/4]
        conv = conv.permute(2, 0, 1)  # [SeqLen, B, 256]

        conv = self.cnn_dropout(conv)

        seq_len = conv.size(0)
        real_seq_lens = None

        # 双向 GRU 必须真正跳过右侧 padding。仅将 padding 特征置零并不够：
        # 训练后的 GRU bias 会让反向隐状态在零输入区继续变化，进而污染有效字符。
        if img_widths is not None:
            real_seq_lens = self._calc_cnn_output_widths(img_widths)
            real_seq_lens = real_seq_lens.clamp(min=1, max=seq_len)
            packed_conv = pack_padded_sequence(
                conv,
                real_seq_lens.detach().cpu(),
                enforce_sorted=False,
            )
            packed_rnn_out, _ = self.rnn(packed_conv)
            rnn_out, _ = pad_packed_sequence(
                packed_rnn_out,
                total_length=seq_len,
            )
        else:
            rnn_out, _ = self.rnn(conv)

        rnn_out = self.rnn_dropout(rnn_out)
        output = self.embedding(rnn_out)  # [SeqLen, B, n_class]

        # CTC/解码看到的 padding 时间步固定为 blank。
        if real_seq_lens is not None:
            time_indices = torch.arange(seq_len, device=output.device).unsqueeze(1)
            padding_mask = time_indices >= real_seq_lens.to(output.device).unsqueeze(0)
            padding_output = torch.full_like(output, -1e9)
            padding_output[:, :, 0] = 0.0
            output = torch.where(padding_mask.unsqueeze(2), padding_output, output)

        return F.log_softmax(output, dim=2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.GRU, nn.LSTM)):
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 4. 随机输入的 CPU 前向计时，不评估识别精度

if __name__ == '__main__':
    import time

    IMG_H = 32  # 计时样例高度，单位 px。
    IMG_W = 512  # 计时样例有效宽度，单位 px。
    NUM_CHANNELS = 1  # 随机灰度样例通道数，不代表训练默认 RGB。
    NUM_CLASSES = 6000  # 计时样例类别数，不从真实字典推导。
    BATCH_SIZE = 8  # 随机计时样例每批图片数。

    device = torch.device('cpu')  # 专门测 CPU 速度

    # 轻量模型
    model_lite = CRNN(img_height=IMG_H, nc=NUM_CHANNELS, n_class=NUM_CLASSES).to(device)
    model_lite.eval()

    dummy = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMG_H, IMG_W).to(device)
    dummy_widths = torch.full((BATCH_SIZE,), IMG_W, dtype=torch.int32)

    # 参数量统计
    total_params = sum(p.numel() for p in model_lite.parameters())
    print(f"{'='*50}")
    print(f"轻量模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_lite(dummy, img_widths=dummy_widths)

    # 速度测试
    N_RUNS = 50  # 正式计时的前向次数，不含预热。
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_RUNS):
            output = model_lite(dummy, img_widths=dummy_widths)
    elapsed = time.perf_counter() - start

    print(f"输入尺寸: {dummy.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"宽度下采样倍率: {IMG_W / output.shape[0]:.1f}x")
    print(f"CPU 推理: {elapsed/N_RUNS*1000:.1f} ms/batch ({BATCH_SIZE} 张)")
    print(f"每张图片: {elapsed/N_RUNS/BATCH_SIZE*1000:.1f} ms")

    seq_len = output.shape[0]
    # 连续重复字符需要 blank 分隔；此处只是保守估计，不是所有文本的字符上限。
    max_chars = (seq_len - 1) // 2
    print(f"时间步: {seq_len}, 最大可识别字符数: ~{max_chars}")
    print(f"{'='*50}")
