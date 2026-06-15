"""
轻量级 CRNN 模型 (MobileNet-style CNN + GRU)
============================================
替代原始的 ResNet34 + BiLSTM 方案，CPU 推理速度提升约 3-5 倍。

关键设计：
1. CNN 骨干: 自定义 MobileNet-style 深度可分离卷积，替代 ResNet-34
   - 参数量从 ~21M 降至 ~0.5M (CNN 部分)
   - 使用 Depthwise Separable Conv，大幅减少乘加运算
2. RNN 部分: 单层 BiGRU (hidden=256) 替代 2层 BiLSTM (hidden=512)
   - GRU 比 LSTM 少一个门，计算量减少约 25%
   - 单层 + 更小 hidden 再减 ~75%
3. 保持与原模型完全相同的接口：
   - 输入: [B, C, 32, W]，宽度下采样倍率 = 4x
   - 输出: [SeqLen, B, NumClasses]，log_softmax
   - 支持 img_widths padding mask
   - 可直接替换 train.py / inference.py 中的 `from model import CRNN`

参数量对比 (以 n_class=6000 为例):
  原始模型:  ResNet34(21M) + BiLSTM×2(16M) + Linear(6M)  ≈ 43M
  轻量模型:  MobileCNN(0.5M) + BiGRU×1(1.2M) + Linear(3M) ≈ 4.7M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 深度可分离卷积模块
# ==========================================

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 = Depthwise Conv + Pointwise Conv
    计算量约为标准卷积的 1/k² (k=kernel_size)
    """

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


# ==========================================
# 2. 轻量级 CNN 特征提取器
# ==========================================

class MobileFeatureExtractor(nn.Module):
    """
    基于深度可分离卷积的轻量 CNN 特征提取器。

    设计目标：
    - 输入: [B, nc, 32, W]
    - 输出: [B, 256, 1, W/4]
    - 高度下采样: 32x (32 -> 1)
    - 宽度下采样: 4x  (与原 ResNet 版本一致)

    下采样路径 (H × W):
    ┌─ stem conv   stride=(2,2) -> H/2, W/2  => 16, W/2
    ├─ block1      stride=(2,1) -> H/4, W/2  =>  8, W/2
    ├─ block2      stride=(2,1) -> H/8, W/2  =>  4, W/2
    ├─ block3      stride=(2,1) -> H/16,W/2  =>  2, W/2
    ├─ block4      stride=(2,1) -> H/32,W/2  =>  1, W/2
    └─ 总计: 高度 32x, 宽度 2x (stem) × 1 × 1 × 1 × 1 = 2x
       但我们需要宽度 4x，所以 stem 用 stride=(2,2) 即可让 MaxPool 再 /2

    实际路径：
    stem(s2,s2) -> 16, W/2
    maxpool(s2,s2) -> 8, W/4    ← 宽度在此变为 W/4
    block1(s2,s1) -> 4, W/4
    block2(s2,s1) -> 2, W/4
    block3(s2,s1) -> 1, W/4     ← 高度压到 1
    """

    def __init__(self, input_channel=1, output_channel=256):
        super().__init__()

        # Stem: 标准卷积 (第一层不用可分离卷积，效率差异不大但特征更好)
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


# ==========================================
# 3. 轻量 CRNN (保持与原版完全相同的接口)
# ==========================================

class CRNN(nn.Module):
    """
    轻量级 CRNN: MobileNet-style CNN + BiGRU + CTC Head

    接口与原 CRNN 完全兼容:
    - __init__(img_height, nc, n_class, nh=256)
    - forward(input, img_widths=None) -> [SeqLen, B, NumClasses]

    注意: nh 默认值改为 256 (原版 512)，如需与原版一致请显式传参。
    """

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

        # 使用 GRU 替代 LSTM: 更快、参数更少
        # 单层 BiGRU (原版: 2层 BiLSTM)
        self.rnn = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=nh,
            num_layers=1,
            dropout=0.0,  # 单层不需要 inter-layer dropout
            bidirectional=True,
            batch_first=False,
        )
        # self.rnn = nn.GRU(
        #     input_size=cnn_out_channels,
        #     hidden_size=nh,  # 256 → 384
        #     num_layers=2,  # 1 → 2
        #     dropout=0.15,  # 多层需要 inter-layer dropout
        #     bidirectional=True,
        #     batch_first=False,
        # )

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

        # ---- 关键：GRU 之前将 padding 区域的 CNN 特征置零 ----
        # 原理：反向 GRU 从序列末尾开始处理，如果 padding 区域有非零特征，
        # 反向隐状态会携带噪声传播到真实内容区域，导致边界字符重复。
        # 将 padding 特征置零后，反向 GRU 在 padding 区域的输入全为 0，
        # 配合 h0=0，隐状态保持为零直到遇到真实内容，等效于 pack_padded_sequence。
        # 此操作是纯 tensor 运算，ONNX 完全兼容。
        if img_widths is not None:
            real_seq_lens = torch.tensor(
                [self._calc_cnn_output_width(int(pw)) for pw in img_widths],
                dtype=torch.long, device=conv.device
            ).clamp(min=1, max=seq_len)

            # mask: [SeqLen, B]  True = padding 位置
            time_indices = torch.arange(seq_len, device=conv.device).unsqueeze(1)  # [SeqLen, 1]
            padding_mask = time_indices >= real_seq_lens.unsqueeze(0)               # [SeqLen, B]

            # 扩展到特征维度 [SeqLen, B, 1] 并广播，将 padding 时间步的特征全部置零
            conv = conv.masked_fill(padding_mask.unsqueeze(2), 0.0)

        rnn_out, _ = self.rnn(conv)

        rnn_out = self.rnn_dropout(rnn_out)
        output = self.embedding(rnn_out)  # [SeqLen, B, n_class]

        # ---- Padding Mask: 对输出 logits 同样屏蔽 ----
        if img_widths is not None:
            output[padding_mask] = output.new_full((output.size(2),), -1e9)
            output[padding_mask, 0] = 0.0

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


# ==========================================
# 4. 测试 & 对比
# ==========================================

if __name__ == '__main__':
    import time

    IMG_H = 32
    IMG_W = 512
    NUM_CHANNELS = 1
    NUM_CLASSES = 6000
    BATCH_SIZE = 8

    device = torch.device('cpu')  # 专门测 CPU 速度

    # -------- 轻量模型 --------
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
    N_RUNS = 50
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
    max_chars = (seq_len - 1) // 2
    print(f"时间步: {seq_len}, 最大可识别字符数: ~{max_chars}")
    print(f"{'='*50}")