# 历史模型：CRNN 初始化

本文解释旧版 ResNet 与双向 LSTM 识别模型的初始化结构，供回查历史模型组成。

> 本文保留旧模型/旧批处理方案的说明，不作为当前实现。原文未注明版本时不可据此修改现有权重或预处理；当前契约见[架构](../../recognition.md)。

```python
    def __init__(self, img_height, nc, n_class, nh=512):
        super(CRNN, self).__init__()
        assert img_height == 32, "ResNet 版本的 CRNN 输入高度必须固定为 32"

        self.cnn = ResNetFeatureExtractor(input_channel=nc, output_channel=512)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=nh,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        self.embedding = nn.Linear(nh * 2, n_class + 1)
        self._initialize_weights()

```

这几行代码位于 `CRNN` 类的构造函数 `__init__` 中，主要负责**搭建网络的骨架**和**定义每一层的维度**。

下面我逐行拆解它们的含义和背后的逻辑：

---

### 1. 强制输入高度检查

```python
assert img_height == 32, "ResNet 版本的 CRNN 输入高度必须固定为 32"

```

* **含义**：这就好比是一个“安检门”。如果输入的图片高度不是 32 像素，程序直接报错并停止运行。
* **为什么要这样做？**
* 这是因为前面的 `ResNetFeatureExtractor` 的数学计算是写死的。它经过了 5 次高度下采样（stride 累计为 ）。
* **数学逻辑**：输入高度 32  下采样 32 = 输出高度 1。
* CRNN 的核心逻辑是将图片的高度压缩为 1，只保留宽度方向作为“时间序列”喂给 LSTM。如果不强制为 32（比如你传了 64），输出高度就是 2，后面的代码就会因为维度不匹配而崩掉。



---

### 2. 实例化 CNN (特征提取器)

```python
self.cnn = ResNetFeatureExtractor(input_channel=nc, output_channel=512)

```

* **含义**：初始化我们在上面定义的 ResNet 类。
* **参数**：
* `nc` (Number of Channels)：输入图片的通道数。如果是灰度图就是 1，彩色图就是 3。
* `512`：这是 CNN 最后输出的**特征深度**（Feature Map Depth）。这意味着对于图片上的每一个细小的切片，CNN 会提取出 512 个数值来描述它（比如是否有竖线、是否有圆弧等）。这个 512 将作为下一步 LSTM 的输入大小。



---

### 3. 实例化 RNN (序列建模)

```python
self.rnn = nn.LSTM(
    input_size=512,
    hidden_size=nh,
    num_layers=2,
    bidirectional=True,
    batch_first=False
)

```

这是核心的序列处理模块，使用长短期记忆网络（LSTM）：

* **`input_size=512`**：**关键对接点**。它必须等于 CNN 输出的通道数（512）。CNN 把图切成一条条宽为 1 的细条，每一条由 512 个特征表示，喂给 LSTM。
* **`hidden_size=nh`**：LSTM 内部的记忆容量。通常设为 256 或 512。数值越大，模型能记住的上下文越复杂，但计算量也越大。
* **`num_layers=2`**：堆叠了两层 LSTM。第一层的输出作为第二层的输入。深层网络能学到更抽象的语法或语义规则。
* **`bidirectional=True`**：**双向 LSTM**。
* 这是 OCR 的标配。识别一个字时，既要看前面的字（前向），也要看后面的字（后向）。
* *例子*：手写体如果不看上下文，很难分清是 "rn" 还是 "m"。双向网络能解决这个问题。


* **`batch_first=False`**：
* PyTorch LSTM 的默认输入格式是 `[Time_Step, Batch_Size, Feature]`。
* 如果设为 `True`，输入格式则是 `[Batch_Size, Time_Step, Feature]`。
* 代码后续用了 `permute(2, 0, 1)` 生成了 `[W, B, C]` 的格式，其中 W（宽度）代表时间步，所以这里保持 `False` 是为了配合后续的数据流。



---

### 4. 实例化分类头 (Embedding/Classifier)

```python
self.embedding = nn.Linear(nh * 2, n_class + 1)

```

* **含义**：这是一个全连接层（Linear Layer），它的作用是把 LSTM 提取的抽象特征映射到具体的字符类别上。
* **`nh * 2` (输入维度)**：
* 为什么要乘以 2？因为 LSTM 是**双向**的 (`bidirectional=True`)。
* 前向 LSTM 输出一个 `nh` 维向量，后向 LSTM 也输出一个 `nh` 维向量，PyTorch 会把它们拼接在一起，所以总维度是 `nh + nh`。


* **`n_class + 1` (输出维度)**：
* `n_class` 是你要识别的字符总数（比如 6000 个汉字）。
* **`+1` 是什么？** 这是 **CTC Loss** 必须的一个特殊占位符，称为 **"Blank" (空白符)**。
* CTC 算法需要这个 Blank 来区分相邻的相同字符（例如单词 "hello" 中的两个 'l'，在 CTC 路径中通常是 `l - blank - l`）。



---

### 5. 初始化权重

```python
self._initialize_weights()

```

* **含义**：调用类下面定义的 `_initialize_weights` 函数。
* **作用**：在训练开始前，给网络里的参数（卷积核、偏置等）赋予合理的初始值。
* 如果不初始化（默认随机），模型可能很难收敛（Loss 降不下去）。
* 代码里用了 `Xavier` 和 `Orthogonal` 等高级初始化方法，有助于模型更快地进入最佳训练状态。



### 总结数据流向

1. **图片输入** (Height=32)

2. **CNN** (`ResNetFeatureExtractor`)：把图“压扁”，高度变 1，深度变 512。

3. **LSTM** (`nn.LSTM`)：读取 512 维的特征序列，结合上下文，输出 `nh * 2` 维的特征。

4. **Linear** (`self.embedding`)：把 `nh * 2` 维特征翻译成 `n_class + 1` 个字符的概率分数。