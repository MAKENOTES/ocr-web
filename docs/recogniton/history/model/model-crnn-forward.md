# 历史模型：CRNN 前向传播

> 本文保留旧模型/旧批处理方案的说明，不作为当前实现。原文未注明版本时不可据此修改现有权重或预处理；当前契约见[架构](../../recognition.md)。

```python
    def forward(self, input):
        conv = self.cnn(input)

        b, c, h, w = conv.size()
        assert h == 1, f"CNN 输出高度必须为 1，当前为 {h}"

        conv = conv.squeeze(2)  # [B, 512, W]
        conv = conv.permute(2, 0, 1)  # [W, B, 512]

        rnn_out, _ = self.rnn(conv)
        output = self.embedding(rnn_out)

        return F.log_softmax(output, dim=2)
```

这段 `forward` 函数定义了数据在 CRNN 模型中的完整流向：**从图片输入到字符概率输出**。

这是整个模型最核心的“流水线”。我们可以把它拆解为三个主要阶段：**特征提取 (CNN)**  **维度重塑 (Reshape)**  **序列推理 (RNN + CTC)**。

以下是逐行深度解析：

### 第一阶段：CNN 特征提取

```python
conv = self.cnn(input)

```

* **操作**：将原始图片输入到之前定义的 `ResNetFeatureExtractor` 中。
* **输入 Shape**：`[Batch, Channel, 32, Width]`（例如 `[16, 1, 32, 1024]`）。
* **输出 Shape**：`[Batch, 512, 1, Width/4]`（例如 `[16, 512, 1, 256]`）。
* **含义**：图片的高度被压缩没了（变成了 1），深度变厚了（512），宽度缩小了（变为原来的 1/4）。这时候，原本的图片像素变成了高维的特征块。

---

### 第二阶段：维度检查与重塑（关键步骤）

```python
b, c, h, w = conv.size()
assert h == 1, f"CNN 输出高度必须为 1，当前为 {h}"

```

* **含义**：这是个“保险”。CRNN 的基本假设是图片高度被完全压缩，只剩下宽度方向作为时间序列。如果 ，说明前面的 CNN 没设计好，后面的代码会出错。

```python
conv = conv.squeeze(2)  # [B, 512, W]

```

* **操作**：`squeeze(2)` 表示移除第 2 个维度（也就是高度 h=1 的那个维度）。
* **变化**：从 4D 张量 `[Batch, 512, 1, Width]` 变为 3D 张量 `[Batch, 512, Width]`。
* **物理意义**：现在它不再被视为一张“图片”，而被视为一堆排列在一起的“特征条”。

```python
conv = conv.permute(2, 0, 1)  # [W, B, 512]

```

* **操作**：`permute` 重新排列维度的顺序。
* 原第 2 维 (Width)  变成第 0 维。
* 原第 0 维 (Batch)  变成第 1 维。
* 原第 1 维 (Channel/Feature)  变成第 2 维。


* **结果 Shape**：`[Width, Batch, 512]`。
* **为什么要这样做？**
* **核心逻辑**：CRNN 将图片的 **宽度 (Width)** 视为 **时间 (Time)**。
* PyTorch 的 `LSTM` 默认输入格式要求是 `[Time_Step, Batch_Size, Feature_Size]`。所以我们必须把 Width 放到第一维。



---

### 第三阶段：序列建模与分类

```python
rnn_out, _ = self.rnn(conv)

```

* **操作**：将调整好形状的数据喂给双向 LSTM。
* **输入**：`[Seq_Len, Batch, 512]` (这里的 Seq_Len 就是特征图的宽度)。
* **输出 (`rnn_out`)**：`[Seq_Len, Batch, Hidden_Size * 2]`。
* LSTM 输出了每个时间步（即图片横向的每一小段）的上下文特征。


* **`_` 是什么**：LSTM还会返回隐藏状态 `(h_n, c_n)`，但在 OCR 任务中我们需要的是整个序列的输出，不需要最后的那个状态，所以用下划线忽略掉。

```python
output = self.embedding(rnn_out)

```

* **操作**：通过全连接层（Linear Layer）进行分类。
* **变化**：将特征维度从 `Hidden_Size * 2` 映射到 `Class_Num + 1`。
* **输出 Shape**：`[Seq_Len, Batch, Class_Num + 1]`。
* **含义**：对于图片宽度的每一个切片，模型都打出了一个分数，表示它是“a”、“b”、“c”...还是“空白”的可能性。

---

### 第四阶段：概率归一化 (CTC 适配)

```python
return F.log_softmax(output, dim=2)

```

* **操作**：对最后一维（类别维度）做 Log Softmax。
* **含义**：
1. **Softmax**：把神经网络输出的任意数值（Logits）转换成概率（加起来等于 1）。
2. **Log**：取对数。


* **为什么？**
* 因为 PyTorch 的 **`nn.CTCLoss`** 明确要求输入必须是 **Log Probabilities**（对数概率），而不是原始概率或原始分数。这是为了数值稳定性（避免概率太小导致下溢出）。

