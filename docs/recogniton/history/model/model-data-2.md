# 历史模型：宽度下采样

本文追踪旧识别模型宽度下采样的结构依据，供理解历史序列长度的计算方式。

> 本文保留旧模型/旧批处理方案的说明，不作为当前实现。原文未注明版本时不可据此修改现有权重或预处理；当前契约见[架构](../../recognition.md)。

“宽缩 4 倍”这个结论是由 **`conv1`**、**`maxpool`** 和 **后续 Layer 的 Stride 修改** 共同决定的。

简单来说：**前两层就把宽度缩了 4 倍，而后面所有的层特意修改了代码，不让它们继续缩宽度。**

下面是具体的代码证据追踪：

### 1. 哪里**造成**了 4 倍下采样？

真正的下采样发生在 ResNet 最开始的两个通用层（这部分代码直接继承自 `torchvision.models.resnet34`，虽然代码里没显式写出 stride，但这是标准 ResNet 的定义）：

1. **Conv1 (7x7 卷积)**
* 在标准 ResNet 中，`conv1` 的参数是 `stride=2`。
* **宽度变化**：
* *代码对应位置*：虽然这里修改了 input_channel，但并没有修改 stride，所以保持默认。


```python
# 继承自 models.resnet34，默认 stride=2
self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

```


2. **MaxPool (3x3 池化)**
* 在标准 ResNet 中，紧接 Conv1 之后的 `maxpool` 参数也是 `stride=2`。
* **宽度变化**：
* *代码对应位置*：


```python
self.maxpool = resnet.maxpool # 默认 stride=2

```



**结论**：经过这两步，，宽度已经变成了原来的 1/4。

---

### 2. 为什么**仅仅**是 4 倍（而不是 8倍、16倍...）？

在标准的 ResNet-34 中，Layer2、Layer3、Layer4 都会继续进行 2倍下采样，最终宽度会变成 input 的 1/32。

**但是**，你的代码中显式地**修改了 stride**，**禁止了宽度方向的进一步缩放**。

请看 `__init__` 函数中的这段关键代码：

```python
# Layer 2: 原 stride=(2,2) -> 改为 (2, 1)
# 这里的 (2, 1) 代表 (高度Stride, 宽度Stride)
self.layer2[0].conv1.stride = (2, 1)
self.layer2[0].downsample[0].stride = (2, 1)

# Layer 3: 原 stride=(2,2) -> 改为 (2, 1)
self.layer3[0].conv1.stride = (2, 1)
self.layer3[0].downsample[0].stride = (2, 1)

# Layer 4: 原 stride=(2,2) -> 改为 (2, 1)
self.layer4[0].conv1.stride = (2, 1)
self.layer4[0].downsample[0].stride = (2, 1)

```

**关键点解析：`stride = (2, 1)**`

* 第一个数字 `2` 是 **高度 Stride**：高度继续减半（为了把高度从 32 压到 1）。
* 第二个数字 `1` 是 **宽度 Stride**：**宽度保持不变**（Stride=1 不进行下采样）。

---

### 3. 全程算账表

让我们把一个宽度  的图片走一遍流程：

| 层级名称 | 标准 ResNet Stride | **你的代码 Stride** | 你的宽度计算 | 你的高度计算 |
| --- | --- | --- | --- | --- |
| **Conv1** | 2 | **2** (默认) | $1024 \div 2 = 512$ | $32 \div 2 = 16$ |
| **MaxPool** | 2 | **2** (默认) | $512 \div 2 = 256$ | $16 \div 2 = 8$ |
| **Layer 1** | 1 | **1** (默认) | $256 \div 1 = 256$ | $8 \div 1 = 8$ |
| **Layer 2** | 2 | **(2, 1)** (修改) | $256 \div 1 = 256$ | $8 \div 2 = 4$ |
| **Layer 3** | 2 | **(2, 1)** (修改) | $256 \div 1 = 256$ | $4 \div 2 = 2$ |
| **Layer 4** | 2 | **(2, 1)** (修改) | $256 \div 1 = 256$ | $2 \div 2 = 1$ |
| **最终结果** |  |  | **256 (即 W/4)** | **1** |

**总结**：
代码里的 **`conv1`** 和 **`maxpool`** 贡献了 4 倍的宽度压缩，而后面的代码通过设置 **`stride=(2, 1)`** 就像踩了刹车一样，保住了宽度不再继续缩小，从而保留了 4 倍下采样的结果。