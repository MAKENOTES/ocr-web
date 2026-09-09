# Detection 是否需要 blank：评估报告

评估日期：2026-09-08。对象是当前矩形 EAST 主链路，对照
[`recognition/components/converter.py`](../../recognition/components/converter.py)。
配套文件：可运行的[评估脚本](evaluate_blank.py)、本次运行的[完整 JSON 结果](blank-evaluation.json)。

## 1. 结论

**当前 detection 不需要增加 CTC `[blank]` 类别；现有背景监督需要保留。
Recognition 的 CTC blank 则需要保留，不能直接删除。**

| 概念 | 当前作用 | 结论 |
| --- | --- | --- |
| Recognition `[blank] = 0` | 时间序列对齐、不输出字符、隔开相邻重复字 | 当前 CTC 架构需要 |
| Detection `score = 0` | 收缩文字区域之外的负标签，包括部分框内边缘及 padding | 需要，但它是背景监督，不是 CTC token |
| Detection 独立 blank 输出通道 | 当前没有该通道，也没有字符表或 CTC 解码 | 无需增加 |
| 普通空格 `" "` | Recognition 中可输出的真实字符 | 不能拿 blank 替代 |
| 没有文字框的图片 | Detection 的全背景样本 | 当前标签形式能表达；是否增补、增补比例需另做训练对照 |

本次完成了代码核查、CPU 机制实验以及 64 张真实图片的标签抽样。
未训练新模型、未评估已有权重的准确率，也没有改动生产模型、损失函数、字符表或本地数据。

## 2. 两阶段为什么不同

Recognition 的 [`LabelConverter`](../../recognition/components/converter.py)
分配 `0 = blank`、`1 = UNK`、`2… = 字符`。`encode()` 只编码目标文字，
不往目标里插入 blank；`decode()` 合并相邻重复编号并移除 blank。
[`recognition/train.py`](../../recognition/train.py) 使用 `CTCLoss(blank=0, zero_infinity=True)`；
[`recognition/models/model.py`](../../recognition/models/model.py) 还把无效 padding 时间步固定到 blank。
因此删除该编号会同时影响损失、解码、padding 和模型类别 ABI。

PyTorch 对 CTC 的接口约定也是输出类别数包含 blank、目标标签不能包含 blank。
参见 [PyTorch 2.9 CTCLoss 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html)。

Detection 的实际链路是：

1. [`OCRDataset`](../../detection/components/dataset.py) 读取框坐标。
2. [`get_transforms`](../../detection/components/transforms.py) 缩放、padding 并生成 GT。
   送入训练的只有 image、score_map、geo_map，transcriptions 不参与损失。
3. [`generate_ground_truth`](../../detection/components/ground_truth.py) 将 score 初始化为 0，
   收缩后的文字区域设为 1；geometry 为四个方向的距离，经 `log1p` 编码。
4. [`EASTModel`](../../detection/models/east_model.py) 输出单通道 sigmoid score 和四通道 geometry。
5. [`EASTLoss`](../../detection/components/loss.py) 在整张 score 图上计算 Dice，
   只在正样本位置计算 geometry Smooth L1。
6. [`detection_inference_640.py`](../../detection_inference_640.py)
   将 geometry 用 `expm1` 还原，score 以 `> 0.7` 二值化，经过四连通域恢复文字框。

这条链路没有“多个时间步合并为一个字符”的过程，自然没有 CTC blank 的职责。
这里依据实际执行代码：loss 文件中部分旧注释仍写五通道、ReLU 或 IoU，
当前主链路实际是四通道、log 距离和 Smooth L1。

## 3. 实验方法与实测结果

环境：Python 3.10.16、PyTorch 2.9.0+cu130、NumPy 2.2.6、OpenCV 4.12.0。
全部实验在 CPU、单线程下运行；输入为当前默认 `256 × 768`，score 图为 `64 × 192`。
随机种子 `20260908`。JSON 保存了源文件 SHA-256、抽样路径和标签 SHA-256。

### 3.1 CTC：连续重复字需要 blank

脚本创建临时教学字典 `A` 和普通空格，调用仓库原始 converter，得到：

| 输入路径 | 解码结果 |
| --- | --- |
| `A → A → A` | `A` |
| `A → blank → A` | `AA` |
| `blank → blank → blank` | 空字符串 |
| `A → 空格 → A` | `A A` |

穷举四类、三个时间步的全部 64 条路径，能得到目标 `AA` 的路径只有 1 条，
即 `A → blank → A`；排除 blank 后是 0 条。
给这条正确路径高概率时，CTC loss 为 `1.072883e-6`；
将 blank 概率强制为 0 并重新归一化后，loss 为无穷大。

这里保留原始类别编号，用 `zero_infinity=False` 暴露不可对齐的结果。
训练入口的 `zero_infinity=True` 可以把这种无穷损失及其梯度置零，
并不能使原本不可对齐的目标变得可学。
这验证的是当前 CTC 的结构要求，不表示其他识别架构都需要 blank。

### 3.2 Detection：单通道已经可以表达文字与背景

对 `EASTModel(pretrained=False)` 做一次前向，仅检查结构：

| 输出 | 实测形状 |
| --- | --- |
| score | `[1, 1, 64, 192]` |
| geometry | `[1, 4, 64, 192]` |

当前 score 头只有 33 个参数。sigmoid 输出 `p` 表达文字概率，
背景概率可用 `1-p` 表达，不需要额外编号。
对于任意 logit `z`，`softmax([0, z])[1] = sigmoid(z)`；
脚本在 `[-12, 12]` 的 1001 个值上用 float64 验证，最大误差 `1.665335e-16`。

因此，换成“背景 / 文字”两通道 softmax 是一种二分类参数化选择，
并没有引入 CTC blank 的对齐能力；增加第三个“blank”还需要定义独立的标签语义。
此实验没有比较两种头的训练效果，不能据此声称准确率完全相同或存在实测速度收益。

### 3.3 Detection：背景监督不能删

构造一个带文字框的样本，GT 中 1536 个正像素、10752 个负像素，
将所有预测 score 设为 `0.5`，使用当前 EASTLoss 求梯度：

| 指标 | 实测值 |
| --- | ---: |
| score Dice loss | 0.800000 |
| 正像素 score 梯度均值 | -2.343750e-4 |
| 背景像素 score 梯度均值 | +2.604167e-5 |
| 背景位置 geometry 梯度最大绝对值 | 0 |
| 对照：只在正像素上计算 Dice 后，背景 score 梯度最大绝对值 | 0 |

梯度是对概率 `p` 求导：梯度下降会提高文字位置分数、降低背景位置分数。
geometry 不监督背景是合理的，因为背景没有要回归的文字框。
但如果同样把背景从 score loss 中屏蔽，它就失去了直接的背景抑制梯度。

再构造两个相隔的文字框，使用 GT geometry，调用当前推理入口的真实后处理函数：

| 人工 score 对照 | 四连通区域数 | 输出框数 |
| --- | ---: | ---: |
| 正确文字区域及背景 | 2 | 2 |
| 全部置为文字 | 1 | 1 |
| 全部置为背景 | 0 | 0 |

正确 score 恢复的两个框与 GT 坐标误差小于 `1e-4` 像素；全部置为文字后输出
`[10, 6, 758, 250]` 的大框。这说明背景还承担空间分隔的作用。
这是固定 geometry、干预 score 的机制实验，不是删除背景监督后重新训练得到的指标。

### 3.4 空白图片：当前纯 Dice 的梯度很弱

将单张图片的 12288 个 GT 像素都设为背景，预测 score 都设为 `0.5`：

| 指标 | 当前 Dice | BCE 对照 |
| --- | ---: | ---: |
| 分类 loss | 1.000000 | 0.693147 |
| 对 score 概率的梯度均值 | 2.649096e-13 | 1.627604e-4 |

同时 geometry loss 为 0。BCE 只在评估脚本中作为梯度参照，没有替换生产损失。

由当前 Dice 的公式，整批全背景时有：

```text
L = 1 - smooth / (sum(p) + smooth)
dL/dp_i = smooth / (sum(p) + smooth)^2
smooth = 1e-5
```

该实验中分母很大，使梯度接近零；float32 输出的 loss 已舍入为 1。
这不等于数学上完全没有梯度，也不能推广成“任何包含空白图的 batch 都没有监督”：
当前 Dice 将 batch 一起展平，混合 batch 中其他图片的正像素会影响分子，
其行为与全背景 batch 不同。

如后续目标是减少空白图误检，更值得单独比较 Dice 与 Dice+BCE 等损失、
以及全背景样本的采样方式，并验证实际误检率；新增 blank 通道不能自动解决这里的梯度问题。

### 3.5 真实数据抽样：负像素占比

从 `detection/data/val_images` 的 5920 张 PNG 中，按排序后的路径、固定随机种子
不放回抽取 64 张，复用当前验证预处理和 GT 生成过程：

| 指标 | 实测值 |
| --- | ---: |
| 抽样图片数 | 64 |
| score 图总像素 | 786432 |
| 正像素 | 53450（6.7965%） |
| 背景标签像素 | 732982（93.2035%） |
| 没有正像素的 score 图 | 0 |

背景比例是**预处理后 score 标签的负像素比例**，包含 padding 和框收缩后的边缘，
不是原图视觉空白率。零正像素图数也不等于肉眼确认的空白图片数。
该统计说明即使有文字的样本也包含大量负标签，不支持删除背景监督。

本地没有 `detection/data/split_manifest.csv`，因此这里仅做目录抽样，
未认证 train/val 独立性，也未据此报告模型泛化能力。
另外，当前 `detection/train_resnet.py` 的 `VAL_IMAGE_DIR` 和 `VAL_LABEL_DIR`
均指向 train 目录；未来做模型性能对照前应单独核对这个配置与 manifest。
本次没有调整训练配置或移动数据。

## 4. 建议与边界

当前建议保持 detection 的一通道 score、四通道 geometry、背景负标签和连通域后处理。
Recognition 保留 blank 编号及其 CTC/padding 约定。

若“blank”实际指的是添加空白图片，应将它作为负样本和损失设计的问题处理。
本次只证明现有全零 GT 能表达这类样本，并测到了全背景 batch 的弱梯度；
不能从 64 张抽样或人工实验推出应添加多少空白图、能提升多少 F1。

若需要判断改损失或增补负样本的实际收益，后续对照应固定训练/验证划分、训练预算和随机种子，
同时报告文字框 Precision/Recall/F1 与无文字图片的误检图片率、每图误检框数。
本报告没有用人工框结果冒充这些模型指标。

## 5. 复现

在仓库根目录执行本次同样的评估：

```bash
.venv/bin/python docs/detection/evaluate_blank.py \
  --image-dir detection/data/val_images \
  --label-dir detection/data/val_labels \
  --max-samples 64 --seed 20260908 \
  --output docs/detection/blank-evaluation.json
```

不带数据目录参数可只运行机制实验；建议用 `--output /tmp/blank-evaluation.json`
保留本次结果快照。脚本写 JSON，本 Markdown 是本次结果的人工分析报告，重跑不会自动改写正文。
只有输出结构检查会实例化随机初始化的模型，不下载预训练权重，不使用 GPU。
抽样遇到缺图、缺标签、同 stem 双标签、解析警告或非有限 GT 时会报错，不静默跳过。

针对性验证：

```bash
.venv/bin/python -m pytest tests/test_detection_blank_evaluation.py \
  tests/test_detection_rectangular_pipeline.py tests/test_detection_split_dataset.py -q
```

本次结果：**9 passed in 2.60s**。评估脚本的 `py_compile` 检查通过。
