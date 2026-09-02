# Recognition 架构

## 1. 职责

Recognition 接收 detection 输出的单个文字框，输出 Unicode 文本。当前它是项目的实际主链路，支持彩色/深浅背景、PNG 和动态宽度处理。

当前基线为 MobileNet 风格 CNN + 单层双向 GRU + CTC head，输入高度固定为 32，宽度按比例缩放并在 batch 内动态 padding。

## 2. 数据与预处理

```mermaid
flowchart TD
    A[annotation: relative_path,label] --> DS[TextRecognitionDataset]
    I[元素图片] --> DS
    DS --> PNG[严格读取标注中的 PNG 路径]
    PNG --> RGB[读取 RGB + 透明通道合成]
    RGB --> RESIZE[完整 detection 框固定高度32 + 等比宽度]
    RESIZE --> MINW[窄图按边缘背景色补到宽48]
    MINW --> AUG{训练集?}
    AUG -->|是| TA[颜色/极性/模糊/轻噪声]
    AUG -->|否| NORM[仅 Normalize]
    TA --> BUCKET[按宽高比桶采样]
    NORM --> PAD[batch 动态 padding 到8倍数]
    BUCKET --> PAD
    PAD --> OUT[images, labels/paths, img_widths]
```

相关实现：

- `recognition/components/dataset.py`：图片路径校验、PNG 透明通道处理、缩放、增强、桶采样和动态 padding；
- `recognition/utils/checkpoint.py`：checkpoint 解包以及旧灰度/新 RGB 模型通道识别；
- `recognition/components/converter.py`：字典编码和显式 TBC/BTC 的 CTC greedy decode。

训练、验证、PyTorch 推理和多后端推理复用同一套不裁剪基础预处理。训练可额外使用随机增强；验证和测试禁止随机增强。

## 3. 模型与张量契约

```mermaid
flowchart LR
    X[图像 B,C,32,W] --> CNN[MobileFeatureExtractor]
    CNN --> F[特征 B,256,1,W/4]
    F --> T[转为 T,B,256]
    W[img_widths B] --> MASK[真实序列长度/Mask]
    MASK --> T
    T --> GRU[单层 BiGRU]
    GRU --> FC[Linear classes]
    FC --> LS[LogSoftmax]
    MASK --> LS
    LS --> Y[输出 T,B,C]
    Y --> CTC[CTC Loss / Greedy Decode]
```

稳定契约：

- 输入图像：`float32 [B, C, 32, W]`，归一化到以 0.5 为 mean/std 的空间；
- `img_widths`：`int32/int64 [B]`，表示 padding 前像素宽度；
- 输出：`float32 [T, B, num_classes]`，简称 TBC；
- 类别：0=`blank`，1=`UNK`，2 起来自 `keys.txt`；
- `keys.txt` 的内容和顺序是模型 ABI，变更字典必须训练/导出新模型版本；
- CNN 宽度下采样约为 4 倍，CTC 长度必须依据每张图片真实宽度计算。

`LabelConverter.decode` 已要求显式 `layout="TBC"` 或 `"BTC"`。部署入口会用已知 batch size 和类别数把后端输出规范为 TBC，不再用维度大小关系猜测。

当前模型在 BiGRU 前将 padding 特征置零，但严格意义上反向 GRU 仍可能受尾部长度影响。完成长度感知 RNN 和 batch=1/混合宽度 batch 的自动一致性门禁前，生产建议使用 `batch_size=1`。

## 4. 训练、评测、导出与推理

```mermaid
flowchart LR
    TRAIN[recognition/train.py] --> PTH[checkpoint]
    VAL[独立 val/test manifest] --> EVAL[recognition/evaluate_val.py]
    PTH --> EVAL
    PTH --> EX[recognition_export_onnx.py]
    EX --> ONNX[FP32 ONNX 动态B/W]
    PTH --> INF[recognition_inference.py]
    ONNX --> INF
    INF --> TXT[文本/耗时/错误]
```

主入口：

- `recognition/train.py`：训练与训练期验证；
- `recognition/evaluate_val.py`：独立 manifest 的 Accuracy/CER/错误样本；
- `recognition/inference.py`：PyTorch checkpoint 调试；
- `recognition_export_onnx.py`：带 `img_widths`、动态 batch/width 的 ONNX；
- `recognition_inference.py`：ONNX/OpenVINO/PyTorch 部署推理。

示例：

```bash
python recognition/train.py

python recognition/evaluate_val.py \
  --model_path recognition/checkpoints/<model>.pth \
  --annotation recognition/data/val_annotation.txt \
  --batch_size 1

python recognition_export_onnx.py \
  --model_path recognition/checkpoints/<model>.pth \
  --output_path artifacts/recognition/<version>/model.onnx \
  --keys_path recognition/keys.txt \
  --img_height 32 \
  --img_width 256

python recognition_inference.py \
  --model_path artifacts/recognition/<version>/model.onnx \
  --backend auto \
  --keys_path recognition/keys.txt \
  --image_path recognition/data/test_images1 \
  --batch_size 1
```

仓库中的 `recognition/data/val_annotation.txt` 暂无完整可用的对应图片。建立无泄漏划分前，不应将训练集评估结果作为发布精度。

## 5. 评测维度

总体 Accuracy/CER 之外，至少按以下网页场景报告：

- `menu-white-text`：菜单、按钮、有色/深色背景白字；
- `small-text-or-sparse`：小文字、大背景、文字占比低；
- `has-icon-or-arrow`：箭头、图标、角标邻接；
- `long-text`：长文本、空格和中英文标点；
- `light-dark-theme`：浅色/深色主题；
- `scale-dpr`：浏览器缩放和不同设备像素比；
- `unknown-char`：字典外字符和规范化策略。

门禁必须包含：跨 split 重复为 0、冲突 label 为 0、batch 一致率 100%、PyTorch/ONNX 文本一致率 100%、关键分组不低于已批准基线。阈值和基线报告都要版本化。

## 6. 已知问题与演进边界

- `train.py` 仍以脚本常量为主，需迁移为可序列化配置与 run manifest。
- 训练数据与标注统一使用 `.png`；路径或扩展名不一致会直接报错，避免自动回退掩盖数据问题或以 dummy tensor 静默污染训练。
- recognition 不再做二次裁剪；detection 框质量需要在 detection 环节单独评估。
- 输入高度目前固定 32；48 高度只能作为同数据、同训练预算的消融实验，不能直接替换现有权重。
- `recognition_quantize_static.py` 当前不进入发布路径。FP32 ONNX/OpenVINO 是默认优化方向。
- `model_resnet50.py` 和历史说明文档属于实验资产，未通过同一门禁前不替代当前基线。
