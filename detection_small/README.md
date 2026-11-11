# UI 小图文字检测（历史实现）

本文介绍 UI 小图文字检测包的用途、工程目录、常用入口和关键约束，并提供相关文档索引。

统一 `detection` 是当前主链路；本包保留原有实现、数据和历史脚本，不再作为默认训练和推理入口。

本包用 ResNet18 EAST 定位 UI 小图中的水平文字框。训练入口为 [train_resnet.py](train.py)，
矩形多后端推理入口为根目录的 [detection_small_inference_640.py](../detection_small_inference_640.py)。
文件名中的 `640` 不代表当前尺寸；默认输入为高 256、宽 768 像素。

本包与统一 `detection`、`detection_larger` 分别维护代码、配置、数据和权重，不能互相导入。
接口细节见[架构](../docs/detection_small/architecture.md)，推理操作见[推理说明](inference.md)。

## 工程目录层级

下图列出主要目录和文件；省略包初始化文件与缓存。

```text
detection_small/
├── README.md                 # 包用途、训练约束与检查入口
├── train_resnet.py           # 历史小图 ResNet18 训练入口
├── inference.md              # 历史根目录推理入口的使用说明
├── inference_v3resnet18_640.py  # 历史矩形输入推理实现
├── components/
│   ├── dataset.py            # 图片与文字框标签加载
│   ├── transforms.py         # 颜色增强、缩放补边与张量转换
│   ├── ground_truth.py       # 历史置信度和几何真值生成
│   ├── loss.py               # Dice 与几何损失
│   └── eval_metrics.py       # 独立框级指标评估
├── models/
│   └── east_model.py         # 历史 ResNet18 EAST
├── data/                     # 本地图片、标签及 split_manifest.csv，不提交
└── checkpoints/              # 本地模型权重，不提交
```

当前多后端推理和导出位于根目录：`detection_small_inference_640.py`、`detection_small_export_onnx.py`。
架构与评估边界放在 `docs/detection_small/architecture.md`，本包不依赖统一检测和大图检测包。

## 训练前检查

`train.py` 当前的 `VAL_IMAGE_DIR`、`VAL_LABEL_DIR` 指向训练目录。
直接运行得到的是训练数据上的损失，不能作为独立验证精度。正式训练前应将这两个常量
指向已准备好的独立验证目录，并核对 `data/split_manifest.csv`；本次文档核对没有改动该配置。

训练参数集中在文件顶部：RGB PNG、输入 `(256, 768)`、batch 8、最多 200 轮。
使用 Adam 和 OneCycleLR，实际峰值学习率为 `3e-4`；`LEARNING_RATE=1e-4` 是优化器构造值。
第 80 轮起逐轮验证，此前每 10 轮验证；早停按连续 15 次验证未改善计数。
常规权重每 5 轮保存，保留最近 5 份；另保留验证损失最低的 3 份。
保存的是模型参数，不包含优化器、调度器或完整续训状态。

```bash
# 先核对顶部路径、输入尺寸和验证集配置
.venv/bin/python detection_small/train_resnet.py
```

## 历史实现与检查

`inference.py` 保留历史矩形输入推理配置，不作为当前矩形入口的默认参数来源。
模型和真值文件也有历史变体；实际导入关系见架构文档。

```bash
.venv/bin/python -m pytest tests/test_detection_rectangular_pipeline.py tests/test_detection_isolation.py -q
```

尚待处理的加载、评估和数据隔离问题见[核对待办](../docs/code-doc-todolist.md)。
