# UI 小图文字检测

本包用 ResNet18 EAST 定位 UI 小图中的水平文字框。训练入口为 [train_resnet.py](train_resnet.py)，
矩形多后端推理入口为根目录的 [detection_inference_640.py](../detection_inference_640.py)。
文件名中的 `640` 不代表当前尺寸；默认输入为高 256、宽 768 像素。

本包与 `detection_larger` 分别维护代码、配置、数据和权重，不能互相导入。
接口细节见[架构](../docs/detection/architecture.md)，推理操作见[推理说明](inference.md)。

## 训练前检查

`train_resnet.py` 当前的 `VAL_IMAGE_DIR`、`VAL_LABEL_DIR` 指向训练目录。
直接运行得到的是训练数据上的损失，不能作为独立验证精度。正式训练前应将这两个常量
指向已准备好的独立验证目录，并核对 `data/split_manifest.csv`；本次文档核对没有改动该配置。

训练参数集中在文件顶部：RGB PNG、输入 `(256, 768)`、batch 8、最多 200 轮。
使用 Adam 和 OneCycleLR，实际峰值学习率为 `3e-4`；`LEARNING_RATE=1e-4` 是优化器构造值。
第 80 轮起逐轮验证，此前每 10 轮验证；早停按连续 15 次验证未改善计数。
常规权重每 5 轮保存，保留最近 5 份；另保留验证损失最低的 3 份。
保存的是模型参数，不包含优化器、调度器或完整续训状态。

```bash
# 先核对顶部路径、输入尺寸和验证集配置
.venv/bin/python detection/train_resnet.py
```

## 固定训练集与验证集归属

[split_dataset.py](split_dataset.py) 以已有 manifest 为准，保留旧样本归属，仅为新增样本分组，
尽量使验证集占 20%。按图片 stem 配对标签，支持 `gt_<stem>.txt` 和 `<stem>.txt`；
缺失或歧义标签会报错。划分器的校验比当前训练加载器严格，二者不能混为一谈。

```bash
# 划分器默认执行移动，检查时必须显式加 --dry-run
.venv/bin/python detection/split_dataset.py --dry-run

# 需要执行划分时使用；会移动图片和标签并更新 manifest
.venv/bin/python detection/split_dataset.py
```

可用 `--source-images`、`--source-labels` 指定新增数据目录，其他路径及 `move/copy` 选项见 `--help`。
不要让训练和验证目录包含同一样本；`copy` 模式不能代替源目录与训练目录的隔离检查。

## 历史实现与检查

`inference.py`、`inference_v*.py`、`train_resnet_metrics.py`、`train_resnet_overfit.py`
和 `train_mobilenet.py` 保留各自实验配置，不作为当前矩形入口的默认参数来源。
模型和真值文件也有历史变体；实际导入关系见架构文档。

```bash
.venv/bin/python -m pytest tests/test_detection_rectangular_pipeline.py tests/test_detection_split_dataset.py tests/test_detection_isolation.py -q
```

尚待处理的加载、评估和数据隔离问题见[核对待办](../docs/code-doc-todolist.md)。
