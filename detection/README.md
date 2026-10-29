# 统一文字检测

本包用 ResNet18 EAST 同时定位 UI 小图、PDF 页面和网页表格中的水平文字框。
训练、推理、评估和导出都只依赖包内实现，不导入两个历史检测目录或识别子系统。

接口细节见[统一检测架构](../docs/detection/architecture.md)；历史实现见
[小图检测](../detection_small/README.md)和[大图检测](../detection_larger/README.md)。

## 目录

```text
detection/
├── README.md                 # 用途、数据迁移、常用入口和约束
├── train.py                  # 统一检测训练入口
├── inference.py              # PyTorch/ONNX/OpenVINO 推理与后处理
├── export_onnx.py            # 动态高宽 ONNX 导出
├── components/
│   ├── image.py              # EXIF 方向与透明通道合成
│   ├── dataset.py            # 严格标签配对和 bbox4/quad8 解析
│   ├── augmentation.py       # 30 种按框安全增强与批次课程
│   ├── transforms.py         # 增强接线、缩放补边、真值和张量
│   ├── ground_truth.py       # 置信度、log1p 几何和忽略掩码
│   ├── loss.py               # Dice + Smooth L1
│   ├── postprocess.py        # 四连通域解码
│   ├── onnx_runtime.py       # ONNX Runtime/OpenVINO 配置
│   ├── eval_metrics.py       # 目录级 Precision/Recall/F1
│   └── training_options.py   # 本次运行的检查点输出目录
├── models/east_model.py      # ResNet18 多尺度融合网络
├── utils/checkpoint.py       # 检查点配置契约与读写
└── data/                     # 本地迁移数据、清单和推理产物，不提交
```

## 数据

`tools/prepare_unified_detection_data.py` 从两个历史检测目录读取图片和标签，
按图片内容 MD5 去重后写入 `detection/data/`，并用固定种子对每个来源域做
80/20 划分。当前结果为 4974 张训练图、1243 张验证图；同一内容不会同时出现在
train 和 val。`split_manifest.csv` 记录样本归属、来源路径和内容摘要。

文件名用 `ui_` 和 `doc_` 前缀区分来源，标签统一为 `gt_<stem>.txt`。
`detection/data/` 全部由本地生成，不提交；完整训练前不要手工复制验证图片
到训练目录。`test_images/` 是从历史文档检测目录复制的推理演示图片，其中部分
内容与 train/val 重复，只能用于演示，不能作为独立测试集；独立评估使用
`val_images/`。

## 训练

```bash
.venv/bin/python detection/train.py
.venv/bin/python detection/train.py --output /path/to/new_run
```

配置集中在 [train.py](train.py) 顶部：默认 RGB、`640×640`、batch 4、最多 620 轮，
Adam + OneCycleLR；训练使用 30 种按框安全增强，验证使用确定性缩放。检查点只保存
`config` 和 `model`；训练从 `PRETRAINED` 指定的 ImageNet 初始化或随机初始化开始。
训练和验证目录路径必须不同。

训练输出在 `detection/checkpoints/<run>/`，验证损失最低的 Top-3 文件名为
`best_detection_epoch_<轮数>_loss_<损失>.pth`。

## 数据增强与批次执行

增强代码集中在 [components/augmentation.py](components/augmentation.py)，共 30 种；
验证和推理不启用。

| 类别 | 操作 |
| --- | --- |
| 几何（4） | 随机裁剪、随机补边、随机宽高补边、轻微各向异性缩放 |
| 光度（23） | 亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、JPEG 压缩、低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色 |
| 遮挡（3） | 避文字框的随机擦除、随机表格线、随机斑点 |

执行方案：

- 几何增强在原图执行，缩放补边到 640 后再执行光度/遮挡增强，避免在数千像素的
  PDF 页面上重复做模糊、JPEG 和 CLAHE。
- 每个样本先从当前阶段权重采样 `light`、`standard` 或 `strong`；三个阶段合计
  采样 2-4、3-6、5-8 种操作，概率和幅度随阶段放大。
- 操作固定按“几何 → 光度 → 遮挡”执行；每个几何操作后裁剪四边形并删除退化框，
  标签不会越界。
- 轮次课程由 `policy_weights_for_epoch` 控制：前 15% 以轻增强为主，15%-70% 以标准
  增强为主，最后 30% 提高强增强比例。`train.py` 只在阶段边界重建训练 DataLoader，
  不会每轮重启 worker；设置 `USE_AUGMENTATION_CURRICULUM=False` 可固定标准增强。
- 同一批次自然混合不同强度样本；验证集只做确定性缩放，保证指标可复现。
- 旋转、透视和剪切会破坏水平框契约，因此不在默认增强中；JPEG 压缩、CLAHE、运动
  模糊和中值模糊属于较重操作，已使用较低概率。若数据加载成为瓶颈，优先降低这些
  操作的概率或提高 `NUM_WORKERS`。

## 推理与导出

```bash
# 直接运行包内入口：修改 detection/inference.py 的 MODEL_PATH 和 IMAGE_PATH
.venv/bin/python detection/inference.py

# 根目录部署入口：修改 detection_inference_onnx.py 的 MODEL_PATH 等常量
.venv/bin/python detection_inference_onnx.py

# 导出前修改 detection_export_onnx.py 的 CHECKPOINT_PATH 和 OUTPUT_PATH
.venv/bin/python detection_export_onnx.py
```

推理默认 `DYNAMIC_INPUT=True`：按原图宽高比把最长边缩放到 `INPUT_SIZE`，每条边
向上取整到 32 的倍数，最少 64。训练仍是固定尺寸，动态只减少推理补边计算，
不改变输入像素分布。固定尺寸导出的 ONNX/OpenVINO 自动使用模型自身尺寸；
动态导出的 ONNX/OpenVINO 使用导出元数据记录的 `input_size` 作为尺寸上限。
输出 `images/<原图文件名>` 画框预览和 `labels/gt_<stem>.txt` bbox4 标签；
`SAVE_CROPS=True` 时额外保存原图裁片。

导出默认 `DYNAMIC_SPATIAL=True`，同时校验动态 batch 和不同高宽；任何数值校验
失败都不会生成目标 ONNX。目标文件已存在时拒绝覆盖。导出尺寸和动态开关在
[export_onnx.py](export_onnx.py) 顶部；根目录入口只负责本次检查点和输出路径。

## 评估与检查

```bash
.venv/bin/python detection/components/eval_metrics.py \
  --model /path/to/model.pth \
  --image-dir detection/data/val_images \
  --label-dir detection/data/val_labels

.venv/bin/python -m pytest tests/test_detection_unified.py tests/test_detection_augmentation.py tests/test_detection_isolation.py -q
```

评估按置信度一对一匹配，默认 IoU 0.5，`###` 区域内的预测不计误检。
