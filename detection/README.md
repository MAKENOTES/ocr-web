# 统一文字检测

本包用 ResNet18 EAST 同时定位 UI 小图、PDF 页面和网页表格中的水平文字框。
训练、推理和评估只依赖包内实现，不导入两个历史检测目录或识别子系统；ONNX
导出在根目录 [detection_export_onnx.py](../detection_export_onnx.py) 完成，复用包内
模型加载和运行时配置。

接口细节见[统一检测架构](../docs/detection/architecture.md)；历史实现见
[小图检测](../detection_small/README.md)和[大图检测](../detection_larger/README.md)。

## 目录

```text
detection/
├── README.md                 # 用途、数据迁移、常用入口和约束
├── train.py                  # 统一检测训练入口
├── inference.py              # PyTorch/GPU 推理与共享后处理
├── report.py                 # 训练分析静态 HTML 报告入口
├── inference_report.py       # 部署入口推理评估报告入口
├── components/
│   ├── image.py              # EXIF 方向与透明通道合成
│   ├── dataset.py            # 严格标签配对和 bbox4/quad8 解析
│   ├── augmentation.py       # 30 种按框安全增强与批次课程
│   ├── transforms.py         # 增强接线、缩放补边、真值和张量
│   ├── ground_truth.py       # 置信度、log1p 几何和忽略掩码
│   ├── loss.py               # Dice + Smooth L1
│   ├── postprocess.py        # 四连通域解码与词/短语拆分
│   ├── onnx_runtime.py       # ONNX Runtime/OpenVINO 配置
│   ├── eval_metrics.py       # 目录级 Precision/Recall/F1
│   ├── training_report.py    # 报告数据收集、曲线与 HTML 渲染
│   ├── inference_report.py   # 部署推理指标、后端一致性与报告渲染
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

配置集中在 [train.py](train.py) 顶部：默认 RGB、`768×768`、batch 8、`NUM_WORKERS=8`、
最多 620 轮；Adam 构造学习率 1.5e-4，OneCycleLR 峰值 4.5e-4；训练使用 30 种
按框安全增强，验证使用确定性缩放。检查点只保存 `config` 和 `model`；训练从
`PRETRAINED` 指定的 ImageNet 初始化或随机初始化开始。训练和验证目录路径必须不同。

训练输出在 `detection/checkpoints/<run>/`。同时保留两套 Top-3：

- 验证损失最低：`best_detection_epoch_<轮数>_loss_<损失>.pth`；
- 验证 F1 最高：`best_f1_detection_epoch_<轮数>_f1_<F1>.pth`。

F1 每 `METRIC_VAL_INTERVAL`（默认 10）轮评估一次，默认按 `METRIC_INPUT_SIZE=768`
（当前部署最佳尺寸）推理，并按 `ui_`/`doc_` 前缀分组记录，避免只因 loss 轮换而删除
实际 F1 更好的检查点。

## TensorBoard 记录

保留的曲线：

- `epoch/loss`、`epoch/score Loss`、`epoch/geo Loss`：训练/验证损失；
- `Learning Rate`、`Augmentation/*`：学习率和增强阶段权重；
- `Val/Precision`、`Val/Recall`、`Val/F1`，以及 `_ui`、`_doc` 分组；
- `Val/best_F1`：当前最佳 F1；
- `Performance/train_epoch_seconds`、`train_images_per_second`、`validation_seconds`。

降频/关闭的高开销记录：

- `batch/loss` 每 `BATCH_LOG_INTERVAL`（默认 50）个 batch 一次；
- `Gradient/total_norm` 每 `GRADIENT_LOG_INTERVAL`（默认 500）个 batch 一次；
- `Vis/*` 每 `VIS_INTERVAL`（默认 20）轮一次，只写输入图和预测置信度图；
- 不再写逐参数 `Gradients/*` 直方图、`Gradient_Norm/*`、逐 batch score/geo loss，
  以及重复的 `epoch/loss_train` 组。旧 run 的事件文件约 5.4GB，主要来自逐参数梯度
  直方图和每轮可视化；新配置会显著减小体积。

## 训练分析报告

```bash
.venv/bin/python detection/report.py --run detection/checkpoints/<run>
```

默认输出 `detection/reports/<run>/index.html` 和 `summary.json`，报告自包含
内嵌 SVG 曲线和错误图片，可以直接用浏览器离线打开。内容包括：

- 验证集 F1、训练/验证 loss、学习率和增强阶段曲线；
- 主配置的总体/doc/ui TP/FP/FN/P/R/F1；
- 训练尺寸与更大输入、不同置信度阈值的工作点对比；
- 最佳 F1、最佳 loss 和最后一轮检查点的主配置指标；
- 错误集中度、横向/纵向文档页拆分，以及最差图片的漏检/误检可视化；
- 历史标注复核是否过期、推理和 ONNX 导出默认值是否与最佳检查点一致。

报告会在验证集上重新评估最佳 F1、最佳 loss 和最后一轮检查点，建议在 GPU
环境运行。验证集同时用于检查点选择和工作点对比，报告中的最佳工作点存在乐观
偏差；上线前应使用未参与训练和调参的独立标注集复核。`detection/reports/` 属于
本地产物，不提交到仓库。

## 部署推理评估报告

```bash
.venv/bin/python detection/inference_report.py
.venv/bin/python detection/inference_report.py --skip-inference
.venv/bin/python detection/inference_report.py --skip-inference --error-previews
```

默认用 [inference.py](inference.py)（PyTorch/GPU）和根目录
[detection_inference_onnx.py](../detection_inference_onnx.py)（ONNX/OpenVINO CPU）
分别在 `detection/data/train_images` 和 `detection/data/val_images` 上推理，按训练期
相同口径（匹配 IoU 0.5、`###` 忽略区）重新计算总体和 doc/ui 的 P/R/F1，并与
`detection/reports/<run>/summary.json` 的训练期验证指标、TensorBoard 曲线和入口耗时
对照。输出：

- `detection/data/inference_eval/<backend>_<split>/`：画框图、预测 bbox4 标签和耗时汇总；
- `detection/reports/<run>/inference/index.html` 和 `summary.json`：可离线打开的报告；
- `detection/reports/<run>/inference/errors_train.csv`、`errors_val.csv`：逐图错误明细，
  只保留至少一个后端有 FP/FN 的图片，并同时列出 PyTorch 与 ONNX 的 TP/FP/FN。

`--error-previews` 在上述错误明细生成后，用 `detection/inference.py` 的 PyTorch
入口对这些错误图片重新推理，并按该入口相同的红色预测框、原尺寸和原格式保存预览图到
错误 CSV 同层的 `error_previews/<split>/`；逐图命中数和耗时记录到
`error_previews_<split>.csv`。预览 CSV 和图片齐全时自动跳过；`--force-previews`
只重算预览，`--force` 在重跑四次推理时一并重算预览。

预测标签齐全时自动跳过对应推理，`--skip-inference` 只重建报告，`--force` 强制重跑。
ONNX 入口使用其默认 OpenVINO CPU 后端；ONNX Runtime 在当前机器上单图前向明显更慢。
训练集指标是 in-sample 结果，只表示拟合程度；验证集参与过检查点和阈值选择，离线 F1
仍存在乐观偏差，上线前需独立留出集复核。

## 数据增强与批次执行

增强代码集中在 [components/augmentation.py](components/augmentation.py)，共 30 种；
验证和推理不启用。

| 类别 | 操作 |
| --- | --- |
| 几何（4） | 随机裁剪、随机补边、随机宽高补边、轻微各向异性缩放 |
| 光度（23） | 亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、JPEG 压缩、低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色 |
| 遮挡（3） | 避文字框的随机擦除、随机表格线、随机斑点 |

执行方案：

- 几何增强在原图执行，缩放补边到训练输入尺寸（默认 768）后再执行光度/遮挡增强，
  避免在数千像素的 PDF 页面上重复做模糊、JPEG 和 CLAHE。
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

`detection/inference.py` 面向 PyTorch/GPU 推理；根目录 `detection_inference_onnx.py`
是单文件 ONNX CPU 部署入口，预处理、解码、NMS 和输出全部内联，不导入
`detection` 包，部署时只需复制该文件、`.onnx` 模型和依赖（numpy、Pillow、
OpenCV、onnxruntime；OpenVINO 后端另需 openvino 和 onnx）。

```bash
# 直接运行包内入口（PyTorch/GPU）：修改 detection/inference.py 的 MODEL_PATH 和 IMAGE_PATH
.venv/bin/python detection/inference.py

# 独立 ONNX CPU 入口：常量提供默认值，命令行覆盖本次输入
.venv/bin/python detection_inference_onnx.py --image_path detection/data/test_images
.venv/bin/python detection_inference_onnx.py --backend openvino --save_crops

# 导出前确认 detection_export_onnx.py 的 CHECKPOINT_PATH、OUTPUT_PATH 和 INPUT_SIZE
.venv/bin/python detection_export_onnx.py
```

推理默认 `DYNAMIC_INPUT=True`：按原图宽高比把最长边缩放到 `INPUT_SIZE`（当前 768），
每条边向上取整到 32 的倍数，最少 64。训练仍是固定尺寸，动态只减少推理补边计算，
不改变输入像素分布。固定尺寸导出的 ONNX/OpenVINO 自动使用模型自身尺寸；
动态导出的 ONNX/OpenVINO 使用导出元数据记录的 `input_size` 作为尺寸上限。
输出 `images/<原图文件名>` 画框预览和 `labels/gt_<stem>.txt` bbox4 标签；
`SAVE_CROPS=True` 时额外保存原图裁片。

后处理默认按低占用列把粘连的整行连通域拆成词/短语框，可通过
`decode_maps(..., split_word_gaps=False)` 保留整段输出。当前部署配置为
Epoch 560 / 768 / 0.70；历史评估中 566/768/0.88 在验证集上拆分前后 F1 相同，
说明模型已基本输出词/短语级框；拆分器主要作为低分辨率或后续模型的兼容保护。

导出默认 `DYNAMIC_SPATIAL=True`，同时校验动态 batch 和不同高宽。默认输出
`best_detection.onnx`：导出会直接覆盖同名目标文件并随后完成数值校验，不再创建
`onnx-export-*` 临时目录；校验失败时目标文件保留本次产物，需修复后重新导出。
导出实现、尺寸和动态开关都在根目录
[detection_export_onnx.py](../detection_export_onnx.py) 顶部，按本次检查点和输出路径运行。
导出时只定向抑制 torch 2.9 对 torchvision schema 和动态 shape 的内部 WARNING，
ERROR 及以上仍会输出，PyTorch/ONNX Runtime 数值校验不受影响。
当前推理默认检查点为 Epoch 560、最长边 768；修改检查点或尺寸后必须重新导出
ONNX/OpenVINO，已有模型文件不会自动使用新权重。

## 评估与检查

```bash
.venv/bin/python detection/components/eval_metrics.py \
  --model /path/to/model.pth \
  --image-dir detection/data/val_images \
  --label-dir detection/data/val_labels

.venv/bin/python -m pytest tests/test_detection_unified.py tests/test_detection_augmentation.py tests/test_detection_isolation.py -q
```

评估按置信度一对一匹配，默认 IoU 0.5，`###` 区域内的预测不计误检。
