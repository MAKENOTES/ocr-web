# PDF 页面与网页表格文字检测

本包检测完整页面图片中的水平文字框，不直接读取 PDF，不恢复表格行列结构。
模型、数据、增强、训练、推理、评估和部署均独立于统一 `detection` 和小图检测包；
不再依赖 `recognition` 的图片预处理。

统一 `detection` 是当前主链路；本包保留为 PDF/表格场景的独立实现。接口细节见
[大图检测架构](../docs/detection_larger/larger-architecture.md)。

## 目录

```text
detection_larger/
├── README.md                 # 用途、训练、报告和部署入口
├── train.py                  # 整图训练、验证损失和框级 F1 监控
├── inference.py              # PyTorch/ONNX/OpenVINO 推理与共享后处理
├── report_training.py        # 训练分析静态 HTML 报告入口
├── components/               # 只放训练链路需要的模块
│   ├── image.py              # EXIF 方向与透明通道合成
│   ├── preprocessing.py      # 尺寸校验、缩放补边、归一化和坐标还原
│   ├── dataset.py            # 严格标签配对和 bbox4/水平 quad8 解析
│   ├── augmentation.py       # 30 种按框安全增强与批次课程
│   ├── transforms.py         # 训练增强接线、真值和张量
│   ├── ground_truth.py       # 置信度、log1p 几何和忽略掩码
│   ├── loss.py               # Dice + Smooth L1
│   ├── eval_metrics.py       # 训练期与目录级 P/R/F1
│   └── training_options.py   # 初始权重和本次输出目录参数
├── models/east_model.py      # 独立 ResNet18 EAST 网络
├── runtime/                  # 部署运行包；生成文件勿手改
│   ├── backends.py           # ONNX Runtime/OpenVINO 后端
│   ├── cli.py                # 推理命令行
│   ├── engine.py             # 单图预测和批量输出
│   ├── config.py             # 配置契约（生成）
│   ├── image.py              # 图片读取（生成）
│   ├── postprocess.py        # 解码、NMS 和裁片命名（生成）
│   └── preprocessing.py      # 预处理和坐标还原（生成）
├── utils/                    # 检查点、后处理和报告
│   ├── config.py             # 检查点/ONNX 配置契约与校验
│   ├── checkpoint.py         # 检查点读写
│   ├── postprocess.py        # 连通域解码、NMS、扩框和裁片命名
│   └── training_eval.py      # 报告数据收集、曲线与 HTML 渲染
├── data/                     # 训练/验证图片与标签，不提交
├── checkpoints/              # 按运行保存的本地权重，不提交
├── runs/                     # TensorBoard 运行日志，不提交
└── reports/                  # 训练报告，不提交
```

根目录 `detection_larger_export_onnx.py` 负责导出，`detection_larger_inference_onnx.py`
只保留部署常量；runtime 快照由 `tools/sync_detection_larger_runtime.py` 生成，
独立部署包由 `tools/build_detection_larger_bundle.py` 构建。

## 训练

先准备 `detection_larger/data/train_images`、`train_labels`、`val_images`、`val_labels`。
支持 JPG/JPEG/PNG/TIFF/BMP，扩展名不区分大小写。训练标签默认
`x_min y_min x_max y_max [文本]`，支持浮点数；`###` 是忽略区域，空文件是无文字样本。
图片与标签按 stem 配对，同一张图必须恰好存在一份标签。

```bash
.venv/bin/python detection_larger/train.py
# 初始化权重，输出到新的空目录；不是完整续训
.venv/bin/python detection_larger/train.py --weights /path/to/model.pth --output detection_larger/checkpoints/new_run
```

| 顶部配置 | 当前值及含义 |
| --- | --- |
| INPUT_SIZE / BATCH_SIZE | 640×640 像素 / 每批 4 张整图 |
| EPOCHS | 最多 620 轮 |
| 学习率 | Adam 构造值 1e-4，OneCycleLR 峰值 3e-4 |
| 验证损失 | 前 50 轮每 10 轮验证，第 50 轮起逐轮验证 |
| 框级 F1 | 每 `METRIC_VAL_INTERVAL`（默认 10）轮评估，默认 640、阈值 0.7、NMS 0.3 |
| 早停 | 连续 100 次验证未改善超过 1e-5 |
| 保存 | 每 5 轮及末轮常规保存；另保留最低验证损失 Top-3 和最高 F1 Top-3 |
| PRETRAINED | True，无初始权重时使用 ImageNet 参数 |
| NUM_WORKERS / DEVICE | 4 个加载子进程 / 优先 CUDA，否则 CPU |

每次运行在 `detection_larger/checkpoints/<run>/` 下新建检查点目录，并写入
`run_config.json`；TensorBoard 事件写入 `detection_larger/runs/<run>/`。
检查点按验证损失和验证 F1 分别保留：

- `best_detection_epoch_<轮数>_loss_<损失>.pth`；
- `best_f1_detection_epoch_<轮数>_f1_<F1>.pth`。

F1 使用当前内存模型在验证集上评估，不重新加载检查点；F1 与 loss 可能不同步，
部署选型优先比较 F1。

## 数据增强

增强代码集中在 [components/augmentation.py](components/augmentation.py)，共 30 种；
验证和推理不启用。

| 类别 | 操作 |
| --- | --- |
| 几何（4） | 随机裁剪、随机补边、随机宽高补边、轻微各向异性缩放 |
| 光度（23） | 亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、JPEG 压缩、低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色 |
| 遮挡（3） | 避文字框的随机擦除、随机表格线、随机斑点 |

执行方案：

- 几何增强在原图执行，缩放补边到 640 画布后再执行光度/遮挡增强，避免在数千像素
  的 PDF 页面上重复做模糊、JPEG 和 CLAHE。
- 每个样本从当前阶段权重采样 `light`、`standard` 或 `strong`；三个阶段分别采样
  2-4、3-6、5-8 种操作，概率和幅度随阶段放大。
- 几何操作后统一裁剪四边形并删除退化框，标签不会越界；旋转、透视和剪切会破坏
  水平框契约，不在默认流程中。
- 轮次课程前 15% 以轻增强为主，15%-70% 以标准增强为主，最后 30% 提高强增强比例；
  `train.py` 只在阶段边界重建 DataLoader。设置 `USE_AUGMENTATION_CURRICULUM=False`
  可固定标准增强。

## TensorBoard 记录

保留 `Loss/Train`、`Loss/Validation`、`ScoreLoss/*`、`GeoLoss/*`、`Learning Rate`、
`Augmentation/*`、`Val/Precision`、`Val/Recall`、`Val/F1`、`Val/best_F1`，以及
`Performance/train_epoch_seconds`、`train_images_per_second`、`validation_seconds`。
batch loss 每 `BATCH_LOG_INTERVAL`（默认 50）个 batch 一次，只记录总梯度范数
（默认每 500 个 batch），验证可视化每 `VIS_INTERVAL`（默认 20）轮一次；
不再写逐参数梯度直方图。

## 训练分析报告

```bash
.venv/bin/python detection_larger/report_training.py --run detection_larger/checkpoints/<run>
```

默认输出 `detection_larger/reports/<run>/index.html` 和 `summary.json`。报告自包含
SVG 曲线和错误图片，可直接用浏览器离线打开，内容包括：

- 验证集 F1、训练/验证 loss、学习率和增强阶段曲线；
- 主配置的总体 TP/FP/FN/P/R/F1；
- 训练尺寸与 +128 更大输入、不同置信度阈值的工作点对比；
- 最佳 F1、最佳 loss 和最后一轮检查点的主配置指标；
- 错误集中度、横向/纵向页面拆分，以及最差图片的漏检/误检可视化；
- 推理入口与 ONNX 导出的检查点和尺寸是否与最佳检查点一致。

报告会在验证集上重新评估主检查点，建议在 GPU 环境运行。验证集同时用于检查点
选择和工作点对比，最佳工作点存在乐观偏差；上线前应使用独立标注集复核。

## 推理与导出

`detection_larger/inference.py` 支持 PyTorch `.pth` 和 ONNX `.onnx`；
`.onnx` 的 `auto` 后端使用 ONNX Runtime，显式 `openvino` 使用 OpenVINO CPU。
固定高宽导出的模型自动使用导出尺寸，`dynamic_spatial=True` 的模型使用元数据
记录的尺寸上限。PyTorch 默认固定 640×640 方形画布，`DYNAMIC_INPUT=True` 时按
原图宽高比选择 32 的倍数尺寸。

```bash
# 修改共享图片/预处理/后处理/配置后先同步部署快照
.venv/bin/python tools/sync_detection_larger_runtime.py
.venv/bin/python tools/sync_detection_larger_runtime.py --check

# 构建独立部署目录；默认在隔离路径下完成一次推理自检
.venv/bin/python tools/build_detection_larger_bundle.py \
  --model best_detection_larger.onnx --out dist/detection_larger_runtime

.venv/bin/python detection_larger/inference.py
.venv/bin/python detection_larger_inference_onnx.py --image_path detection_larger/data/test_images
.venv/bin/python detection_larger_export_onnx.py
```

包内推理与 ONNX 入口默认输出 `images/<原图文件名>` 画框预览和
`labels/gt_<stem>.txt` bbox4 标签；`SAVE_CROPS=True` 时按
`<原图 stem>-<5 位序号>.png` 保存裁片，并清理该原图多余的旧序号。

导出默认 `INPUT_SIZE=640`、固定高宽、动态 batch 1-32；`DYNAMIC_SPATIAL=True`
时额外导出动态高宽。导出直接覆盖同名目标文件，随后比较 PyTorch 与 ONNX Runtime
的 batch 1/2 输出（动态导出还覆盖不同高宽）；`BENCHMARK_ENABLED=True` 时额外执行
CPU 测速。修改检查点或尺寸后必须重新导出，已有 ONNX 不会自动更新。

## 评估与测试

```bash
.venv/bin/python detection_larger/components/eval_metrics.py \
  --model /path/to/document-model.pth \
  --image-dir detection_larger/data/val_images \
  --label-dir detection_larger/data/val_labels

.venv/bin/python -m pytest tests/test_detection_larger.py tests/test_detection_larger_augmentation.py \
  tests/test_detection_larger_report.py tests/test_detection_larger_onnx.py \
  tests/test_detection_isolation.py -q
```

评估按置信度一对一匹配，默认 IoU 0.5，`###` 区域内的预测不计误检，输出总体和
按扩展名分组的 TP/FP/FN、Precision、Recall、F1。单元测试不代表当前本地权重精度；
验证集参与过调参时仍需独立留出集复核。
