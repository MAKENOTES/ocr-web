# UI 小图文字检测（历史实现）

本包用 ResNet18 EAST 定位 UI 小图中的水平文字框。统一 `detection` 是当前主链路；
本包保留独立的数据、模型和训练入口，供小图专用模型继续迭代。本包与统一 `detection`、
`detection_larger` 分别维护代码、配置、数据和权重，不能互相导入。

训练与评估入口都在包内；矩形多后端推理入口为根目录
[detection_small_inference_640.py](../detection_small_inference_640.py)。
文件名中的 `640` 不代表当前尺寸；默认输入为高 256、宽 768 像素。

接口细节见[架构说明](../docs/detection_small/architecture.md)，推理操作见[推理说明](inference.md)。

## 工程目录

```text
detection_small/
├── README.md                 # 包用途、训练与报告入口
├── train.py                  # 训练入口：增强课程、F1 指标和检查点管理
├── report_training.py        # 训练运行静态 HTML 报告入口
├── inference.py              # 历史矩形输入推理实现
├── inference.md              # 历史推理说明
├── components/
│   ├── image.py              # RGB 读取、透明通道合成和边缘背景色
│   ├── preprocessing.py      # 尺寸校验、固定/动态缩放补边、归一化和坐标还原
│   ├── augmentation.py       # 29 种按框安全增强与轮次课程
│   ├── dataset.py            # 严格配对标签和 bbox4/quad8 解析
│   ├── transforms.py         # 增强接线、真值生成和张量化
│   ├── ground_truth.py       # 置信度、几何真值和 ### 掩码
│   ├── loss.py               # Dice 与正样本几何损失
│   ├── postprocess.py        # 连通域解码、NMS、扩框和裁片命名
│   └── eval_metrics.py       # 框级匹配和 Precision/Recall/F1
├── models/
│   └── east_model.py         # ResNet18 EAST
├── utils/
│   ├── checkpoint.py         # 纯 state_dict 读写
│   ├── config.py             # 检查点/ONNX 元数据契约
│   └── training_eval.py      # 报告数据收集与 HTML 渲染
├── runtime/                  # ONNX/OpenVINO 自包含部署包
│   ├── backends.py           # ONNX Runtime、OpenVINO 和元数据校验
│   ├── engine.py             # 单图/批量推理、标签和裁片输出
│   ├── cli.py                # 命令行入口
│   └── image.py 等           # 由同步脚本从 components/utils 生成
├── data/                     # 本地图片和标签，不提交
├── checkpoints/              # 本地权重，不提交；新训练按运行目录保存
├── runs/                     # TensorBoard 日志，不提交
└── reports/                  # 静态分析报告，不提交
```

## 数据约定

- 图片与标签按 stem 配对；`gt_<stem>.txt` 和 `<stem>.txt` 必须恰好存在一份，
  缺失或同时存在都会立即报错，并给出具体路径。
- 标签支持水平 `bbox4` 和水平 `quad8`；旋转四边形直接拒绝。坐标超出图片范围、
  退化框或非有限值会带文件路径和行号报错。
- 行尾 `###` 表示忽略区域：四边形保留，但不进入置信度/几何真值，对应网格被训练掩码排除。
- 空标签文件表示已确认的无文字图片。
- 带透明通道的 PNG 通过 `components/image.py` 按可见前景亮度合成黑/白背景。

## 训练

训练和验证目录必须独立，默认使用 `data/train_*` 和 `data/val_*`。正式训练前核对
`data/split_manifest.csv` 与目录实际内容；不要通过复制验证图片扩充训练集。

```bash
.venv/bin/python detection_small/train.py
```

配置集中在文件顶部：

- 输入 `(256, 768)`、batch 8、最多 400 轮；Adam + OneCycleLR，峰值学习率 `3e-4`。
- 数据增强 29 种，按轮次从轻到强切换；关闭 `USE_AUGMENTATION_CURRICULUM` 可固定标准增强。
- 第 80 轮起逐轮验证损失，此前每 10 轮验证；早停按连续 30 次验证未改善计数。
- 每 `METRIC_VAL_INTERVAL=10` 轮在内存模型上计算验证集 P/R/F1，默认使用训练输入尺寸、
  置信度阈值 0.7、NMS IoU 0.3 和匹配 IoU 0.5。
- 权重按运行目录保存到 `checkpoints/ocr_experiment_<时间戳>/`：
  常规检查点保留最近 5 份，验证损失 Top-3 和验证 F1 Top-3 分别保留。
  同目录写入 `run_config.json`，记录输入尺寸、数据目录和指标口径。
- TensorBoard 日志写入 `runs/ocr_experiment_<时间戳>/`，只记录 epoch 级 loss/F1/
  学习率/增强阶段/耗时和低频梯度范数。

## 训练报告

```bash
.venv/bin/python detection_small/report_training.py \
    --run detection_small/checkpoints/ocr_experiment_<时间戳>
```

默认输出 `detection_small/reports/<运行名>/index.html` 和 `summary.json`。报告包含：
训练/验证 loss 与 F1 曲线、主配置 P/R/F1、输入尺寸 × 阈值工作点、逐图错误统计和
最差图片、检查点 F1 对比、数据构成，以及推理入口和 ONNX 导出的尺寸/权重检查。
报告只读取检查点和验证目录，不修改部署默认值。

## ONNX 导出与部署

导出配置集中在根目录 [detection_small_export_onnx.py](../detection_small_export_onnx.py)：
默认固定 `(256, 768)`，`DYNAMIC_SPATIAL=True` 时导出动态高宽；目标先写临时文件，
PyTorch 与 ONNX Runtime 数值校验通过后才覆盖，失败不会留下可误用的模型。

```bash
# 先在文件顶部把 CHECKPOINT_PATH 指向本次训练的最佳检查点
.venv/bin/python detection_small_export_onnx.py
```

ONNX 元数据写入 `ocrsmall_config`，记录输入尺寸上限、步长、几何编码、预处理版本和
`dynamic_spatial`。部署入口 [detection_small_inference_onnx.py](../detection_small_inference_onnx.py)
只保留常量，推理实现在 `detection_small/runtime/`：

```bash
.venv/bin/python detection_small_inference_onnx.py \
    --model_path best_detection_small.onnx \
    --image_path detection_small/data/test/4 \
    --output_dir detection_small/data/test_results/onnx/4 \
    --backend onnx
```

输入尺寸从模型元数据读取：固定导出始终使用导出尺寸，动态导出才按图片宽高比选择尺寸。
runtime 输出 `images/` 原尺寸画框预览、`labels/gt_<stem>.txt`（bbox4）和可选裁片，
默认执行 NMS IoU 0.3。修改 `components/image.py`、`components/preprocessing.py`、
`components/postprocess.py` 或 `utils/config.py` 后必须重新同步：

```bash
.venv/bin/python tools/sync_detection_small_runtime.py
```

构建可独立部署的目录（包含 runtime、入口、模型、依赖清单和 manifest）：

```bash
.venv/bin/python tools/build_detection_small_bundle.py \
    --model best_detection_small.onnx \
    --out dist/detection_small_runtime \
    --force
```

部署包默认在隔离路径下实际推理一次；`dist/`、导出的 ONNX 和模型权重都不提交。

## 检查

```bash
.venv/bin/python -m pytest tests/test_detection_small_augmentation.py \
    tests/test_detection_small_report.py tests/test_detection_small_onnx.py -q
.venv/bin/python -m pytest tests/test_detection_rectangular_pipeline.py \
    tests/test_detection_isolation.py -q
```

## 历史说明

`inference.py` 和根目录 `detection_small_inference_640.py` 保留历史矩形/多后端实验行为，
不作为当前 ONNX 部署的默认入口；`detection_small_quantize_int8.py` 仍可用于 INT8 量化。
训练新模型后先更新导出脚本的 `CHECKPOINT_PATH`，校验通过后再同步推理入口和部署包。
