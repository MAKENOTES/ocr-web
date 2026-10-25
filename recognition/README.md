# 文字识别

本文介绍文字识别包的用途、工程目录和常用操作，并提供架构、工具、新手导读及历史文档索引。

本包将完整 detection 文字框识别为 Unicode 文本。当前模型为 Mobile 风格 CNN、
2 层双向 GRU 和 CTC，默认输入 RGB、高度 32 像素。

## 工程目录层级

下图列出主要目录和文件，省略包初始化文件、缓存及具体数据产物。

```text
recognition/
├── README.md                 # 包入口与文档快速索引
├── train.py                  # 训练与训练期验证
├── evaluate_val.py           # 独立标注集评估与错误报告
├── evaluate_robustness.py    # 几何/成像扰动下的鲁棒性验收
├── inference.py              # PyTorch 推理
├── split_dataset.py          # 初始训练集/验证集划分
├── keys.txt                  # 字符字典，内容与顺序绑定模型
├── components/
│   ├── dataset.py            # 图片加载、预处理、增强、分桶与动态补边
│   ├── geometry.py           # 源分辨率几何增强与背景/墨迹估计
│   └── converter.py          # 字符编码与 CTC 解码
├── models/
│   ├── model.py              # 当前 Mobile CNN + 多层 BiGRU 模型
│   └── model_resnet50.py      # 历史模型，实际使用 ResNet34 + BiLSTM
├── utils/
│   ├── annotation.py         # Tab 标注解析与格式化
│   ├── checkpoint.py         # 权重解包与输入通道推断
│   ├── training_run.py       # 训练运行配置及恢复流程
│   ├── training_checkpoint.py # 完整训练状态的保存与校验
│   ├── openvino.py           # 长度感知 GRU 的 OpenVINO 图适配
│   ├── character_stats.py    # 字符统计；同目录另有校对和清理工具
│   └── synthesis/            # 补字、红章、边线及定向合成入口
├── datadoc/                  # 数据维护、合成说明和整理历史
│   └── analysisdata/         # 按日期保存的错误样本分析
├── train.md                  # 历史训练记录
├── data/                     # 本地图片与标注，不提交
├── checkpoints/              # 本地训练检查点，不提交
└── runs/                     # TensorBoard 运行日志，不提交
```

包外的 `docs/recogniton/` 保存架构、运行说明和历史模型文档，见下方快速索引。
根目录 `recognition_inference.py`、`recognition_export_onnx.py` 分别负责多后端推理和导出；
多层 BiGRU 模型用 `recognition_inference_layer.py` 和 `recognition_export_onnx_layer.py`
（导出默认生成 `best_recognition_layer.onnx`，不覆盖单层产物）；
`recognition_benchmark_cpu.py`、`recognition_quantize_static.py` 负责 CPU 评测和静态量化。

## 常用入口

以下命令在仓库根目录执行；先核对脚本顶部的数据、权重和输出路径。

```bash
.venv/bin/python recognition/train.py
.venv/bin/python recognition/evaluate_val.py
.venv/bin/python recognition/inference.py
```

训练只提供续训检查点、初始权重及原字典、本次检查点目录参数；超参数在 `train.py`。
评估和包内推理使用顶部常量，不提供命令行参数。
根目录 [recognition_inference.py](../recognition_inference.py) 提供多后端命令行入口。

## 关键约束

- 标注为 `relative/path.png<TAB>文字`，`<TAB>` 是真实制表符；不自动去外层引号或替换标点。
- 路径相对于每份标注文件所在目录，必须精确匹配；不猜扩展名、不用缺图替代张量。
- 共享透明合成、局部去红章和动态补边，不二次裁剪正文。
- `img_widths` 不含右侧 32 像素保护区和 batch 补边；混合宽度推理必须传入真实宽度。
- GRU 层数与 checkpoint 绑定，默认 2 层；加载旧单层权重需显式 `CRNN(num_layers=1)`。
- 字典内容与顺序绑定权重。完整续训还校验源码全文哈希，注释改动也会影响旧检查点恢复。

## 文档快速索引

完整分类见 [Recognition 文档索引](../docs/recogniton/README.md)。`docs/recogniton/` 下的
`guides/` 保存当前操作说明，`guides/code-walkthrough/` 保存随代码更新的新手导读；
`history/model/`、`history/batch/`、`history/experiments/` 分别保存旧模型、批量问题和实验记录。
历史资料用于追溯，配置和接口以当前说明为准。

### 当前使用

| 内容 | 文档 |
| --- | --- |
| 图片、模型、张量与解码 | [当前架构](../docs/recogniton/recognition.md) |
| 训练、续训、评估、导出与 CPU 部署 | [运行说明](../docs/recogniton/guides/running.md) |
| 划分、统计、字典和清理工具 | [数据维护](datadoc/dataset.md) |
| 统计列、清理条件、备份和引用检查范围 | [数据工具详解](../docs/recogniton/guides/data-tools.md) |
| 补字、红章、边线和错误定向合成 | [合成说明](datadoc/synthesis.md) |
| CTC、CER、真实宽度等术语 | [术语表](../docs/terminology.md) |
| 未解决的实现和证据问题 | [核对待办](../docs/code-doc-todolist.md) |

### 新手代码导读

建议按 dataset → model → converter 阅读，三篇均以当前实现为准。

| 查询主题 | 文档 |
| --- | --- |
| 图片加载、增强与分桶 | [dataset 导读](../docs/recogniton/guides/code-walkthrough/dataset-analysis.md) |
| CNN、BiGRU 与 CTC 输出 | [model 导读](../docs/recogniton/guides/code-walkthrough/model-analysis.md) |
| 字典编码与解码 | [converter 导读](../docs/recogniton/guides/code-walkthrough/converter-analysis.md) |

### 历史代码与问题说明

| 查询主题 | 文档及适用范围 |
| --- | --- |
| 旧 ResNet/BiLSTM 模型初始化 | [CRNN 初始化](../docs/recogniton/history/model/model-crnn-init.md) |
| 旧模型前向传播 | [CRNN 前向传播](../docs/recogniton/history/model/model-crnn-forward.md) |
| 旧模型张量形状 | [数据流](../docs/recogniton/history/model/model-data.md)、[宽度下采样](../docs/recogniton/history/model/model-data-2.md) |
| 单图与批量推理结果差异 | [批量推理与补边](../docs/recogniton/history/batch/batch-padding.md)，保留旧问题分析 |

### 历史实验与数据记录

| 查询主题 | 文档 |
| --- | --- |
| 历次训练过程 | [训练历史](train.md) |
| 指定运行的收敛与后续建议 | [ocr_exp_1788970646 训练分析](../docs/recogniton/history/experiments/training-analysis-1788970646.md) |
| 红章与训练增强 | [2026-09-07 专项分析](../docs/recogniton/history/experiments/recognition-analysis-2026-09-07.md)、[2026-09-08 去章核对](../docs/recogniton/history/experiments/red-seal-audit-2026-09-08.md) |
| CPU 导出、推理与量化 | [2026-09-09 CPU 报告](../docs/recogniton/history/experiments/recognition-cpu-report-2026-09-09.md) |
| 尾字增加或丢失 | [尾部实验记录](../docs/recogniton/history/experiments/tail-experiments-history.md)，原报告缺失情况已标明 |
| 标点整理及合成图片恢复 | [2026-09-10 数据整理](datadoc/dataset-history-2026-09-10.md) |

## 常用检查

```bash
.venv/bin/python -m pytest tests/test_recognition_preprocess.py tests/test_recognition_batch_invariance.py -q
.venv/bin/python -m pytest tests/test_recognition_geometry_augment.py -q
.venv/bin/python recognition/evaluate_robustness.py   # 几何/成像扰动下的鲁棒性验收
```
