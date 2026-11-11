# 文字识别

本文介绍文字识别包的用途、工程目录和常用操作，并提供架构、工具、新手导读及历史文档索引。

本包将完整 detection 文字框识别为 Unicode 文本。当前模型为 Mobile 风格 CNN、
2 层双向 GRU 和 CTC，默认输入 RGB、高度 32 像素。

## 工程目录层级

下图列出主要目录和文件，省略包初始化文件、缓存及具体数据产物。

```text
recognition/
├── README.md                 # 包入口与文档快速索引
├── train.py                  # 训练循环、验证调度与检查点保存
├── training_data.py          # 训练数据路径、输入尺寸与 DataLoader 构建
├── evaluate_val.py           # 独立标注集评估与错误报告
├── evaluate_robustness.py    # 几何/成像扰动下的鲁棒性验收
├── inference.py              # PyTorch 推理
├── runtime/                  # 自包含部署快照：生成副本 + 手写入口
│   ├── engine.py             # RecognitionEngine 稳定 API（手写）
│   ├── backends.py           # ONNX、OpenVINO、PyTorch 后端（手写）
│   ├── cli.py                # python -m recognition.runtime（手写）
│   ├── loader.py             # 图片收集与 DataLoader（手写）
│   └── ...                   # preprocessing/dataset/model/converter/checkpoint/openvino 由同步脚本生成
├── split_dataset.py          # 初始训练集/验证集划分
├── keys.txt                  # 字符字典，内容与顺序绑定模型
├── components/
│   ├── dataset.py            # 训练数据集、分桶与 DataLoader
│   ├── inference_data.py     # 推理数据集、验证变换与动态补边
│   ├── preprocessing.py      # 路径校验、RGB 合成、去红章与定高缩放
│   ├── augmentation.py       # 源分辨率几何增强与缩放后光度增强
│   └── converter.py          # 字符编码与 CTC 解码
├── models/
│   ├── model.py              # 当前 Mobile CNN + 多层 BiGRU 模型
│   └── model_resnet50.py      # 历史模型，实际使用 ResNet34 + BiLSTM
├── utils/
│   ├── annotation.py         # Tab 标注解析与格式化
│   ├── checkpoint.py         # 检查点读取与通道/层数推断
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
两个推理脚本保留旧命令和默认值，内部统一调用 `recognition/runtime/`。`runtime/` 中
`preprocessing.py`、`dataset.py`、`model.py`、`converter.py`、`checkpoint.py` 和
`openvino.py` 由 `tools/sync_recognition_runtime.py` 从原路径生成，不要手改；修改源文件后
运行同步脚本。`runtime/` 内部全部使用相对导入，可只拷贝该目录到其他项目并按包改名导入；
`tools/build_recognition_bundle.py` 在此基础上附带模型、字典和依赖清单，见
[部署说明](../docs/recogniton/guides/deployment.md)。

训练、评估和导出继续使用 `components/`、`models/`、`utils/` 中的原实现，runtime 只是
单向生成的部署快照，不反向影响训练代码。

## 常用入口

以下命令在仓库根目录执行；先核对脚本顶部的数据、权重和输出路径。

```bash
.venv/bin/python recognition/train.py
.venv/bin/python recognition/evaluate_val.py
.venv/bin/python recognition/inference.py
```

训练数据路径、输入尺寸和加载参数在 `training_data.py`，训练轮数与保存策略在 `train.py`；
评估和包内推理使用各自顶部常量，均不提供命令行参数。
根目录 [recognition_inference.py](../recognition_inference.py) 保留旧命令和默认值；
新接入使用 `python -m recognition.runtime` 或部署产物中的 `example.py`。

## 推理与部署

### 推理

仓库内有两个根目录兼容入口，内部都调用 `recognition/runtime/`；新接入建议直接用 runtime：

```bash
# 包内 PyTorch 推理：路径由 recognition/inference.py 顶部常量控制
.venv/bin/python recognition/inference.py

# 单层模型兼容入口，默认 auto 后端
.venv/bin/python recognition_inference.py \
  --model_path best_recognition.onnx \
  --keys_path recognition/keys.txt \
  --image_path recognition/data/test_images

# 多层 BiGRU 模型兼容入口，默认 OpenVINO
.venv/bin/python recognition_inference_layer.py \
  --model_path best_recognition_layer.onnx \
  --keys_path recognition/keys.txt \
  --image_path recognition/data/test_images

# 不依赖根目录入口的新接入
.venv/bin/python -m recognition.runtime \
  --model_path models/model.onnx \
  --keys_path keys.txt \
  --image_path /path/to/images \
  --backend onnx --batch_size 8
```

`--backend` 支持 `auto`、`onnx`、`openvino` 和 `pytorch`；兼容入口可用默认值，
新接入必须显式提供模型、字典和图片路径。字典内容与顺序必须和模型导出一致；
batch>1 依赖模型包含 `img_widths` 输入和 GRU `sequence_lens`，旧模型会退回逐张行为。
结果按传入图片顺序返回，`--sort_by_width` 只改变组 batch 顺序。示例中的模型需先用
`recognition_export_onnx.py`（单层）或 `recognition_export_onnx_layer.py`（多层）导出，
或替换为已有模型路径。

### 部署

`recognition/runtime/` 是单向生成的部署快照。修改 `components/`、`models/`、`utils/`
中的推理依赖后先同步，再构建部署包：

```bash
.venv/bin/python tools/sync_recognition_runtime.py          # 重新生成 runtime
.venv/bin/python tools/sync_recognition_runtime.py --check  # 只检查

.venv/bin/python tools/build_recognition_bundle.py \
  --model best_recognition_layer.onnx \
  --keys recognition/keys.txt \
  --out dist/recognition_runtime --zip
```

产物包含 `recognition/runtime/`、模型、`keys.txt`、`requirements*.txt`、`example.py`
和 `manifest.json`，默认在隔离路径下实际推理一次。部署方按后端安装
`requirements-onnx.txt`、`requirements-openvino.txt` 或 `requirements-pytorch.txt`，
然后运行 `python example.py <图片或目录>` 或 `python -m recognition.runtime ...`。
只嵌入其他项目时，可以只复制 `recognition/runtime/` 并重命名包，模型、字典和依赖
由目标项目管理。

完整产物结构、参数、升级与回滚见[部署说明](../docs/recogniton/guides/deployment.md)；
模型导出、CPU 评测和量化见[运行说明](../docs/recogniton/guides/running.md)。

## 关键约束

- 标注为 `relative/path.png<TAB>文字`，`<TAB>` 是真实制表符；不自动去外层引号或替换标点。
- 路径相对于每份标注文件所在目录，必须精确匹配；不猜扩展名、不用缺图替代张量。
- 共享透明合成、局部去红章和动态补边，不二次裁剪正文。
- `img_widths` 不含右侧 32 像素保护区和 batch 补边；混合宽度推理必须传入真实宽度。
- GRU 层数与 checkpoint 绑定，默认 2 层；加载旧单层权重需显式 `CRNN(num_layers=1)`。
- 字典内容与顺序绑定权重；修改 `keys.txt` 后必须重新训练、导出并验证模型。

## 文档快速索引

完整分类见 [Recognition 文档索引](../docs/recogniton/README.md)。`docs/recogniton/` 下的
`guides/` 保存当前操作说明，`guides/code-walkthrough/` 保存随代码更新的新手导读；
`history/model/`、`history/batch/`、`history/experiments/` 分别保存旧模型、批量问题和实验记录。
历史资料用于追溯，配置和接口以当前说明为准。

### 当前使用

| 内容 | 文档 |
| --- | --- |
| 图片、模型、张量与解码 | [当前架构](../docs/recogniton/recognition.md) |
| 训练、评估、导出与 CPU 部署 | [运行说明](../docs/recogniton/guides/running.md) |
| 部署产物、依赖与升级 | [部署说明](../docs/recogniton/guides/deployment.md) |
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
.venv/bin/python -m pytest tests/test_recognition_runtime.py tests/test_recognition_runtime_sync.py tests/test_recognition_bundle.py -q
.venv/bin/python tools/sync_recognition_runtime.py --check   # runtime 快照与源文件一致性
.venv/bin/python -m pytest tests/test_recognition_augmentation.py -q
.venv/bin/python -m pytest tests/test_recognition_training.py tests/test_recognition_training_data.py -q
.venv/bin/python recognition/evaluate_robustness.py   # 几何/成像扰动下的鲁棒性验收
```
