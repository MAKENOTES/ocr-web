# 文字识别

本包将完整 detection 文字框识别为 Unicode 文本。当前模型为 Mobile 风格 CNN、
单层双向 GRU 和 CTC，默认输入 RGB、高度 32 像素。

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
- 字典内容与顺序绑定权重。完整续训还校验源码全文哈希，注释改动也会影响旧检查点恢复。

## 文档快速索引

当前架构与运行说明统一放在 `docs/recogniton/`；旧模型说明放在 `history/model/`，
旧批量问题放在 `history/batch/`。历史资料用于追溯，配置和接口以当前说明为准。

### 当前使用

| 内容 | 文档 |
| --- | --- |
| 图片、模型、张量与解码 | [当前架构](../docs/recogniton/recognition.md) |
| 训练、续训、评估、导出与 CPU 部署 | [运行说明](../docs/recogniton/running.md) |
| 划分、统计、字典和清理工具 | [数据维护](datadoc/dataset.md) |
| 统计列、清理条件、备份和引用检查范围 | [数据工具详解](../docs/recogniton/data-tools.md) |
| 补字、红章、边线和错误定向合成 | [合成说明](datadoc/synthesis.md) |
| CTC、CER、真实宽度等术语 | [术语表](../docs/terminology.md) |
| 未解决的实现和证据问题 | [核对待办](../docs/code-doc-todolist.md) |

### 历史代码与问题说明

| 查询主题 | 文档及适用范围 |
| --- | --- |
| 图片加载、增强与分桶 | [dataset 分析](../docs/recogniton/dataset-analysis.md)，2026-09-07 代码快照 |
| CNN、BiGRU 与 CTC 输出 | [model 分析](../docs/recogniton/model-analysis.md)，2026-09-07 代码快照 |
| 字典编码与解码 | [converter 分析](../docs/recogniton/converter-analysis.md)，2026-09-07 代码快照 |
| 旧 ResNet/BiLSTM 模型初始化 | [CRNN 初始化](../docs/recogniton/history/model/model-crnn-init.md) |
| 旧模型前向传播 | [CRNN 前向传播](../docs/recogniton/history/model/model-crnn-forward.md) |
| 旧模型张量形状 | [数据流](../docs/recogniton/history/model/model-data.md)、[宽度下采样](../docs/recogniton/history/model/model-data-2.md) |
| 单图与批量推理结果差异 | [批量推理与补边](../docs/recogniton/history/batch/batch-padding.md)，保留旧问题分析 |

### 历史实验与数据记录

| 查询主题 | 文档 |
| --- | --- |
| 历次训练过程 | [训练历史](train.md) |
| 指定运行的收敛与后续建议 | [ocr_exp_1788970646 训练分析](../docs/recogniton/training-analysis-1788970646.md) |
| 红章与训练增强 | [2026-09-07 专项分析](../docs/recogniton/recognition-analysis-2026-09-07.md)、[2026-09-08 去章核对](../docs/recogniton/red-seal-audit-2026-09-08.md) |
| CPU 导出、推理与量化 | [2026-09-09 CPU 报告](../docs/recogniton/recognition-cpu-report-2026-09-09.md) |
| 尾字增加或丢失 | [尾部实验记录](../docs/recogniton/tail-experiments-history.md)，原报告缺失情况已标明 |
| 标点整理及合成图片恢复 | [2026-09-10 数据整理](datadoc/dataset-history-2026-09-10.md) |

## 常用检查

```bash
.venv/bin/python -m pytest tests/test_recognition_preprocess.py tests/test_recognition_batch_invariance.py -q
```
