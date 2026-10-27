# Recognition 文档索引

本文按当前操作、新手导读和历史记录分类索引识别文档，帮助读者快速找到所需说明。

当前接口以[架构](recognition.md)为准，常用代码入口见[包 README](../../recognition/README.md)。
历史资料保留原版本范围，不作为当前参数或精度依据。

## 目录分类

```text
docs/recogniton/
├── README.md           # 本目录索引
├── recognition.md      # 当前架构与数据契约，保留固定入口路径
├── guides/             # 当前操作与新手导读
│   ├── running.md      # 训练、评估与导出
│   ├── deployment.md   # 部署产物构建、依赖与升级
│   ├── data-tools.md   # 数据校对、统计与清理
│   └── code-walkthrough/ # dataset、model、converter 代码导读
└── history/
    ├── model/          # 旧模型结构与张量说明
    ├── batch/          # 旧批量推理问题分析
    └── experiments/    # 历史训练、预处理与性能实验
```

## 当前操作

| 内容 | 文档 |
| --- | --- |
| 输入图片、真实宽度、模型与解码 | [当前架构](recognition.md) |
| 训练、评估与导出 | [运行说明](guides/running.md) |
| 部署产物、依赖选择与升级 | [部署说明](guides/deployment.md) |
| 划分、字符统计及清理 | [数据工具](guides/data-tools.md) |
| 数据维护与合成操作 | [数据维护](../../recognition/datadoc/dataset.md)、[合成说明](../../recognition/datadoc/synthesis.md) |

## 新手代码导读

以下三篇已按当前实现更新，建议按 dataset → model → converter 阅读：先理解输入，再看模型输出，最后看标签编码与解码。

| 主题 | 文档 |
| --- | --- |
| 加载、增强、分桶与补边 | [dataset 导读](guides/code-walkthrough/dataset-analysis.md) |
| CNN、BiGRU 与序列输出 | [model 导读](guides/code-walkthrough/model-analysis.md) |
| 字典编号与 CTC 解码 | [converter 导读](guides/code-walkthrough/converter-analysis.md) |

## 历史模型与问题

| 主题 | 文档 |
| --- | --- |
| 旧 CRNN 初始化与前向 | [初始化](history/model/model-crnn-init.md)、[前向传播](history/model/model-crnn-forward.md) |
| 旧模型张量与下采样 | [数据流](history/model/model-data.md)、[宽度下采样](history/model/model-data-2.md) |
| 单图与批量结果差异 | [批量推理与补边](history/batch/batch-padding.md) |

## 历史实验

| 主题 | 文档 |
| --- | --- |
| 红章与训练增强 | [2026-09-07 分析](history/experiments/recognition-analysis-2026-09-07.md) |
| 去红章样本核对 | [2026-09-08 记录](history/experiments/red-seal-audit-2026-09-08.md) |
| CPU 推理、导出与量化 | [2026-09-09 报告](history/experiments/recognition-cpu-report-2026-09-09.md) |
| 指定运行的训练收敛 | [ocr_exp_1788970646 分析](history/experiments/training-analysis-1788970646.md) |
| 尾部识别对照 | [尾部实验记录](history/experiments/tail-experiments-history.md)，原报告缺失情况已标明 |

未解决的问题及检查方法见[核对待办](../code-doc-todolist.md)。
