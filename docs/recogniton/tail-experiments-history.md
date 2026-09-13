# 尾部识别实验历史

以下保留原架构文档中的实验结论；本次未重新训练或复核本地产物。
当前长度契约见[架构](recognition.md)。

2026-09-12 的尾部增漏专项实验（原报告 `recognition/datadoc/analysisdata/errdata20260912-tail-experiment.md` 当前缺失）
已对真实训练／验证集比较有效背景扩展，并完成同起点两轮训练对照；8 px 扩展仍使验证错误增加，
因此保留当前 `img_widths` 和 32 px CNN 保护区契约，不在推理入口给有效宽度直接加常数。
后续验证（原报告 `recognition/datadoc/analysisdata/errdata20260912-tail-followup.md` 当前缺失）进一步核对了 CTC 文本概率，
前缀搜索、跨上下文重评分及两轮背景 blank 监督均未通过尾部和整体验收，仍不替换正式解码或权重。

以上是保留的历史文字，原报告缺失时无法重新核对其量化结论；补证据要求见[待办](../code-doc-todolist.md)。
