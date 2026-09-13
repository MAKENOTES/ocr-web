# Recognition 数据维护

本文指导识别数据的日常维护，包括标注约定、字典管理、数据检查和合成样本清理。

图片使用 PNG，标注为 `relative/path.png<TAB>文字`。共享解析器不做标点替换或外层引号剥离；
文字首尾空白由加载器去除。流程细节见[数据工具](../../docs/recogniton/guides/data-tools.md)，
模型输入见[当前架构](../../docs/recogniton/recognition.md)。

## 常用检查

以下命令在仓库根目录执行；报告可能覆盖同名结果，运行前核对输入与输出。

```bash
# 从指定标注提取新字典，默认输入实际为 append1_annotation.txt
.venv/bin/python recognition/utils/dictionary.py --input recognition/data/train_annotation.txt
# 比较 keys-new 与 keys，结果写 keys-append.txt，不修改 keys
.venv/bin/python recognition/utils/compare_key_files.py
# 核对 val 的字典外字符，写报告；不传 --move-unknown 不搬移图片
.venv/bin/python recognition/utils/check_val_annotation_keys.py
# 比较 train/val 中文字完全相同的记录；文字相同不等于图片泄漏
.venv/bin/python recognition/utils/compare_train_val_same_text.py
```

新增字典只用于核对，不可直接配旧模型。修改 keys 的内容或顺序后需重新训练、导出及验证。

## 评估与错误分析

配置 `recognition/evaluate_val.py` 顶部权重、标注和报告路径后运行；确认报告属于本次评估，
再用下列工具提取真实字符中的错误部分：

```bash
.venv/bin/python recognition/evaluate_val.py
.venv/bin/python recognition/utils/extract_error_keys.py
```

错误字符清单不等于缺数据清单。先结合原图区分错标、裁切、去章、低清及字典问题，
再决定补图或合成；当前缺少可验证结论的事项见[待办](../../docs/code-doc-todolist.md)。

## 统计与清理

```bash
# 默认只统计 train、val、synthesis1、synthesis2 四份标注
.venv/bin/python recognition/utils/character_stats.py
# 默认仅预览命中字符、整条标注及图片的删除计划
.venv/bin/python recognition/utils/remove_characters_by_stats.py
# 预览 train+val 已覆盖字符构成的合成样本
.venv/bin/python recognition/utils/remove_common_synthesis.py
```

`remove_characters_by_stats.py` 不是“删除 keys 外字符”的工具：它从 keys 内筛选
train/val 零出现、合成合计不足 10 次或仅在一份合成数据出现的字符。
显式 `--apply` 会备份并删除整条样本及相应字典行；只校验四份清单，不检查其余训练输入。
`remove_common_synthesis.py` 默认要求整条文字的每个字符在 train+val 至少出现 100 次，且不修改 keys。
具体条件、引用检查范围与恢复边界见[数据工具](../../docs/recogniton/guides/data-tools.md)。

## 划分与格式迁移

`recognition/split_dataset.py` 默认预览，`--apply` 才从完整 all 数据复制出独立 train/val，
输出非空时拒绝执行。已有人工调整的数据不能直接重跑初始划分。
`check_dataset_annotations.py` 专门核对 all=train+val+val_unknown；只有这些来源清单齐备才可使用，
不能用于代替当前九份训练清单的全量一致性检查。

`convert_annotation_delimiter.py --check` 只预览首逗号到 Tab 的转换，去掉 `--check` 会覆盖输入。
该工具不负责去正文外层引号；只适用于已确认格式的旧数据，不能对当前 Tab 文件做 CSV 猜测。

## 合成与历史

生成、重渲染和补回图片见[合成说明](synthesis.md)。
[2026-09-10 整理记录](dataset-history-2026-09-10.md)保留当时标点处理和恢复数量，
不表示加载器会自动归一化文本，也不证明当前磁盘数据与该批次一致。
历史错误分类见 [2026-09-06](analysisdata/errdata20260906.md)、[2026-09-09](analysisdata/errdata20260909.md)、
[2026-09-12](analysisdata/errdata20260912.md)。
