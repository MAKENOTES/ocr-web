# 2026-09-10 数据整理历史

以下保留原有整理记录及数量，本次未重新测量本地数据。当前操作见[数据维护](dataset.md)。

2026-09-10 已按以下规则更新 train/val 和字典，并重新生成八组合成数据。
- 无空格
- 只有中文括号、冒号
- ；;  只保留中文分号；
- ，,  只保留中文逗号，
- `_`：填空线、文字装饰线不标注；文件名、邮箱、引物名等正文中的下划线保留，字典也保留 `_`。
- —  这个横杠归一化成 一 ？

本次 train 修改 709 条、val 修改 168 条，合计去掉 246 个版式下划线；图片及样本条数不变。
字典移除 `;`、`,`，由 3009 字符变为 3007 字符，其余相对顺序保留。
旧权重必须绑定旧字典，新字典需重新训练和导出。

八组合成标注共 77,303 条，当时图片路径为 `../data/synthesis1_images/`、
`synthesis2_images/`、`synthesis3_images/` 及各专项的 `synthesis_*_images/`。
图片删除后，可使用 `restore.py` 按现有 txt 和对应 `samples.jsonl` 补回，
保留文字、文件名、字体、背景及专项效果类别，不改写 txt 和字典；
用法见 [数据合成说明](synthesis.md)。
2026-09-10 已按当前八份 txt 补回全部 77,303 张图片，标注与字典哈希保持不变；
执行参数、逐图记录及全量校验见
[图片恢复报告](../../outputs/recognition_restore_20260910_135144/README.md)。
标点规范化时同时纠正了旧 degraded 标注误指向 confusable 图片目录的问题。
该次标点调整的备份、逐条变更、执行参数和校验结果见
[`outputs/recognition_symbol_normalization_20260910_132714`](../../outputs/recognition_symbol_normalization_20260910_132714/README.md)。


