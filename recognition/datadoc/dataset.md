# 数据集

## 数据集后续计划

### 不同场景数据在训练和推理中的处理


## 数据集脚步

### 新增数据集
1、从数据集txt文件中 提取keys 到 keys-new.txt
python recognition/utils/dictionary.py
2、对比 keys-new.txt 与 keys.txt 得到哪些文字是新增的文字
python recognition/utils/compare_key_files.py

### 数据集评估
1、从 不同的数据集txt文件，对模型评估
python recognition/evaluate_val.py

2、从error_cases.txt中提取 错误字符 到 keys-error.txt
python recognition/utils/extract_error_keys.py

3、核对  keys-error.txt 和  error_cases.txt
需注意，暂时需人工核对，对识别错误样本进行分类。只有 确定 缺少数据集 再去收集数据集。

### 精简数据集

1、 统计字符占比
python recognition/utils/character_stats.py

2、移除非目标keys中字符
预览
python recognition/utils/remove_characters_by_stats.py
真移除
python recognition/utils/remove_characters_by_stats.py --apply

### 数据集核对工具
1、recognition/utils/check_val_annotation_keys.py   得到对应数据集中字符， 不在keys.txt ， 并可以移动到 unknown 中
2、recognition/utils/compare_train_val_same_text.py  对比训练集与验证集中完全相同的识别文字。
3、recognition/utils/character_stats.py  统计字符占比
4、recognition/utils/remove_characters_by_stats.py 移除非目标keys中字符

4、recognition/split_dataset.py 把某个数据集划分成 训练和验证，首次用后续预计不会用。
5、recognition/utils/check_dataset_annotations.py  核对多个数据集 ，暂时不能用 all 删除了 而且新增了几个数据集。
6、被依赖
convert_annotation_delimiter

### 数据归一化
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

八组合成标注共 77,303 条，当前图片路径为 `../data/synthesis1_images/`、
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


### 数据合成
[synthesis.md](synthesis.md)

## 错误数据集记录

### 20260906
[errdata20260906.md](analysisdata/errdata20260906.md)

[errdata20260909.md](analysisdata/errdata20260909.md)
