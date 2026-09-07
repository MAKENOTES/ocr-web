# 数据集

## 数据集后续计划

### 统计不同字符 占比

### 梳理数据集增强

### 不同场景数据在训练和推理中的处理



## 数据集脚步

### 新增数据集
1、从数据集txt文件中 提取keys 到 keys-new.txt
recognition/utils/dictionary.py
2、对比 keys-new.txt 与 keys.txt 得到哪些文字是新增的文字
recognition/utils/compare_key_files.py

### 数据集评估
1、从 不同的数据集txt文件，对模型评估
recognition/evaluate_val.py

2、从error_cases.txt中提取 错误字符 到 keys-error.txt
recognition/utils/extract_error_keys.py

3、核对  keys-error.txt 和  error_cases.txt
需注意，暂时需人工核对，对识别错误样本进行分类。只有 确定 缺少数据集 再去收集数据集。

### 精简数据集

1、 统计字符占比
recognition/utils/character_stats.py

2、移除非目标keys中字符
预览
recognition/utils/remove_characters_by_stats.py
真移除
recognition/utils/remove_characters_by_stats.py --apply

### 数据集核对工具
1、recognition/utils/check_val_annotation_keys.py   得到对应数据集中字符， 不在keys.txt ， 并可以移动到 unknown 中
2、recognition/utils/compare_train_val_same_text.py  对比 训练集和验证集 中 重复的样本。
3、recognition/utils/character_stats.py  统计字符占比
4、recognition/utils/remove_characters_by_stats.py 移除非目标keys中字符

4、recognition/split_dataset.py 把某个数据集划分成 训练和验证，首次用后续预计不会用。
5、recognition/utils/check_dataset_annotations.py  核对多个数据集 ，暂时不能用 all 删除了 而且新增了几个数据集。
6、recognition/utils/convert_annotation_delimiter.py  转换数据集分割符号工具，后续应该不用。
7、之前的工具，应该不用了。
recognition/utils/normalization_label.py     
recognition/utils/remove_annotation_outer_quotes.py

### 数据归一化
- 无空格
- 只有中文括号
- /home/weiqiang/AI/ocr-web/recognition/data/bench1_annotation.txt   中 做了 ()  和 :   归一化成中文的

### 数据合成
合成工具（可自己生成网站/PDF 风格数据）：PaddleOCR 文档里推荐了 text_renderer、SynthText、TextRecognitionDataGenerator 等。

## 错误数据集记录

### 20260906
[errdata20260906.md](analysisdata/errdata20260906.md)