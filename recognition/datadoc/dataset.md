# 数据集

## 数据集后续计划

### 统计不同字符 占比

### 梳理数据集增强

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

使用 [oh-my-ocr/text_renderer](https://github.com/oh-my-ocr/text_renderer) 生成 Recognition
补字数据，入口为 [recognition/utils/synthesis/generate.py](../utils/synthesis/generate.py)。

#### 生成规则

- 目标字来自 `recognition/keys-append.txt`，默认每个字生成 **100 张**图片。
- 配字来自 `recognition/keys.txt`，排除所有目标字，并且必须受当前字体支持。
  每张只含一个目标字，保证每个目标字的图片数和出现次数都严格等于 100。
- 两份字符文件均要求每行一个可见字符、无重复；目标字不在 `keys.txt` 中或没有字体支持时直接报错。
- 风格参考 `recognition/data/train_images` 和 `recognition/data/val_images`，默认按图片数量比例
  抽样 1,600 张，提取背景色、文字色、高度及留白，不复制原图中的文字，也不读取验证集标注。
- 输出为完整水平文字框、RGB PNG，高度 16～64 像素、文本长度 1～24 字符。
  配字是随机字符组合，包含轻微模糊和噪声，风格以宋体/黑体、白底黑字及彩色界面文字为主。
- 默认使用本机思源黑体/宋体常规与粗体、Droid Sans Fallback、文鼎宋体中可用的字体，
  逐字检查字体覆盖；可通过 `--fonts` 指定 TTF、OTF 或 TTC 文件。

#### 使用方法

以下命令在仓库根目录运行：

```bash
# 首次安装：锁定 text_renderer 版本，保留现有训练依赖
.venv/bin/python -m pip install --no-deps -r recognition/utils/synthesis/requirements.txt

# 默认每个目标字生成 100 张
.venv/bin/python recognition/utils/synthesis/generate.py

# 每字生成 2 张作预览，使用独立输出路径
.venv/bin/python recognition/utils/synthesis/generate.py --per-char 2 \
  --output-dir outputs/synthesis_preview/images \
  --annotation outputs/synthesis_preview/annotation.txt \
  --report-dir outputs/synthesis_preview/report
```

图片和报告目录必须为空，标注文件必须不存在。已有产物时再次运行会报错，需通过
`--output-dir`、`--annotation`、`--report-dir` 指定新的输出位置。
生成中断时保留已有图片和 `.partial` 标注供排查，全部生成成功后才发布正式标注。

常用参数：

| 参数 | 用途与默认值 |
| --- | --- |
| `--targets` | 目标字文件，默认 `recognition/keys-append.txt` |
| `--keys` | 配字及模型字典，默认 `recognition/keys.txt` |
| `--per-char` | 每个目标字的图片数，默认 `100` |
| `--fonts` | 一个或多个字体文件，例如 `--fonts /path/font.ttf /path/font.ttc` |
| `--references` | 一个或多个风格参考图片目录，默认 train_images、val_images |
| `--style-samples` | 参考图片抽样数，默认 `1600` |
| `--seed` | 随机种子，默认 `20260907` |

#### 输出与训练接入

| 默认输出路径 | 内容 |
| --- | --- |
| `recognition/data/synthesis_images/` | 合成 PNG，按 `uXXXXXX_NNN.png` 命名，包含目标字 Unicode 码点及样本序号 |
| `recognition/data/synthesis_annotation.txt` | 训练标注，格式为 `relative/path.png<TAB>文字`，路径相对于标注文件所在目录 |
| `recognition/data/synthesis_report/summary.json` | 总图片数、逐字计数、随机种子、字体计数及输入校验和 |
| `recognition/data/synthesis_report/samples.jsonl` | 每张图片的文字、目标字、字体及参考风格 |
| `recognition/data/synthesis_report/preview.png` | 抽样图片预览 |

标注复用当前 Recognition 加载器的 **Tab 分隔格式**。需要参与训练时，将
`recognition/data/synthesis_annotation.txt` 加入 `recognition/train.py` 的
`TRAIN_ANNOTATION` 列表，验证集继续使用原来的 val 标注。
工具不会自动修改训练配置、字典或已有训练/验证数据；生成产物被 Git 忽略。

2026-09-07 已生成一批：399 个目标字 × 100 张，共 **39,900 张**。
全部 PNG 解码、标注与图片对应关系、字典范围和逐字配额校验通过，并抽查了 100 个样本的
Recognition 张量加载。该批核验结果保存在
`recognition/data/synthesis_report/verification.json`（本次额外核验产物，生成脚本不自动输出）。

## 错误数据集记录

### 20260906
[errdata20260906.md](analysisdata/errdata20260906.md)
