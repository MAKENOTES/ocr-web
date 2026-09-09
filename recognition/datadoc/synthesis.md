# Recognition 数据合成

## 字体和渲染

合成脚本共用 [generate.py](../utils/synthesis/generate.py) 中的字体加载和正文渲染逻辑。
默认从本机以下字体中选择，未安装的默认字体不加入候选：

| 字体 | 字重 | 默认字体文件位置 |
|---|---|---|
| Noto Sans CJK SC（思源黑体） | Regular、Bold | `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`、`NotoSansCJK-Bold.ttc`（同目录） |
| Noto Serif CJK SC（思源宋体） | Regular、Bold | `/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc`、`NotoSerifCJK-Bold.ttc`（同目录） |
| Droid Sans Fallback | 常规 | `/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf` |
| AR PL UMing CN（文鼎宋体） | 常规 | `/usr/share/fonts/truetype/arphic/uming.ttc` |

正文使用 `text_renderer` 的 `FontText` / `draw_text_on_bg` 接口。依赖版本由
[requirements.txt](../utils/synthesis/requirements.txt) 锁定，首次使用时安装：

```bash
.venv/bin/python -m pip install --no-deps -r recognition/utils/synthesis/requirements.txt
```

三个入口的日期均在启动时自动取系统本地日期，序号从 `00001` 开始，在整批内连续递增。
报告统一放在 `recognition/utils/synthesis/synthesis_report/` 下，按批次建立子目录：
`synthesisN_YYYYMMDD/`、`synthesis_balanced_YYYYMMDD/`、`synthesis_red_YYYYMMDD/`。
各子目录分别保存 `summary.json`、`samples.jsonl`、`preview.png` 等，互不覆盖。
显式传入 `--report-dir` 时直接使用指定目录；三个入口仍支持自定义图片目录和标注文件路径。

## 1、全新或者占比少的字符
对应脚本：[recognition/utils/synthesis/generate.py](../utils/synthesis/generate.py)。
适用于已整理好目标字清单，希望给每个字新增固定数量样本的情况。

### 字符来源说明
- **目标字**默认读取 `recognition/keys-append.txt`，可用 `--targets` 指定其他字符文件。
- **配字**来自 `recognition/keys.txt`，排除本批全部目标字，并检查当前字体是否支持。


### 文件夹和文件命名规则

| 产物 | 默认路径或命名 |
|---|---|
| 图片目录 | `recognition/data/synthesis1_images/` |
| 图片文件 | `synthesis1_YYYYMMDD_00001.png`、`synthesis1_YYYYMMDD_00002.png`……  |
| 标注文件 | `recognition/data/synthesis1_annotation.txt` |
| 报告目录 | `recognition/utils/synthesis/synthesis_report/` |

默认从 `synthesis1` 开始选择未占用编号。如果对应图片目录、标注、`.partial` 标注或当天报告批次存在，
自动尝试 `synthesis2`、`synthesis3`，以此类推；图片文件名前缀与选中的编号一致。
显式指定的图片目录或标注名称包含 `synthesisN` 时沿用该编号，已有非空输出仍拒绝覆盖。

已有 `synthesis1_annotation.txt`、`synthesis2_annotation.txt` 中的图片按各自标注顺序重新编号，
日期取迁移当天。可用 `python recognition/utils/rename_synthesis_images.py` 预览，
加 `--apply` 同步改名和更新标注路径；正文和图片内容不变。
原始标注、新旧文件名映射及图片 SHA-256 保存在 `outputs/synthesis_naming_migration/rename-*/`。

### 风格要求
默认从 `recognition/data/train_images/`、`val_images/` 合并抽样最多 **1600 张**参考图，
按图像数量比例取样。提取背景色、文字色、高度、文字高度占比和左右留白，排除纯色等无有效文字风格的框。

### 使用命令
```bash
# 默认每个目标字新增 100 张
python recognition/utils/synthesis/generate.py
```

## 2、根据统计字符占比文件，生成字符来平衡字符占比
对应脚本：[recognition/utils/synthesis/generate_balanced.py](../utils/synthesis/generate_balanced.py)。
用于根据统计结果补齐低频字，当前默认生成 **50000 张**。
实际约束是补齐字符出现次数到指定范围，不是让所有字符的占比百分数相等。

### 字符来源说明
- 默认读取 `outputs/recognition_character_stats/character_stats.csv`，可用 `--stats` 指定其他文件。
- 目标字为原次数 **严格小于 100** 的字符，包含 0 次。
- 目标字必须在 `recognition/keys.txt` 中。配字只能从 CSV 次数已达 100、在 `keys.txt` 中且当前字体支持的字符中选择。


### 文件夹和文件命名规则
| 产物 | 默认路径或命名 |
|---|---|
| 图片目录 | `recognition/data/synthesis_balanced_YYYYMMDD_images/` |
| 图片文件 | `synthesis_balanced_YYYYMMDD_00001.png`、`synthesis_balanced_YYYYMMDD_00002.png`…… |
| 标注文件 | `recognition/data/synthesis_balanced_YYYYMMDD_annotation.txt` |
| 报告目录 | `recognition/utils/synthesis/synthesis_report/` |


### 风格要求
沿用第一组的 train/val 风格采样、字体、完整正文渲染和轻微模糊/噪声。
默认参考图数量为 **1600 张**，随机种子为 `20260908`。

### 使用命令
```bash
# 校验 CSV、字体、参考图和完整配额计划，不生成图片或报告文件
python recognition/utils/synthesis/generate_balanced.py --dry-run
# 当前默认：生成 50000 张，低频目标字最终补到 110 次
python recognition/utils/synthesis/generate_balanced.py
```

## 3、特殊样本：生成受红章干扰的字符样本
对应脚本：[recognition/utils/synthesis/generate_red_stamp.py](../utils/synthesis/generate_red_stamp.py)。
默认生成 **1000 张**，用于增强红章遮挡、叠印情况下的正文识别能力，不设置逐字补齐配额。

### 字符来源说明
- **正文**默认从 `recognition/data/train_annotation.txt` 读取，可用 `--corpus` 指定其他标注文件。
- 正文中的字符必须已在 `keys.txt` 中，整条正文须由一个字体完整覆盖。
- **红章文字**默认是 `合成样本识别测试专用章`，可用 `--stamp-text` 修改；章字只要求字体支持，作为干扰层，不加入正文标签。

### 文件夹和文件命名规则
| 产物 | 默认路径或命名 |
|---|---|
| 图片目录 | `recognition/data/synthesis_red_YYYYMMDD_images/` |
| 图片文件 | `YYYYMMDD_00001.png` 至默认批次的 `YYYYMMDD_01000.png` |
| 标注文件 | `recognition/data/synthesis_red_YYYYMMDD_annotation.txt` |
| 报告目录 | `recognition/utils/synthesis/synthesis_report/` |

### 风格要求
参考清单为 [red_stamp_references.txt](../utils/synthesis/red_stamp_references.txt)，
包含指定的 **43 张图：train 29 张、val 14 张**。条目路径相对于 `recognition/data/`；
使用 `--data-dir` 时相对于该目录。`--references` 在本脚本中表示**清单文件**，不是前两组的图片目录列表。

### 使用命令
```bash
# 默认生成 1000 张红章干扰样本
python recognition/utils/synthesis/generate_red_stamp.py
```

## 4、特殊样本：四个方向的表格边线

脚本：[generate_lines.py](../utils/synthesis/generate_lines.py)，默认生成 **1000 张**。
上横线、下横线、左竖线、右竖线、四边框各 **200 张**。其他数量也按五种类型均分，差值不超过 1。
参考清单 [line_references.txt](../utils/synthesis/line_references.txt) 包含指定的 15 张 val 图片。
仅提取纸面颜色、正文颜色、字号和长度统计；统计正文时排除长边线，不复制参考像素或标签。

字符从实际字典 `recognition/keys.txt` 随机采样（仓库没有 `keys.py`），排除空白及不可打印字符，
整行由同一字体完整覆盖。字体以常规宋体为主，文字长度为 2～24，保留完整字形和原有轻微噪声、模糊。
在正文画布外增加 1～2 像素边线，边线距正文画布 1～3 像素；左右边线为竖线。
线条不属于正文标注，不承诺 1000 张覆盖全部字典或逐字配额。

```bash
.venv/bin/python recognition/utils/synthesis/generate_lines.py --images 1000
```

默认产物（日期为运行当天）：

| 产物 | 路径 |
|---|---|
| RGB PNG | `recognition/data/synthesis_lines_YYYYMMDD_images/synthesis_lines_YYYYMMDD_00001.png` 起 |
| 逗号加引号标注 | `recognition/data/synthesis_lines_YYYYMMDD_annotation.txt` |
| 当前训练加载器使用的 Tab 标注 | `recognition/data/synthesis_lines_YYYYMMDD_annotation_training.txt` |
| 报告及预览 | `recognition/utils/synthesis/synthesis_report/synthesis_lines_YYYYMMDD/` |

两份标注的图片路径和文字完全一致，路径均相对于标注所在目录；正文中的双引号在逗号格式中按 CSV 规则转义。
**加入训练时使用 `_annotation_training.txt`**，当前加载器只接受 Tab 分隔格式。
`summary.json` 保存方向计数、参考计数、字符次数和输入校验和；`samples.jsonl` 保存逐图字体、风格及边线参数。
可用 `--keys`、`--references`、`--fonts`、`--seed`、`--output-dir`、`--annotation`、`--report-dir` 修改输入输出。
已有非空产物拒绝覆盖；生成失败保留 `.partial` 排查，不发布完整标注。不会自动合并进 train/val。
