# Recognition 数据合成

## 字体和渲染

三个脚本共用 [generate.py](../utils/synthesis/generate.py) 中的字体加载和正文渲染逻辑。
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

## 1、全新或者占比少的字符
对应脚本：[recognition/utils/synthesis/generate.py](../utils/synthesis/generate.py)。
适用于已整理好目标字清单，希望给每个字新增固定数量样本的情况。

### 字符来源说明
- **目标字**默认读取 `recognition/keys-append.txt`，可用 `--targets` 指定其他字符文件。
- **配字**来自 `recognition/keys.txt`，排除本批全部目标字，并检查当前字体是否支持。


### 文件夹和文件命名规则

| 产物 | 默认路径或命名 |
|---|---|
| 图片目录 | `recognition/data/synthesis_images/` |
| 图片文件 | `u{Unicode码点:06x}_{字内序号:03d}.png` |
| 标注文件 | `recognition/data/synthesis_annotation.txt` |
| 报告目录 | `recognition/data/synthesis_report/` |

码点使用至少 6 位小写十六进制，序号按每个目标字分别从 `000` 开始。
例如“中”的码点是 U+4E2D，前两张为 `u004e2d_000.png`、`u004e2d_001.png`。
报告中的 `counts` 用于核对每个目标字是否达到本批配额。

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
| 图片文件 | `YYYYMMDD_00001.png`、`YYYYMMDD_00002.png`…… |
| 标注文件 | `recognition/data/synthesis_balanced_YYYYMMDD_annotation.txt` |
| 报告目录 | `outputs/synthesis_balanced_YYYYMMDD_report/` |



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
| 图片目录 | `recognition/data/synthesis_red_stamp_YYYYMMDD_images/` |
| 图片文件 | `YYYYMMDD_00001.png` 至默认批次的 `YYYYMMDD_01000.png` |
| 标注文件 | `recognition/data/synthesis_red_stamp_YYYYMMDD_annotation.txt` |
| 报告目录 | `outputs/synthesis_red_stamp_YYYYMMDD_report/` |

### 风格要求
参考清单为 [red_stamp_references.txt](../utils/synthesis/red_stamp_references.txt)，
包含指定的 **43 张图：train 29 张、val 14 张**。条目路径相对于 `recognition/data/`；
使用 `--data-dir` 时相对于该目录。`--references` 在本脚本中表示**清单文件**，不是前两组的图片目录列表。

### 使用命令
```bash
# 默认生成 1000 张红章干扰样本
.venv/bin/python recognition/utils/synthesis/generate_red_stamp.py
```
