# Recognition 数据合成

各入口的写入行为不同：普通生成直接创建新产物，重渲染和恢复默认只预览。
生成前核对 `--help`、输出目录及实际字典。术语见[术语表](../../docs/terminology.md)。

## 按原标注补回已删除的图片

图片已删除、同批次 `samples.jsonl` 仍在时，使用 `restore.py` 按 txt 中的原路径逐条重建。
正文和文件名必须与生成记录一致；只允许目录前缀变化。脚本不改写标注和字典，
默认只校验，`--apply` 才生成，目标图片已存在时拒绝覆盖。

```bash
.venv/bin/python recognition/utils/synthesis/restore.py \
  --annotation recognition/data/synthesis_red_annotation.txt \
  --records outputs/recognition_symbol_normalization_20260910_132714/reports/synthesis_red/samples.jsonl \
  --report-dir outputs/synthesis_red_restored
# 检查完成后，在同一命令末尾加 --apply 执行。
```

字体、颜色、字号和效果类别来自逐图记录，正文渲染复用 `generate.py`；
红章调用 `overlay_stamp`，边线调用 `add_lines`，低清调用 `degrade`，边缘调用 `apply_edge`。
细节随机参数重新采样，不承诺与已删除图片逐像素相同。每行使用 `--seed + 行号`
作为独立种子，新报告记录图片 SHA-256、实际效果参数和标注/字典哈希。
生成失败会保留已经成功写入的图片与记录，须核对失败原因后处理该批产物，不能直接覆盖重跑。

## 按已有文本重新渲染

`regenerate.py` 用于保留已整理的样本条数和文本覆盖、重新生成图片的情况，
复用 `generate.py` 的字体、渲染器和风格采样。输入必须先完成标注校对；脚本不猜测
下划线语义、不静默删除字典外字符，也不读取旧合成图片作为底图。
默认只检查文本、字典、字体与参考风格，传入 `--apply` 才生成。输出路径必须全新。

```bash
.venv/bin/python recognition/utils/synthesis/regenerate.py \
  --source-annotation recognition/data/synthesis1_annotation.txt \
  --prefix synthesis1 \
  --output-dir recognition/data/synthesis1_regenerated_images \
  --annotation recognition/data/synthesis1_regenerated_annotation.txt \
  --report-dir outputs/synthesis1_regenerated
# 检查完成后，在同一命令末尾加 --apply 执行。
```

默认只从 `train_images/` 抽取 1,600 张风格参考，支持 `--references`、`--fonts`、
`--style-samples`、`--seed` 和 `--keys` 覆盖。报告保存输入/字典/字体哈希和每张图的
正文、源行号、字体与风格；全部成功后才发布 Tab 标注，失败保留 `.partial` 供排查。
红章、边线、形近字、低清和边缘效果仍使用各自的专项入口，避免普通重渲染丢失专项效果。

分号和逗号的当前标注规范分别为 `；`、`，`；字典仍保留正文字符 `_`。
填空线或文字装饰线不应作为 `_` 写入标签，文件名、邮箱、引物名等实际包含的下划线保留。
更新字典后须重新训练和导出；旧 checkpoint 应绑定其原字典。

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

历史迁移工具 `rename_synthesis_images.py` 当前不存在，不能再按旧命令改名。
现有图片路径必须与标注精确一致；如需重新生成，使用上方 `regenerate.py` 或 `restore.py`，
不要手工只改一侧路径。历史迁移产物仅作为本地记录，不是当前入口依赖。

### 风格要求
默认从 `recognition/data/train_images/`、`val_images/` 合并抽样最多 **1600 张**参考图，
按图像数量比例取样。提取背景色、文字色、高度、文字高度占比和左右留白，排除纯色等无有效文字风格的框。

### 使用命令
```bash
# 默认每个目标字新增 100 张
.venv/bin/python recognition/utils/synthesis/generate.py
```

## 2、根据统计字符占比文件，生成字符来平衡字符占比
对应脚本：[recognition/utils/synthesis/generate_balanced.py](../utils/synthesis/generate_balanced.py)。
用于根据统计结果补齐低频字，当前默认生成 **50000 张**。
实际约束是补齐字符出现次数到指定范围，不是让所有字符的占比百分数相等。
读取 CSV 第一列字符、最后一列次数，默认补至 110 次，可设 100～120；新增次数为目标减基线。
默认基线包含 train、val、synthesis1/2，不能当作完整训练集次数；后续批次须重算统计避免重复补齐。

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
.venv/bin/python recognition/utils/synthesis/generate_balanced.py --dry-run
# 当前默认：生成 50000 张，低频目标字最终补到 110 次
.venv/bin/python recognition/utils/synthesis/generate_balanced.py
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
| 图片文件 | `synthesis_red_YYYYMMDD_00001.png` 至默认批次的 `synthesis_red_YYYYMMDD_01000.png` |
| 标注文件 | `recognition/data/synthesis_red_YYYYMMDD_annotation.txt` |
| 报告目录 | `recognition/utils/synthesis/synthesis_report/` |

### 风格要求
参考清单为 [red_stamp_references.txt](../utils/synthesis/red_stamp_references.txt)，
包含 **161 张图：train 33 张、val 128 张**。保留原 43 张，追加
[2026-09-09 错误复核](analysisdata/errdata20260909.md) 中确认有章的 118 张，覆盖全部红章主类及其他分类中的叠印样本；
混合错标样本也只提取数值风格，不读取 val 标注或复制参考像素。条目路径相对于 `recognition/data/`；
使用 `--data-dir` 时相对于该目录。`--references` 在本脚本中表示**清单文件**，不是前两组的图片目录列表。

默认均分 10 种场景：参考位置、上弧、下弧、星形、行首、行尾、仅残弧、淡章、磨损、浓章。
同时变化整章角度、圆章纵横比、局部印泥压力与遮盖程度；新增短框参考使正文长度覆盖单字至长行。
正文仍为训练语料整条标签，章字不入标注。每张检查可见红色与正文交叠，报告记录场景分布及兜底次数。
预处理边界见 [Recognition 当前架构](../../docs/recogniton/recognition.md#保守去红章与训练增强)；
生成器只叠加干扰，不执行识别去章，不保证每张合成章都能被预处理移除。

### 使用命令
```bash
# 默认生成 1000 张红章干扰样本，每种场景 100 张
.venv/bin/python recognition/utils/synthesis/generate_red_stamp.py

# 只生成淡章、磨损和行尾残章，仍需使用空的输出目录
.venv/bin/python recognition/utils/synthesis/generate_red_stamp.py --scenes faded worn right_edge
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

## 5、裁切、低清和汉字形近错误定向合成

三个独立入口默认各生成 **1000 张 RGB PNG**，参考
[2026-09-09 错误分析](analysisdata/errdata20260909.md) 对应分类的全部图片：
裁切或边缘残字 37 张、低清或笔画粘连 18 张、汉字形近误识 43 张。
只测量参考图的纸面色、墨色、字号和留白，按参考文本长度抽样训练语料；不复制验证图片中的文字像素。
字体沿用公共字体候选，以宋体为主、黑体为辅；粘连模式提高粗体采样权重。
字号沿用公共渲染器的 16～64 像素图片高度范围，不保证与每张原图逐像素一致。

```bash
.venv/bin/python recognition/utils/synthesis/generate_edge.py
.venv/bin/python recognition/utils/synthesis/generate_degraded.py
.venv/bin/python recognition/utils/synthesis/generate_confusable.py
```

| 入口 | 合成规则 | 默认 1000 张的分配 |
|---|---|---|
| [generate_edge.py](../utils/synthesis/generate_edge.py) | 贴边、轻裁、相邻字一列残片 | 右贴边 400、右轻裁 200、左贴边 100、左轻裁 100、上下轻裁 100、残片 100 |
| [generate_degraded.py](../utils/synthesis/generate_degraded.py) | 缩小放大并模糊、笔画粘连、局部笔画减淡、灰色扫描带 | 四种各 250 |
| [generate_confusable.py](../utils/synthesis/generate_confusable.py) | 从分析表替换段提取汉字目标，覆盖真实字及误识字，保留正常清晰度 | 按目标字均分指定目标图片数，余数依目标顺序分配 |

### 正文和标注边界

- 正文默认读取 `recognition/data/train_annotation.txt`，按长度选取并重新渲染。
  形近字类会替换其中一个汉字为本张目标字，得到新组合；标签始终使用实际渲染文字。
  这些组合不保证语义通顺。排除与三类参考图完整文本相同的候选（比较时去普通空格）。
- 裁切最多进入正文一像素；左右轻裁仅用于正文高度至少 20 像素的汉字边界，上下至少 24 像素。
  若受影响字区墨迹损失超过 10%，或上下裁切墨迹损失超过 5%，退回贴边。
  窄标点只贴边。模式配额包含退回贴边的样本，实际裁入像素数见 `cut_ink_pixels`。
  残片模式保留完整正文，在左右边缘加入相邻合成字的一列墨迹，**残片不加入标签**。
  本批不模拟严重缺字或需要猜测框外文字的情况。
- 低清模式缩放比例为 0.65～0.85；粘连只混入部分扩展墨迹；破碎减淡少量墨迹，不整块抹字。
- 形近字从文本对齐的差异段提取，不将分类内所有差异都断言为严格的一对一形近字。
  汉字误识成符号时仍补真实汉字；符号不作为本类目标。每张目标至少出现一次，
  语料内可能另含目标字，因此指定目标图片数与字符总出现次数不同。

### 参数、产物与复现

三个入口共用 `error_common.py` 的参数、输入校验和产物写出，各自显式传入 `make_sample` 实现类别算法。
复用现有 `generate.py` 的字体和正文渲染、`generate_lines.py` 的正文风格测量、
`generate_red_stamp.py` 的训练文本加载；不改变训练、评估和推理预处理。

支持 `--images`（默认 1000）、`--seed`（默认 20260910）、`--analysis`、`--corpus`、
`--keys`、`--fonts`、`--data-dir`、`--output-dir`、`--annotation`、`--report-dir`。
`--dry-run` 校验输入并测量风格，不写产物；参考文件缺失、字典或字体不覆盖时带具体输入报错。
`--data-dir` 控制默认输出根目录，输入语料另用 `--corpus` 指定。

将下表中的 `类别` 替换为 `edge`、`degraded` 或 `confusable`；日期在启动时取本机日期。

| 产物 | 默认位置 |
|---|---|
| 图片 | `recognition/data/synthesis_类别_YYYYMMDD_images/synthesis_类别_YYYYMMDD_00001.png` 起 |
| **训练用 Tab 标注** | `recognition/data/synthesis_类别_YYYYMMDD_annotation.txt` |
| 报告和预览 | `recognition/utils/synthesis/synthesis_report/synthesis_类别_YYYYMMDD/` |

每类只生成 **`_annotation.txt`**，格式为 `relative/path.png<TAB>文字`，可直接用于训练。
不额外生成 `_annotation_training.txt`。图片路径相对于标注文件所在目录，现有训练清单不自动修改。
报告含 `summary.json`、逐图 `samples.jsonl` 和 `preview.png`，记录参考图、字体、语料行号、目标字及退化参数，
并保存分析文档、语料、字典、参考图和字体的 SHA-256。预览长行会缩小，原图保留完整尺寸。
已有输出拒绝覆盖；中断时保留 `.partial`，不发布未完成的标注。重新生成请指定新的输出和报告路径。

### 2026-09-10 历史生成记录（本次未重测）

三类各 1000 张，合计 3000 张，已逐张验证 PNG 可读、RGB 模式、字典覆盖、两版标注一致及路径存在，
并抽查最终预览。37、18、43 张参考图均被采样且文件校验和保持不变。
边缘类实际裁入正文一像素的样本为 234 张，其余为贴边或残片；
低清四种模式各 250 张；形近字覆盖 84 个目标汉字，每字指定 11～12 张。
相关合成及识别预处理测试 123 项通过；尚未进行模型训练或评估识别收益。

本批标注已按后续要求统一为单个 `_annotation.txt`，内容沿用原训练版的 Tab 格式；
已核对三类各 1000 条并移除冗余 `_annotation_training.txt`，图片与文字内容不变。
