# Recognition 架构

## 1. 职责

Recognition 接收 detection 输出的单个文字框，输出 Unicode 文本。当前它是项目的实际主链路，支持彩色/深浅背景、PNG 和动态宽度处理。

当前基线为 MobileNet 风格 CNN + 单层双向 GRU + CTC head，输入高度固定为 32，宽度按比例缩放并在 batch 内动态 padding。

## 2. 数据与预处理

```mermaid
flowchart TD
    A[annotation: relative_path TAB unquoted label] --> DS[TextRecognitionDataset]
    I[元素图片] --> DS
    DS --> PNG[严格读取标注中的 PNG 路径]
    PNG --> RGB[读取 RGB + 透明通道合成]
    RGB --> SEAL[按局部叠印关系去红章 / 保护同行红字]
    SEAL --> RESIZE[完整 detection 框固定高度32 + 等比宽度]
    RESIZE --> MINW[窄图按边缘背景色补到宽48]
    MINW --> AUG{训练集?}
    AUG -->|是| TA[颜色/极性/模糊/轻噪声]
    AUG -->|否| NORM[仅 Normalize]
    TA --> BUCKET[按宽高比桶采样]
    NORM --> CTX[右侧至少32px CNN背景保护区]
    BUCKET --> CTX
    CTX --> PAD[batch 动态 padding 到8倍数]
    PAD --> OUT[images, labels/paths, img_widths]
```

相关实现：

- `../../recognition/components/dataset.py`：图片路径校验、PNG 透明通道处理、缩放、增强、桶采样和动态 padding；
- `../../recognition/utils/checkpoint.py`：checkpoint 解包以及旧灰度/新 RGB 模型通道识别；
- `../../recognition/components/converter.py`：字典编码和显式 TBC/BTC 的 CTC greedy decode。

训练、验证、PyTorch 推理和多后端推理复用同一套不裁剪基础预处理。训练可额外使用随机增强；验证和测试禁止随机增强。

### 保守去红章与训练增强

`remove_red_seal` 在透明背景合成之后、缩放和随机增强之前统一执行，输入为 RGB。
现在按**局部红色与正文的交错叠印关系**判断，支持紧框内的断章、章字、星形和局部弧线，
不再要求每个连通区域都能拟合成完整圆章，也不要求正文墨迹比章墨迹多。

- 颜色候选：R ≥150、R 比 G/B 至少高25、G/B 色差不超过40；红色总面积不超过35%。
  纸面至少40%、边缘至少50%为近中性浅色，允许灰纸、轻微偏色及扫描颗粒。
- 正文证据：灰黑像素的 RGB 最大值不超过170、通道色差不超过35，且明确排除红色候选，
  防止红字被同时计为“黑字”。正文至少16像素，且不少于红像素数量的15%。
- 局部判定：相距不超过两倍正文行高的红色列分组；按正文列密度检查组内黑字支持率，
  同时要求红色既进入正文行、又有至少8%越过上边或下边。只剩半个章也可通过。
- 逐区域保护：分组通过后仍检查每个红色连通区域的黑字支持；完全位于正文行内的小红字
  保留。只有一个连通区域时，额外检查弧线、椭圆或复杂章形，避免简单边框和折线误触发。

最终只将通过检查的原始红像素替换为纸面中位色，不进行掩码膨胀、闭运算或内容裁剪。
纯红字、普通同行黑红混排、红底白字及无正文支持的红色内容保留；同组出现一个章也不会
清空整组红色。与正文完全对齐的章字、极暗的印泥和缺乏正文证据的片段仍可能残留。
红色标记/合法红字若与黑字交错且越过正文行，也存在误判可能；不透明印泥覆盖的正文无法
由此恢复。这些阈值属于启发式规则，上线效果仍需用红章及合法红字分组评估。
已有权重的验证/推理输入也会受这一确定性预处理影响。

2026-09-08 使用用户指定的43张问题图复核：触发从4张提高到42张（val 14/14，train 28/29）；
`000351-000003.png` 的 R 超出 G/B 较大值的最大色差仅14，没有满足候选条件的红色，保持原样。
对照376张含候选红色、标签为“删除”或“导入说明”的图片，均未改写。
43张中的近中性深色像素（RGB 最大值≤170、通道色差<25）改写数为0。
这批图片用于规则调试，不是独立测试集；触发率不等于去章完整率或 OCR 准确率，
对照结果也不代表所有合法红字均能保留。逐图统计和前后对比图保存在本地
`../../outputs/red_seal_audit_20260908`，原图未修改。

训练随机增强保持现有参数，各项作用如下（验证/推理不启用）：

| 类型 | 概率 | 覆盖场景 |
| --- | --- | --- |
| ColorJitter | 70% | 曝光变化、扫描反差、彩字浓淡、轻微白平衡偏色 |
| RandomAutocontrast | 20% | 扫描软件拉伸灰度、截图对比度差异 |
| RandomInvert | 8% | 深色主题、黑底白字；残留红章也会随之反色 |
| GaussianBlur | 15% | 轻微失焦、低分辨率截图、缩放造成的模糊 |
| AddGaussianNoise | 15% | 扫描颗粒、相机传感器轻噪声 |
| RandomErasing | 5% | 污点、磨损、局部遮挡；面积 0.5%～3%，仍可能遮挡小数点或短文本关键笔画 |

灰度转换用于兼容旧单通道模型，ToTensor 和 Normalize 用于统一张量及数值范围，
均不是随机增强。完整 detection 框继续等比缩放与动态 padding，不增加旋转、透视或二次裁剪。

### 初始训练集/验证集划分

当 `all_images/` 和 `all_annotation.txt` 保存全量数据时，使用
`../../recognition/split_dataset.py` 做确定性的 80/20 均匀系统抽样。脚本按标注顺序每隔约
5 条抽取一条到验证集，其余进入训练集；总数不能被 5 整除时，会在 4～5 条之间
均匀调整间隔，使验证集数量等于最接近 20% 的整数。

```bash
# 先校验数据并预览数量，不修改文件
.venv/bin/python recognition/split_dataset.py

# 确认后从 all_images 复制图片，并生成 train/val 标注
.venv/bin/python recognition/split_dataset.py --apply
```

脚本不会修改 `all_images/` 和 `all_annotation.txt`。为避免覆盖手工调整，
`train_images/`、`val_images/`、`train_annotation.txt`、`val_annotation.txt` 必须为空；
缺图、重复标注、未标注 PNG 或目标文件冲突也会让脚本在写入前停止。

### 四份标注校对

手工调整 train/val 或隔离字典外字符样本后，运行下列只读脚本校对四份标注：

```bash
.venv/bin/python recognition/utils/check_dataset_annotations.py
```

脚本仅按不含目录的图片文件名判断，例如 `du20260902-002391.png`。它要求
`all = train + val + val_unknown`、三个子集两两没有同名图片，并逐项比较 all 与所属
split 中同名图片后面的识别文字。校对通过返回 0；缺失、多余、跨 split 重复或文字不同
会列出具体文件名并返回 1。每类默认最多显示 20 条详情，传入 `--max-details 0` 可显示全部。

### 字符频次与数据集覆盖率

```bash
# 默认统计 train、val、synthesis 三份标注，按合计占比倒序打印全部字符，并将完整明细写入 CSV
.venv/bin/python recognition/utils/character_stats.py

# 终端只显示“王”的次数和占比，CSV 仍保留 keys 中的全部字符
.venv/bin/python recognition/utils/character_stats.py --chars 王

# 按 train.py 的五份训练标注统计，排除 val
.venv/bin/python recognition/utils/character_stats.py \
  --annotations recognition/data/train_annotation.txt \
  recognition/data/append_annotation.txt recognition/data/bench1_annotation.txt \
  recognition/data/bench2_annotation.txt recognition/data/synthesis_annotation.txt \
  --output-dir outputs/recognition_training_character_stats
```

只生成 `../../outputs/recognition_character_stats/character_stats.csv`，不再生成 JSON。
CSV 和终端均按未舍入的合计占比倒序输出，同频保持 keys 原顺序，零出现字符也保留；
只列 keys 中的字符，不修改 keys。占比列在前，次数列统一放在最后。
默认列顺序为 `字符、train_占比、val_占比、synthesis_占比、合计_占比、
train_次数、val_次数、synthesis_次数、合计_次数`。
`次数` 包括同一标签内重复出现的字符；`占比` 为该字符次数除以对应文件全部文字字符数
（含字典外字符），直接显示百分数，保留五位小数，例如 `0.00000%`。出现 80 次、总共
1,000 个字符时，占比为 `8.00000%`。合计占比为所有输入文件的该字总次数除以文字总字符数，不是各占比平均。
合计累加本次传入的全部标注记录，不去重，默认包含 val，不能当作训练集总量。
终端同时提供各文件及合计的文字总字符数，方便核对分母。
默认运行也会在终端逐行打印全部字符；`--chars` 只限制终端展示，不过滤输出文件。
`../../outputs` 在 `../../.gitignore` 中，编辑器全局搜索可能排除它；直接打开文件搜索，或显式运行
`rg -n 王 outputs/recognition_character_stats/character_stats.csv`。
脚本复用训练的 Tab 解析和首尾空白处理规则，不进行标点、Unicode 或转义归一化；
空标签按训练规则跳过并计数，格式错误直接报告具体文件与行号。此脚本不检查图片。

### 按字符统计移除补充数据

`../../recognition/utils/remove_characters_by_stats.py` 使用上述默认 CSV，并核对当前
`keys.txt` 和 train、val、synthesis 三份标注中的实际字符次数。
CSV 缺列、字符集合或次数不一致时停止，需先重新运行 `character_stats.py`。

两种方式均要求字符在 train 和 val 中出现次数为 0：

- `--mode 1`：synthesis 出现次数严格小于 10，包含零次字符。
- `--mode 2`：在目标数据集中恰好一份出现；当前目标只有 synthesis，因此选中其中所有出现过的字符，次数不限。
- `--mode both`：两种方式取并集，也是默认方式；当前会选中所有 train/val 次数为 0 的字典字符，包括 synthesis 零次字符。仅根据本次 CSV 筛选一次，不递归清理。

```bash
# 默认只预览，并打印全部命中字符、各文件删除数量及前 20 条图片明细
.venv/bin/python recognition/utils/remove_characters_by_stats.py

# 单独预览第二种方式；0 表示显示全部图片明细
.venv/bin/python recognition/utils/remove_characters_by_stats.py --mode 2 --max-details 0

# 真正执行两种方式的并集（也可以显式指定 --mode 1 或 --mode 2）
.venv/bin/python recognition/utils/remove_characters_by_stats.py --apply

# 只处理指定字符，仍须满足所选方式；不传 --apply 则只预览
.venv/bin/python recognition/utils/remove_characters_by_stats.py --chars 癫 --apply
```

只要标签包含命中字符，就删除对应的**整条标注和图片**，不会只删标签里的个别字。
只更新 synthesis 标注以及 keys 中命中的字符行，保留其余行内容、换行和
字典相对顺序。图片路径按标注文件所在目录解析并去重；待删图片缺失、路径越出数据目录、
通过符号链接访问，或仍被这三份文件中任一保留标注引用时，在写入前报错。
train、val 标注不修改，不扫描这三份标注之外的图片引用。

只有传入 `--apply` 才写入文件。执行前将所有待修改的原始标注、keys、待删图片及输入
CSV 备份到 `outputs/recognition_character_removal/removal-*/`；`manifest.json` 记录
原路径、备份路径及删除明细。单个文字文件使用原子替换，多文件操作不构成事务，若中断或
磁盘错误导致部分完成，可按 manifest 恢复。预览不创建备份或任何输出文件。
可用 `--csv`、`--keys`、`--data-dir`、`--backup-dir` 指定路径。
删除后需重新生成字符统计。删除整条样本会减少其他字符的次数、改变其数据集分布，
使本轮未命中的字符在下一轮满足条件；此时需基于新 CSV 再次预览、执行。
变更 keys 会改变模型 ABI，必须重新训练、导出并验证模型。

### 移除 train/val 已覆盖字符的合成样本

`../../recognition/utils/remove_common_synthesis.py` 专门处理
`../../recognition/data/synthesis1_annotation.txt` 和 `synthesis2_annotation.txt`。
读取 `../../outputs/recognition_character_stats/character_stats.csv` 的 `train_次数`、`val_次数`，
以 **train + val 合计 >= 100** 判断常见字，包含恰好 100 次；不使用包含 synthesis 的总次数
来判断是否需要补充。可用 `--threshold` 调整阈值。

- `--match all`（默认）：整条非空文字的每个字符均为常见字才移除，保留含低频字或 CSV 未收录字符的样本。
- `--match any`：包含任一常见字就移除整条样本，也会连带减少其中的低频字。

两种方式都按**整条标注和对应 PNG 图片**删除，不单独删标签中的字，不修改 keys、train、val。
预览打印两份标注的原条数、删除/保留条数、连带减少的低频字次数，以及合计从达标降到阈值以下
的字符数量。若所有样本都包含低频字，默认 `all` 会删除 0 条。

```bash
# 默认预览 all 规则，不写文件
.venv/bin/python recognition/utils/remove_common_synthesis.py

# 预览 any 规则，先检查连带减少的低频字数量
.venv/bin/python recognition/utils/remove_common_synthesis.py --match any

# 备份后按指定规则执行；此例采用 all
.venv/bin/python recognition/utils/remove_common_synthesis.py --match all --apply
```

复用现有清理工具的统计校验、备份和原子替换逻辑；检查四份标注实际次数与 CSV 一致，
待删 PNG 必须是数据目录内的普通相对路径且存在，并且未被四份文件中保留的标注引用。
不会扫描其他标注文件的引用。仅 `--apply` 执行写入，先将待改标注和待删图片备份到
`outputs/recognition_common_synthesis_removal/removal-*/`，同时保存原始 CSV 与 `manifest.json`。
多文件操作不是事务，异常中断可按 manifest 从备份恢复。执行后重新运行 `character_stats.py`。
支持 `--csv`、`--data-dir`、`--backup-dir` 和 `--max-details`；默认不执行真实数据删除。

### 补字合成数据

`../../recognition/utils/synthesis/generate.py` 使用
[oh-my-ocr/text_renderer](https://github.com/oh-my-ocr/text_renderer) 的
`FontText` 和 `draw_text_on_bg` 字形渲染接口。针对 `../../recognition/keys-append.txt`
每个字默认生成 100 张；配字仅取 `../../recognition/keys.txt` 中**非目标字**且当前字体支持的字符，
每张仅包含一个目标字，保证每个目标字的图片数和出现次数都严格等于 100。
字符文件要求每行一个可见字符、无重复；目标不在 keys 或没有字体支持时直接报错。
不修改 keys、现有训练/验证数据或训练配置。

```bash
# 可选工具依赖：锁定上游提交，只装绘字模块，保留现有训练依赖版本
.venv/bin/python -m pip install --no-deps -r recognition/utils/synthesis/requirements.txt

# 默认选择未占用的 synthesisN_images 目录
.venv/bin/python recognition/utils/synthesis/generate.py

# 小批量预览，输出到独立目录
.venv/bin/python recognition/utils/synthesis/generate.py --per-char 2 \
  --output-dir outputs/synthesis_preview/images \
  --annotation outputs/synthesis_preview/annotation.txt \
  --report-dir outputs/synthesis_preview/report
```

默认字体为本机思源黑体/宋体常规与粗体、Droid Sans Fallback、文鼎宋体；
不存在的默认字体不加入候选，没有覆盖目标字的字体则停止。也可通过 `--fonts file.ttf file.ttc`
指定字体文件；TTC 默认选择 Noto 简体中文比例字体、文鼎 CN 字形，逐字检查 cmap，
不以缺字方框替代文字。字体文件不复制到仓库。

默认按图片数量比例抽样 train_images/val_images 共 1,600 张，仅提取背景/前景色、
高度、文字占比与左右留白，不取验证集标注或复制真实含字像素。透明 PNG 通过共享
`load_rgb_image` 读取。合成 RGB PNG 高度 16～64、文本长度 1～24，按参考宽高比估计
长度，配字为随机字符组合；采用水平文字、完整字形、轻微模糊/噪声，不添加未标注文字。
默认种子 `20260907`，可用 `--seed`、`--per-char`、`--references`、`--style-samples` 调整。

输出：

- `recognition/data/synthesisN_images/synthesisN_YYYYMMDD_00001.png`：按系统日期和整批连续序号命名。
- `recognition/data/synthesisN_annotation.txt`：复用当前加载器的 `relative/path.png<TAB>文字`
  格式；当前代码已使用 Tab，AGENTS.md 中逗号加引号的描述是旧格式。
- `recognition/utils/synthesis/synthesis_report/synthesisN_YYYYMMDD/summary.json`：逐字计数、种子、字体计数、输入校验和。
- 同目录 `samples.jsonl`、`preview.png`：每张图的目标字/字体/参考风格及抽样预览。

图片和报告目录必须为空，标注必须不存在；失败保留已生成图片和 `.partial` 标注供排查，
不会覆盖或自动清理旧产物。成功后才原子发布正式标注。上述数据均被 Git 忽略。
默认从 N=1 开始，已有图片目录、标注、未完成标注或当天报告批次占用时自动递增编号。
使用合成数据训练时，将实际生成的 `recognition/data/synthesisN_annotation.txt` 加入
`../../recognition/train.py` 的 `TRAIN_ANNOTATION` 列表，验证集继续使用原来的 val 标注。

### 表格边线干扰合成（默认 1000 张）

`../../recognition/utils/synthesis/generate_lines.py` 从 `recognition/keys.txt` 随机取可见字符，
参考 `line_references.txt` 中指定的 15 张 val 框的颜色、字号和文字长度，以细宋体为主重绘正文。
只使用参考风格统计，不复制参考正文。上横线、下横线、左竖线、右竖线、四边框各 200 张，
边线在完整正文画布外绘制，保留字形。字符须由所选字体完整覆盖；不做逐字数量保证。

```bash
.venv/bin/python recognition/utils/synthesis/generate_lines.py --images 1000
```

RGB PNG 输出到 `recognition/data/synthesis_lines_YYYYMMDD_images/`；
同级 `synthesis_lines_YYYYMMDD_annotation.txt` 使用仓库指南的逗号引号格式，
`synthesis_lines_YYYYMMDD_annotation_training.txt` 使用当前加载器需要的 Tab 格式，**训练请使用后者**。
两份标注路径均相对于各自所在目录，标签只含正文；已有输出拒绝覆盖，不自动修改 train/val。
报告和预览在 `recognition/utils/synthesis/synthesis_report/synthesis_lines_YYYYMMDD/`。
具体参数和产物说明见 `../../recognition/datadoc/synthesis.md`。

### 红章干扰合成（默认 1000 张）

`../../recognition/utils/synthesis/generate_red_stamp.py` 默认读取同目录
`red_stamp_references.txt` 中的 161 张参考框（train 33 张、val 128 张），提取正文颜色、
高度、留白、宽高比，以及红章颜色、相对直径、浓淡、覆盖率和横向中心位置。
清单保留原 43 张，追加 `recognition/datadoc/analysisdata/errdata20260909.md` 中确认有红章的
118 张，包括全部 99 张红章主类及其他分类中的叠印样本，去重后保留相对路径。参考图经过共享
`load_rgb_image` 读取，仅保留数值风格，不复制原图像素，不读取 val 标注。
其中 `train_images/000351-000003.png` 未测到明显红色，只提供正文风格，红章参数使用
其余参考的中位数；终端和报告明确标记 `red_measured=false`。整批参考都无红章时停止。

正文从 `train_annotation.txt` 中选择长度接近参考宽高比的整条文字，复用 `generate.py`
的字体渲染重新绘制。按中英文估计字宽选择整条标签，覆盖单字、短数字串、中英混排和长行；
实际文字类型取决于训练语料，不复制参考图标签，不限制到旧补字脚本的 24 字。
保留内部空格、标点和全部正文。字典外字符或整条文字
没有字体覆盖时在写入前报错。字体默认偏重宋体常规字重。

红章用程序绘制圆环、弧形章字、五角星，整章旋转并轻微压扁，叠加局部印泥压力和模糊。
默认按以下 10 种场景均分样本（数量之差不超过 1），可用 `--scenes` 选择子集：

| 场景 | 内容 |
| --- | --- |
| `reference` | 参考覆盖率和红章横向中心，随机上、中、下截面 |
| `top` / `bottom` | 上弧章字 / 下弧与底部章字 |
| `star` | 星形中心对准正文 |
| `left_edge` / `right_edge` | 红色中心限制在行首 / 行尾 35% 区域，覆盖边缘文字、标点 |
| `rim` | 仅圆环残弧，低覆盖率，模拟额外点、斜线等局部干扰 |
| `faded` | 24%～42% 透明度的淡章 |
| `worn` | 局部压力不均加 12%～30% 颗粒缺墨，正文像素不参与磨损 |
| `dense` | 加粗章线、增加章字、82%～98% 透明度及较高局部遮盖 |

以乘色混合为主，保留黑字笔画；浓章增加少量普通混合模拟扫描遮盖。
位置搜索按合成后的实际红色覆盖率打分，同时检查章层与正文墨迹交叠；每图至少 8 个可见红色像素。
残弧允许 0.2% 正文交叠，其他场景至少 1%；短框、浓章允许较高覆盖率。
随机搜索失败时有限次将章墨迹对准正文，实际截面和 `fallback` 记入报告；仍不可见则带路径报错。
短框的兜底可能改变指定截面，按实际 `section` 复核。
**标注只包含正文，章中文字视作干扰，不加入标签**；这与原普通合成图无额外文字的用途不同。

```bash
# 默认生成 1000 张，文件名日期自动取系统当天
.venv/bin/python recognition/utils/synthesis/generate_red_stamp.py

# 小批量效果预览，使用独立输出路径
.venv/bin/python recognition/utils/synthesis/generate_red_stamp.py --images 161 \
  --output-dir outputs/red_stamp_preview/images \
  --annotation outputs/red_stamp_preview/annotation.txt \
  --report-dir outputs/red_stamp_preview/report
```

默认输出图片为 `recognition/data/synthesis_red_YYYYMMDD_images/YYYYMMDD_00001.png`，
标注为同级 `synthesis_red_YYYYMMDD_annotation.txt`，报告位于
`recognition/utils/synthesis/synthesis_report/synthesis_red_YYYYMMDD/`，包含 `summary.json`、逐图 `samples.jsonl`
和最多 40 张抽样的 `preview.png`。记录原 train 文本行号、字体、参考风格、章的位置和正文
重叠比例、场景、实际截面、遮盖比例和兜底状态，汇总 `scenes`、`sections`、`fallback_count`，
便于复核。参考与场景分别均匀采样并打乱；图数不少于参考数时，每张参考至少使用一次。
支持 `--images`、`--references`（清单路径，条目相对于 `--data-dir`）、`--corpus`、
`--fonts`、`--stamp-text`、`--scenes`、`--seed` 及三个输出路径参数。

输出 RGB PNG 和 Tab 分隔标注，拒绝覆盖已有产物；失败保留 `.partial`，成功后才发布正式标注。
训练数据和配置不自动修改，使用时将新标注加入 `TRAIN_ANNOTATION`。本脚本是红章干扰增强，
不保证低频字符补齐配额，也不执行裁章或红色像素删除。

### 按统计补齐低频字（默认 50000 张）

`../../recognition/utils/synthesis/generate_balanced.py` 是独立的补齐入口，复用上述字体检查、
train/val 风格采样和 RGB 渲染。默认读取
`../../outputs/recognition_character_stats/character_stats.csv` 的**第一列字符和最后一列次数**，
只选中原次数严格小于 100 的字符（包含零次），新增次数为 `110 - 原次数`。
可用 `--target-count 100` 到 `120` 调整最终次数；原本已经达到 100 次的字不设补齐配额。
这里的最终次数是 **CSV 基线 + 本批新增标注中的字符出现次数**，同一图片内重复字也累加，
不是每字图片数，也不是仅训练集次数；默认 CSV 基线包含 train、val 和旧 synthesis。

```bash
# 先校验实际 CSV、字典、字体、参考图和全部 50000 条文字配额，不写文件
.venv/bin/python recognition/utils/synthesis/generate_balanced.py --dry-run

# 默认生成恰好 50000 张，年月日前缀自动取系统当天日期，无需传日期参数
.venv/bin/python recognition/utils/synthesis/generate_balanced.py

# 调整图片总数及最终字符次数
.venv/bin/python recognition/utils/synthesis/generate_balanced.py \
  --images 30000 --target-count 110
```

为同时满足图片总数和补齐次数，一张图可以包含多个目标字。先打乱全部缺口字符，
再均匀分配至 `--images` 张图；配字仅从 CSV 中已达到 100 次且 keys/当前字体支持的字中选取，
不会额外增加目标字次数。文字长度沿用参考图估计值；若容纳不下分配的目标字则扩长，
仍最多 24 字，颜色、高度与留白继续使用参考风格。图数不足以容纳全部缺口时提前报错，
不静默减少配额。每条文字须由同一个字体完整覆盖，缺字或配字不足会在写图片前报错。

日期在脚本启动时从系统本地时间读取一次，同一批跨午夜也保持一致；不提供 `--date` 参数。
以系统日期为 2026 年 9 月 8 日为例，默认输出为：

- 图片：`recognition/data/synthesis_balanced_20260908_images/synthesis_balanced_20260908_00001.png` 至 `synthesis_balanced_20260908_50000.png`。
- 标注：`recognition/data/synthesis_balanced_20260908_annotation.txt`，沿用实际加载器的 Tab 分隔格式。
- 报告：`recognition/utils/synthesis/synthesis_report/synthesis_balanced_20260908/`，包含逐字原次数/新增次数/最终次数
  `character_counts.csv`、`summary.json`、逐图 `samples.jsonl` 和 `preview.png`。

支持 `--output-dir`、`--annotation`、`--report-dir` 自定义输出；沿用空目录检查和标注原子发布，
拒绝覆盖旧产物，不自动修改训练配置或合并旧标注。后续再补齐前，应将本批标注纳入
`character_stats.py --annotations ...` 重新统计，并通过 `--stats` 指定新 CSV，避免重复使用旧基线。

## 3. 模型与张量契约

```mermaid
flowchart LR
    X[图像 B,C,32,W] --> CNN[MobileFeatureExtractor]
    CNN --> F[特征 B,256,1,W/4]
    F --> T[转为 T,B,256]
    W[img_widths B] --> LEN[真实序列长度]
    T --> PACK[PackedSequence]
    LEN --> PACK
    PACK --> GRU[长度感知单层 BiGRU]
    GRU --> FC[Linear classes]
    FC --> LS[LogSoftmax]
    LEN --> LS
    LS --> Y[输出 T,B,C]
    Y --> CTC[CTC Loss / Greedy Decode]
```

稳定契约：

- 输入图像：`float32 [B, C, 32, W]`，归一化到以 0.5 为 mean/std 的空间；
- `img_widths`：`int32/int64 [B]`，表示不含右侧背景保护区和 batch padding 的真实图片宽度；
- 输出：`float32 [T, B, num_classes]`，简称 TBC；
- 类别：0=`blank`，1=`UNK`，2 起来自 `keys.txt`；
- `keys.txt` 的内容和顺序是模型 ABI，变更字典必须训练/导出新模型版本；
- CNN 宽度下采样约为 4 倍，CTC 长度必须依据每张图片真实宽度计算。

`LabelConverter.decode` 已要求显式 `layout="TBC"` 或 `"BTC"`。部署入口会用已知 batch size 和类别数把后端输出规范为 TBC，不再用维度大小关系猜测。

每张图片右侧固定增加 32 px 背景保护区，隔离 CNN 感受野与动态 batch 边界；BiGRU 使用 `PackedSequence`，正反向都只处理各图片的有效时间步。自动测试要求同一图片在 batch=1 和混合宽度 batch 中的有效输出一致，并验证 ONNX GRU 带有 `sequence_lens` 输入。

## 4. 训练、评测、导出与推理

```mermaid
flowchart LR
    TRAIN[recognition/train.py] --> PTH[checkpoint]
    VAL[独立 val/test manifest] --> EVAL[recognition/evaluate_val.py]
    PTH --> EVAL
    PTH --> EX[recognition_export_onnx.py]
    EX --> ONNX[FP32 ONNX 动态B/W]
    PTH --> INF[recognition_inference.py]
    ONNX --> INF
    INF --> TXT[文本/耗时/错误]
```

主入口：

- `../../recognition/train.py`：训练与训练期验证；
- `../../recognition/evaluate_val.py`：独立 manifest 的 Accuracy/CER/错误样本；
- `../../recognition/inference.py`：PyTorch checkpoint 调试；
- `../../recognition_export_onnx.py`：带 `img_widths`、动态 batch/width 的 ONNX；
- `../../recognition_inference.py`：ONNX/OpenVINO/PyTorch 部署推理。
- `../../recognition_benchmark_cpu.py`：带标注的 CPU CER、整行准确率、耗时和逐图预测报告；
- `../../recognition_quantize_static.py`：共享预处理的 NNCF 静态量化，以及基于真实 CER 的精度恢复。

`train.py` 的 `TRAIN_ANNOTATION` 可配置一个标注路径或多个标注路径；当前训练集按配置顺序
合并 `train_annotation.txt`、`append_annotation.txt`、`bench1_annotation.txt`、
`bench2_annotation.txt` 和 `synthesis_annotation.txt`，各文件内的图片路径分别相对于
自身所在目录解析。

示例：

```bash
python recognition/train.py

# evaluate_val.py 当前仍使用文件顶部常量，不支持命令行参数。
# 先配置 MODEL_PATH、启用一个 VAL_ANNOTATION，并为本次评估设置 ERROR_OUTPUT_FILE。
.venv/bin/python recognition/evaluate_val.py

python recognition_export_onnx.py \
  --model_path recognition/checkpoints/<model>.pth \
  --output_path artifacts/recognition/<version>/model.onnx \
  --keys_path recognition/keys.txt \
  --img_height 32 \
  --img_width 256

python recognition_inference.py \
  --model_path artifacts/recognition/<version>/model.onnx \
  --backend auto \
  --keys_path recognition/keys.txt \
  --image_path recognition/data/test_images1 \
  --batch_size 32
```

### CPU 部署与静态量化

导出支持 `--output_type logits` 省略 LogSoftmax，仍返回 `[T,B,C]` 并将 padding
位置设为 blank。它适用于当前 greedy CTC 解码；需要概率、置信度或 CTC loss 时，
保留默认的 `log_probs`，或先对 logits 做 LogSoftmax。导出会校验动态 batch/width、
GRU 的 sequence_lens 和 PyTorch 数值一致性，并写入字典 SHA256；ONNX 推理会核对字典。
`--img_width` 仅是导出样例宽度，不限制实际文字框宽度。

推理 `--backend auto` 按 `.pth/.onnx/.xml` 选择后端。`--num_threads` 默认为 1，
可在目标机器实测 1/2/4/6；不能把逻辑 CPU 总数当成最优线程数。
ONNX 默认关闭线程空转，避免等待下一张图时抢占预处理 CPU；`--allow_spinning`
只应在完整流水线实测获益时开启。非 PyTorch 后端的张量预处理/解码固定用一个
PyTorch 线程，推理线程由后端管理，避免线程池互相竞争。
OpenVINO 使用单 stream、LATENCY 和 FP32 浮点计算提示，INT8 层仍按量化图执行。
本机 OpenVINO 2026.0 CPU 的双向 GRU 在实测中仍遍历 padding，造成大量末字丢失。
`../../recognition/utils/openvino.py` 在加载/量化前将其改为两个正向 GRU，并用
batch Gather 按各样本真实长度显式反转反向分支的输入、输出；最终沿用 blank mask。
这样保留完整 CNN 背景保护区，又使有效反向状态不受 padding 污染。该处理仅面向
当前不消费最终 hidden state、有效长度大于零的 CRNN；已修复 IR 再次加载时不重复改图。
回归测试同时要求 OpenVINO 跨 batch 一致和对齐 PyTorch，不能仅比较两个 OpenVINO 结果。
单图默认 `--num_workers 0` 避免创建进程开销；离线批处理可开启 `--sort_by_width`
减少无效 padding，输出恢复原文件排序。显示的后端耗时包含解码，总耗时还包含预处理。

```bash
.venv/bin/python recognition_benchmark_cpu.py \
  --model_path outputs/recognition/model.onnx \
  --annotation recognition/data/val_annotation.txt \
  --num_threads 4 --output outputs/recognition/evaluation.json

.venv/bin/python recognition_quantize_static.py \
  --onnx_model outputs/recognition/model.onnx \
  --calibration_dir recognition/data/train_images \
  --validation_annotation recognition/data/val_annotation.txt \
  --output_dir outputs/recognition/int8 --subset_size 300
```

量化从模型读取 RGB/灰度通道，复用 InferenceDataset 和 dynamic_padding_collate_fn，
包含透明合成、保守去章、`[-1,1]` 归一化、动态宽度及 32px 右侧保护区；禁止单独将校准
长图压到 256。排序后按固定种子抽取训练图片，保存校准名单。默认保留 GRU 和分类层
FP32；`--quantize_head` 是需另行评估的实验选项，不承诺必然提高速度或保持准确率。

`--accuracy_control --max_drop 0.001` 使用 NNCF 的 `1-CER` 指标恢复敏感层精度，
保存前再次要求 CER 增幅不超过 0.1 个百分点；必须提供真实文字标注，禁止用输出余弦
相似度替代 OCR 精度。精度控制集与校准集不能路径重叠，调参数据不能作为独立发布测试集。
`--validation_size 0`、评估脚本 `--limit 0` 表示全量；非零时按固定种子随机抽样。
未开启精度控制的量化只是生成候选模型，需要根据独立数据的 CER 与速度决定是否使用。
`--also_onnx` 可另生成 ONNX INT8，但不能与 OpenVINO 精度恢复同时启用。
该 ONNX 路径固定使用逐张量激活量化，避免 ORT CPU 的 QLinearConv 不支持逐通道
激活零点；保存后实际执行若干不同宽度输入，验证运行时兼容性。

本次指定 epoch 120 权重的全量准确率、CPU 性能、OpenVINO 修复与部署命令见
[CPU 测试报告（2026-09-09）](recognition-cpu-report-2026-09-09.md)。

初次准备数据时，应先从 `all` 使用上述划分脚本建立图片与标注一一对应的独立训练集
和验证集，并运行三份标注校对；划分完成前，不应将训练集评估结果作为发布精度。

## 5. 评测维度

总体 Accuracy/CER 之外，至少按以下网页场景报告：

- `menu-white-text`：菜单、按钮、有色/深色背景白字；
- `small-text-or-sparse`：小文字、大背景、文字占比低；
- `has-icon-or-arrow`：箭头、图标、角标邻接；
- `long-text`：长文本、空格和中英文标点；
- `light-dark-theme`：浅色/深色主题；
- `scale-dpr`：浏览器缩放和不同设备像素比；
- `unknown-char`：字典外字符和规范化策略。

门禁必须包含：跨 split 重复为 0、冲突 label 为 0、batch 一致率 100%、PyTorch/ONNX 文本一致率 100%、关键分组不低于已批准基线。阈值和基线报告都要版本化。

## 6. 已知问题与演进边界

- `train.py` 仍以脚本常量为主，需迁移为可序列化配置与 run manifest。
- 训练数据与标注统一使用 `.png`；路径或扩展名不一致会直接报错，避免自动回退掩盖数据问题或以 dummy tensor 静默污染训练。
- recognition 不再做二次裁剪；detection 框质量需要在 detection 环节单独评估。
- 输入高度目前固定 32；48 高度只能作为同数据、同训练预算的消融实验，不能直接替换现有权重。
- FP32 ONNX/OpenVINO 是准确率优先的默认优化方向；INT8 只有通过带标注精度和目标 CPU 速度对照后才可替代。
- `model_resnet50.py` 和历史说明文档属于实验资产，未通过同一门禁前不替代当前基线。

2026-09-07 对 dataset 历史改动、红章增强和训练机制的复核见
[专项分析](recognition-analysis-2026-09-07.md)。报告区分本次像素/机制测量与尚待完成的
训练精度对照，没有改变上述训练或推理契约。
