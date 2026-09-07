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
    RGB --> RESIZE[完整 detection 框固定高度32 + 等比宽度]
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

- `recognition/components/dataset.py`：图片路径校验、PNG 透明通道处理、缩放、增强、桶采样和动态 padding；
- `recognition/utils/checkpoint.py`：checkpoint 解包以及旧灰度/新 RGB 模型通道识别；
- `recognition/components/converter.py`：字典编码和显式 TBC/BTC 的 CTC greedy decode。

训练、验证、PyTorch 推理和多后端推理复用同一套不裁剪基础预处理。训练可额外使用随机增强；验证和测试禁止随机增强。

### 初始训练集/验证集划分

当 `all_images/` 和 `all_annotation.txt` 保存全量数据时，使用
`recognition/split_dataset.py` 做确定性的 80/20 均匀系统抽样。脚本按标注顺序每隔约
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
# 默认统计五份标注，按合计占比倒序打印全部字符，并将完整明细写入 CSV
.venv/bin/python recognition/utils/character_stats.py

# 终端只显示“王”的次数和占比，CSV 仍保留 keys 中的全部字符
.venv/bin/python recognition/utils/character_stats.py --chars 王

# 按 train.py 的四份训练标注统计，排除 val
.venv/bin/python recognition/utils/character_stats.py \
  --annotations recognition/data/train_annotation.txt \
  recognition/data/append_annotation.txt recognition/data/bench1_annotation.txt \
  recognition/data/bench2_annotation.txt \
  --output-dir outputs/recognition_training_character_stats
```

只生成 `outputs/recognition_character_stats/character_stats.csv`，不再生成 JSON。
CSV 和终端均按未舍入的合计占比倒序输出，同频保持 keys 原顺序，零出现字符也保留；
只列 keys 中的字符，不修改 keys。占比列在前，次数列统一放在最后。
列顺序为 `字符、train_占比、val_占比、bench2_占比、append_占比、bench1_占比、合计_占比、
train_次数、val_次数、bench2_次数、append_次数、bench1_次数、合计_次数`。
`次数` 包括同一标签内重复出现的字符；`占比` 为该字符次数除以对应文件全部文字字符数
（含字典外字符），直接显示百分数，保留五位小数，例如 `0.00000%`。出现 80 次、总共
1,000 个字符时，占比为 `8.00000%`。合计占比为五份文件的该字总次数除以五份文字总字符数，不是各占比平均。
合计累加本次传入的全部标注记录，不去重，默认包含 val，不能当作训练集总量。
终端同时提供各文件及合计的文字总字符数，方便核对分母。
默认运行也会在终端逐行打印全部字符；`--chars` 只限制终端展示，不过滤输出文件。
`outputs/` 在 `.gitignore` 中，编辑器全局搜索可能排除它；直接打开文件搜索，或显式运行
`rg -n 王 outputs/recognition_character_stats/character_stats.csv`。
脚本复用训练的 Tab 解析和首尾空白处理规则，不进行标点、Unicode 或转义归一化；
空标签按训练规则跳过并计数，格式错误直接报告具体文件与行号。此脚本不检查图片。

### 按字符统计移除补充数据

`recognition/utils/remove_characters_by_stats.py` 使用上述默认 CSV，并核对当前
`keys.txt` 和 train、val、append、bench1、bench2 五份标注中的实际字符次数。
CSV 缺列、字符集合或次数不一致时停止，需先重新运行 `character_stats.py`。

两种方式均要求字符在 train 和 val 中出现次数为 0：

- `--mode 1`：append、bench1、bench2 出现次数之和严格小于 10，包含零次字符。
- `--mode 2`：只在 append、bench1、bench2 中恰好一份出现，次数不限。
- `--mode both`：两种方式取并集，也是默认方式；仅根据本次 CSV 筛选一次，不递归清理。

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
只更新 append、bench1、bench2 标注以及 keys 中命中的字符行，保留其余行内容、换行和
字典相对顺序。图片路径按标注文件所在目录解析并去重；待删图片缺失、路径越出数据目录、
通过符号链接访问，或仍被这五份文件中任一保留标注引用时，在写入前报错。
train、val 标注不修改，不扫描五份标注之外的图片引用。

只有传入 `--apply` 才写入文件。执行前将所有待修改的原始标注、keys、待删图片及输入
CSV 备份到 `outputs/recognition_character_removal/removal-*/`；`manifest.json` 记录
原路径、备份路径及删除明细。单个文字文件使用原子替换，多文件操作不构成事务，若中断或
磁盘错误导致部分完成，可按 manifest 恢复。预览不创建备份或任何输出文件。
可用 `--csv`、`--keys`、`--data-dir`、`--backup-dir` 指定路径。
删除后需重新生成字符统计。删除整条样本会减少其他字符的次数、改变其数据集分布，
使本轮未命中的字符在下一轮满足条件；此时需基于新 CSV 再次预览、执行。
变更 keys 会改变模型 ABI，必须重新训练、导出并验证模型。

### 补字合成数据

`recognition/utils/synthesis/generate.py` 使用
[oh-my-ocr/text_renderer](https://github.com/oh-my-ocr/text_renderer) 的
`FontText` 和 `draw_text_on_bg` 字形渲染接口。针对 `recognition/keys-append.txt`
每个字默认生成 100 张；配字仅取 `recognition/keys.txt` 中**非目标字**且当前字体支持的字符，
每张仅包含一个目标字，保证每个目标字的图片数和出现次数都严格等于 100。
字符文件要求每行一个可见字符、无重复；目标不在 keys 或没有字体支持时直接报错。
不修改 keys、现有训练/验证数据或训练配置。

```bash
# 可选工具依赖：锁定上游提交，只装绘字模块，保留现有训练依赖版本
.venv/bin/python -m pip install --no-deps -r recognition/utils/synthesis/requirements.txt

# 默认生成到 recognition/data/synthesis_images
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

- `recognition/data/synthesis_images/uXXXXXX_NNN.png`：Unicode 码点和样本序号命名。
- `recognition/data/synthesis_annotation.txt`：复用当前加载器的 `relative/path.png<TAB>文字`
  格式；当前代码已使用 Tab，AGENTS.md 中逗号加引号的描述是旧格式。
- `recognition/data/synthesis_report/summary.json`：逐字计数、种子、字体计数、输入校验和。
- 同目录 `samples.jsonl`、`preview.png`：每张图的目标字/字体/参考风格及抽样预览。

图片和报告目录必须为空，标注必须不存在；失败保留已生成图片和 `.partial` 标注供排查，
不会覆盖或自动清理旧产物。成功后才原子发布正式标注。上述数据均被 Git 忽略。
使用合成数据训练时，将 `recognition/data/synthesis_annotation.txt` 加入
`recognition/train.py` 的 `TRAIN_ANNOTATION` 列表，验证集继续使用原来的 val 标注。

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

- `recognition/train.py`：训练与训练期验证；
- `recognition/evaluate_val.py`：独立 manifest 的 Accuracy/CER/错误样本；
- `recognition/inference.py`：PyTorch checkpoint 调试；
- `recognition_export_onnx.py`：带 `img_widths`、动态 batch/width 的 ONNX；
- `recognition_inference.py`：ONNX/OpenVINO/PyTorch 部署推理。

`train.py` 的 `TRAIN_ANNOTATION` 可配置一个标注路径或多个标注路径；当前训练集按配置顺序
合并 `train_annotation.txt`、`append_annotation.txt`、`bench1_annotation.txt` 和
`bench2_annotation.txt`，各文件内的图片路径分别相对于自身所在目录解析。

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
- `recognition_quantize_static.py` 当前不进入发布路径。FP32 ONNX/OpenVINO 是默认优化方向。
- `model_resnet50.py` 和历史说明文档属于实验资产，未通过同一门禁前不替代当前基线。

2026-09-07 对 epoch 471 权重、红章/小字清单、字符频次和训练日志的实测见
[专项分析](recognition-analysis-2026-09-07.md)。该报告中的输入通道和边缘探针仅用于诊断，
没有改变上述训练或推理契约。
