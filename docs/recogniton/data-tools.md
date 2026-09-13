# Recognition 数据工具

维护入口见[数据维护](../../recognition/datadoc/dataset.md)。本页只描述脚本行为，不证明本地数据已完成校验。

## 初始训练集/验证集划分

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

## 四份标注校对

手工调整 train/val 或隔离字典外字符样本后，运行下列只读脚本校对四份标注：

```bash
.venv/bin/python recognition/utils/check_dataset_annotations.py
```

脚本仅按不含目录的图片文件名判断，例如 `du20260902-002391.png`。它要求
`all = train + val + val_unknown`、三个子集两两没有同名图片，并逐项比较 all 与所属
split 中同名图片后面的识别文字。校对通过返回 0；缺失、多余、跨 split 重复或文字不同
会列出具体文件名并返回 1。每类默认最多显示 20 条详情，传入 `--max-details 0` 可显示全部。

## 字符频次与数据集覆盖率

```bash
# 默认统计 train、val、synthesis1、synthesis2 四份标注，按合计占比倒序打印全部字符，并将完整明细写入 CSV
.venv/bin/python recognition/utils/character_stats.py

# 终端只显示“王”的次数和占比，CSV 仍保留 keys 中的全部字符
.venv/bin/python recognition/utils/character_stats.py --chars 王

# 自定义统计文件；此例只统计 train 和两份合成数据，排除 val
.venv/bin/python recognition/utils/character_stats.py \
  --annotations recognition/data/train_annotation.txt \
  recognition/data/synthesis1_annotation.txt recognition/data/synthesis2_annotation.txt \
  --output-dir outputs/recognition_training_character_stats
```

只生成 `../../outputs/recognition_character_stats/character_stats.csv`，不再生成 JSON。
CSV 和终端均按未舍入的合计占比倒序输出，同频保持 keys 原顺序，零出现字符也保留；
只列 keys 中的字符，不修改 keys。占比列在前，次数列统一放在最后。
默认列顺序为 `字符、train_占比、val_占比、synthesis1_占比、synthesis2_占比、合计_占比、
train_次数、val_次数、synthesis1_次数、synthesis2_次数、合计_次数`。
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

## 按字符统计移除补充数据

`../../recognition/utils/remove_characters_by_stats.py` 使用上述默认 CSV，并核对当前
`keys.txt` 和 train、val、synthesis1、synthesis2 四份标注中的实际字符次数。
CSV 缺列、字符集合或次数不一致时停止，需先重新运行 `character_stats.py`。

两种方式均要求字符在 train 和 val 中出现次数为 0：

- `--mode 1`：synthesis1 与 synthesis2 的合计出现次数严格小于 10，包含零次字符。
- `--mode 2`：仅在 synthesis1、synthesis2 中的一份出现，次数不限；两份都有出现的字符不满足此条件。
- `--mode both`：两种方式取并集，也是默认方式；包含两份合成数据均为零次的字符，但不包含两份均有出现且合计至少 10 次的字符。仅根据本次 CSV 筛选一次，不递归清理。

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
只更新 synthesis1、synthesis2 标注以及 keys 中命中的字符行，保留其余行内容、换行和
字典相对顺序。图片路径按标注文件所在目录解析并去重；待删图片缺失、路径越出数据目录、
通过符号链接访问，或仍被这四份文件中任一保留标注引用时，在写入前报错。
train、val 标注不修改，不扫描这四份标注之外的图片引用。

只有传入 `--apply` 才写入文件。执行前将所有待修改的原始标注、keys、待删图片及输入
CSV 备份到 `outputs/recognition_character_removal/removal-*/`；`manifest.json` 记录
原路径、备份路径及删除明细。单个文字文件使用原子替换，多文件操作不构成事务，若中断或
磁盘错误导致部分完成，可按 manifest 恢复。预览不创建备份或任何输出文件。
可用 `--csv`、`--keys`、`--data-dir`、`--backup-dir` 指定路径。
删除后需重新生成字符统计。删除整条样本会减少其他字符的次数、改变其数据集分布，
使本轮未命中的字符在下一轮满足条件；此时需基于新 CSV 再次预览、执行。
变更 keys 会改变模型 ABI，必须重新训练、导出并验证模型。

## 移除 train/val 已覆盖字符的合成样本

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

默认统计和清理范围仅为这四份文件，不自动读取 `train.py` 的九份训练输入。
删除字典字符前必须额外核对 synthesis3 及各专项标注的字符与图片引用；否则可能留下训练字典外标签。
`check_dataset_annotations.py` 要求 all/train/val/val_unknown 四份清单仍存在；它不是当前九份训练输入的通用校验器。
