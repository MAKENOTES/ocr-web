# Recognition 数据集准备与核对流程

本文档说明 Recognition 数据从全量标注到开始训练前的推荐处理顺序。所有命令均在
仓库根目录执行，并优先使用仓库内的 `.venv` Python 环境。

Recognition 标注格式为：

```text
relative/path.png, "识别文字"
```

图片路径相对于标注文件所在目录，必须与磁盘上的 `.png` 文件精确对应。文字中可以
包含逗号；读取脚本只使用每行第一个逗号分隔图片路径和文字。

## 推荐顺序

| 顺序 | 脚本 | 作用 | 是否修改数据 |
| --- | --- | --- | --- |
| 1 | `recognition/split_dataset.py` | 校验全量图片和标注，并按 80/20 生成 train/val | 默认只预演；`--apply` 后复制图片并写标注 |
| 2 | `recognition/check_dataset_annotations.py` | 校对 `all = train + val`、train/val 不重叠及同名图片文字一致 | 否 |
| 3 | `recognition/compare_train_val_same_text.py` | 找出 train 与 val 中完全相同的文字，辅助检查数据泄漏或高频重复标签 | 只更新报告文件 |
| 4 | `recognition/utils/dictionary.py` | 只从 train 标注生成新模型使用的候选字符字典 | 写入 `keys-new.txt` |
| 5 | `recognition/check_val_annotation_keys.py` | 找出 val 中包含字典外字符的图片 | 只更新报告文件 |

必须先保证数据划分正确，再生成字典。不要为了让第 5 步通过而直接把 val 独有字符追加
到字典；应调整数据划分，使验证集中的每个字符在训练集中都有训练样本，然后重新生成
候选字典并再次核对。

## 1. 预演并生成训练集、验证集

脚本：`recognition/split_dataset.py`

默认输入：

- `recognition/data/all_images/`
- `recognition/data/all_annotation.txt`

默认输出：

- `recognition/data/train_images/`
- `recognition/data/train_annotation.txt`
- `recognition/data/val_images/`
- `recognition/data/val_annotation.txt`

先运行预演。预演会核对图片与标注是否一一对应、路径是否合法、是否存在重复或缺失，
但不会复制或写入任何数据：

```bash
.venv/bin/python recognition/split_dataset.py
```

确认数量和验证集比例正确后再执行实际划分：

```bash
.venv/bin/python recognition/split_dataset.py --apply
```

默认验证集比例为 20%，也可以在预演和正式执行时指定相同比例：

```bash
.venv/bin/python recognition/split_dataset.py --val-ratio 0.20
.venv/bin/python recognition/split_dataset.py --val-ratio 0.20 --apply
```

注意：

- `all_images/` 和 `all_annotation.txt` 不会被修改。
- 为避免覆盖手工调整的数据，只要 train 或 val 输出不为空，脚本就会停止。
- 如果已经完成划分，不要为了重复运行脚本而删除现有数据；除非明确需要重新划分并已
  备份、核对目标路径。

## 2. 校对 all、train、val 三份标注

脚本：`recognition/check_dataset_annotations.py`

```bash
.venv/bin/python recognition/check_dataset_annotations.py
```

它按图片文件名核对：

- `all` 是否恰好等于 `train` 与 `val` 的并集；
- 同一图片是否同时出现在 train 和 val；
- train/val 是否出现 all 中不存在的图片名；
- all 与所属 split 中同名图片的识别文字是否一致。

每类问题默认最多显示 20 条。需要显示全部问题时：

```bash
.venv/bin/python recognition/check_dataset_annotations.py --max-details 0
```

校对通过返回 `0`，发现集合或文字不一致返回 `1`。本步骤失败时，应先修正标注划分，
不要继续生成字典或训练模型。

## 3. 检查 train 与 val 的相同文字

脚本：`recognition/compare_train_val_same_text.py`

```bash
.venv/bin/python recognition/compare_train_val_same_text.py
```

脚本比较 train 和 val 中规范化后的完整文字，只把两边都出现过的文字写入：

```text
recognition/data/train_val_same_text.txt
```

报告格式：

```text
[1] 文字: '* 共享人：'
[2] 文字: '* 分装送样单：'
```

相同文字不一定就是错误。例如页码、单位名称和通用字段可能自然重复；应重点检查长句、
罕见文本或大量连续重复内容，判断是否存在同一来源页面、近重复图片或数据泄漏。该脚本
只生成报告，不自动移动或删除任何图片。

如标注文件不使用默认路径，可以指定：

```bash
.venv/bin/python recognition/compare_train_val_same_text.py \
  --train-annotation <train标注路径> \
  --val-annotation <val标注路径>
```

## 4. 从训练集生成候选字符字典

脚本：`recognition/utils/dictionary.py`

```bash
.venv/bin/python recognition/utils/dictionary.py
```

默认读取：

```text
recognition/data/train_annotation.txt
```

默认写入：

```text
recognition/keys-new.txt
```

脚本使用与 `TextRecognitionDataset` 一致的文字解析规则，收集 train 中的所有唯一字符，
按 Unicode 顺序排序，并保证字典每个非空行只有一个字符。正文中的逗号不会导致文字被
截断。

也可以明确指定输入和输出：

```bash
.venv/bin/python recognition/utils/dictionary.py \
  --input recognition/data/train_annotation.txt \
  --output recognition/keys-new.txt
```

`keys.txt` 的内容和顺序属于模型 ABI。已有 checkpoint、ONNX 或 OpenVINO 模型必须继续
使用训练该模型时的原始 `keys.txt`。只有准备从头训练新模型，并完成第 5 步核对后，才能
决定是否用 `keys-new.txt` 作为新模型字典；替换字典后必须重新训练和导出模型。

## 5. 核对验证集是否包含字典外字符

脚本：`recognition/check_val_annotation_keys.py`

核对当前正式字典：

```bash
.venv/bin/python recognition/check_val_annotation_keys.py
```

在训练新模型前，应优先核对刚生成的候选字典：

```bash
.venv/bin/python recognition/check_val_annotation_keys.py \
  --keys recognition/keys-new.txt
```

完整报告固定写入：

```text
recognition/data/val_unknown_keys.txt
```

报告会列出验证图片路径、完整文字、标注行号以及该图片中未被字典收录的字符，例如：

```text
[1] 图片: val_images/000011-000017.png (第 78 行)
  文字: '积>3000㎡， 另外还有两栋'
  字典未收录: '㎡'
```

退出码含义：

- `0`：val 中的全部字符都存在于指定字典；
- `1`：发现包含字典外字符的验证图片，报告已正常生成；
- `2`：标注、字典或文件格式错误，核对未正常完成。

如果返回 `1`，推荐处理方式是把含有这些字符的合适样本分配到 train；同一字符有多个
独立样本时，可以同时保留样本在 val。然后依次重新运行第 2～5 步。仅向字典增加字符、
却不给训练集增加对应样本，模型仍然学不会这些字符。

确认要把这些样本从验证集隔离时，显式添加：

```bash
.venv/bin/python recognition/check_val_annotation_keys.py --move-unknown
```

该参数会同时执行以下操作：

- 把问题图片从 `recognition/data/val_images/` 移动到
  `recognition/data/val_unknow/`；
- 从 `recognition/data/val_annotation.txt` 删除对应标注；
- 把对应标注写入 `recognition/data/val_unknow.txt`，其中图片路径同步改为
  `val_unknow/...`；
- 保留 `recognition/data/val_unknown_keys.txt` 核对报告。

不传 `--move-unknown` 时始终只核对，不移动数据。执行转移时，`val_unknow/` 和
`val_unknow.txt` 必须不存在或为空；任一目标非空、源图片缺失或目标冲突都会在移动前
停止。中途写入失败时，脚本会恢复已移动的图片和原始验证标注。成功转移后返回 `0`。

这是隔离操作，不会修改 `all_annotation.txt`。因此转移后数据关系变为
`all = train + val + val_unknow`，原本只接受 `all = train + val` 的
`check_dataset_annotations.py` 会把隔离样本报告为未分配；应保留隔离清单，不要误删
`all` 中的原始记录。

## 训练前最短检查清单

已有 all/train/val 数据时，可以按以下顺序运行：

```bash
# 1. 核对三份标注的集合和文字
.venv/bin/python recognition/check_dataset_annotations.py

# 2. 检查 train/val 是否存在相同文字
.venv/bin/python recognition/compare_train_val_same_text.py

# 3. 从 train 生成新模型的候选字典
.venv/bin/python recognition/utils/dictionary.py

# 4. 检查 val 字符是否全部被候选字典覆盖
.venv/bin/python recognition/check_val_annotation_keys.py \
  --keys recognition/keys-new.txt
```

第 1、4 步必须通过。第 2 步的相同文字报告需要人工审阅，不能只根据数量自动判断。

## 历史工具说明

- `recognition/utils/compare_keys.py` 使用旧的正则解析标注，并同时读取 train 和 val；当前
  流程不再推荐使用。请用 `dictionary.py` 从 train 生成候选字典，再用
  `check_val_annotation_keys.py` 检查 val 覆盖情况。
- `recognition/utils/normalization_label.py` 会生成替换英文逗号后的标注文件，它不是当前
  数据核对流程的一部分。文字中的英文逗号属于合法内容，不应为了适配解析脚本而批量
  替换。

`recognition/data/` 下的数据集、核对报告、训练日志、checkpoint 和导出模型都属于本地
数据或产物，不应提交到版本库。
