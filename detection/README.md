/home/weiqiang/AI/ocr-majorbio/
|-- data/                  # 存放你的数据集
|   |-- train_images/      # 训练图片
|   |-- train_labels/      # 训练标签 (例如, .txt 文件)
|   |-- val_images/        # 验证图片
|   |-- val_labels/        # 验证标签
|-- models/                # 存放模型定义脚本
|   |-- east_model.py      # 模型架构
|-- utils/                 # 工具脚本
|   |-- dataset.py         # 自定义Dataset类，用于加载数据
|   |-- transforms.py      # 数据预处理和增强
|-- train.py               # 训练脚本
|-- eval.py                # 评估脚本
|-- requirements.txt       # 项目依赖

## 稳定划分训练集和验证集

`split_dataset.py` 默认从 `data/train_images` 和 `data/train_labels` 中移动
20% 的图片/标签对到 `data/val_images` 和 `data/val_labels`：

```bash
python detection/split_dataset.py --dry-run  # 只检查，不修改文件
python detection/split_dataset.py            # 正式划分
```

划分结果保存在 `detection/data/split_manifest.csv`。CSV 是后续划分的依据：
已有样本的 `train`/`val` 归属不会改变，脚本只为新增样本分组，并让总体验证集比例
尽量保持为 20%。图片与标签按图片主文件名配对，支持
`gt_<图片名>.txt` 和 `<图片名>.txt`；缺少标签或标签名有歧义时脚本会停止。

下一批数据位于 `all_images` 和 `all_labels` 时，可配置来源目录：

```bash
python detection/split_dataset.py \
  --source-images detection/data/all_images \
  --source-labels detection/data/all_labels
```

其他可配置参数可通过 `python detection/split_dataset.py --help` 查看，包括输出目录、
CSV 路径、验证集比例、随机种子，以及 `--mode move|copy`。默认使用 `move`，避免训练集
和验证集包含相同样本。
