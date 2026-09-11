# PDF 与网页表格文字检测

以 `/home/weiqiang/AI/ocr-majorbio/detection/train_resnet.py` 为训练基线，直接读取已经准备好的图片和标签目录。数据不重新划分、不自动排除样本、不复制或移动文件。

## 直接训练

```bash
.venv/bin/python detection_larger/train_resnet.py
```

数据路径、输入尺寸、batch、轮数、学习率、加载进程数和设备统一在训练入口顶部常量配置：

| 配置 | 默认路径 | 当前图片数 |
| --- | --- | ---: |
| TRAIN_IMAGE_DIR | detection_larger/data/train_images | 593：465 JPG＋128 PNG |
| TRAIN_LABEL_DIR | detection_larger/data/train_labels | 593 份标签 |
| VAL_IMAGE_DIR | detection_larger/data/val_images | 189：157 JPG＋32 PNG |
| VAL_LABEL_DIR | detection_larger/data/val_labels | 189 份标签 |

训练命令行仅保留 `--weights`（初始权重）和 `--output`（本次检查点空目录）。例如：

```bash
.venv/bin/python detection_larger/train_resnet.py \
  --weights /path/to/model.pth --output detection_larger/checkpoints/new_run
```

图片同时加载 JPG/JPEG/PNG（扩展名不区分大小写），也支持 TIFF/BMP。标签按 stem 配对，支持 `gt_<stem>.txt` 或 `<stem>.txt`，歧义或缺失标签报错。训练固定采用本批数据及 majorbio 使用的 bbox4 格式：`x_min y_min x_max y_max [文本]`。保留浮点坐标，避免网页小字因取整损失边界；`###` 区域忽略，空标签表示确认无文字。

## 相对 majorbio 的必要调整及原因

| 项目 | 保留或调整 | 原因 |
| --- | --- | --- |
| 网络、真值、损失 | 原 ResNet18 EAST、r=0.35 收缩、log1p 几何、Dice＋3×Smooth L1 | 已验证有效；当前复用的本仓库三个文件与 majorbio 内容一致 |
| 优化及训练循环 | Adam、OneCycleLR 峰值 3e-4 / pct_start=0.05、梯度裁剪 5、620 轮、早停与 Top-3 | 保持 PDF 基线训练方法 |
| JPG 与 PNG | 同时枚举 | majorbio 先找 JPG，仅无 JPG 时找 PNG，会漏掉混合目录里的网页图 |
| 透明 PNG | 加载时合成可见背景并转 RGB | 防止透明背景直接转 RGB 后黑字消失；当前数据全部为 RGB |
| 输入尺寸 | 默认 1024 方形、batch 2；整图等比缩放 | 表格中位字高：640 输入约 4.6px，1024 约 7.4px，1280 约 9.2px；网络输出步长为 4 |
| 居中补边 | 保留原居中布局，补边改为边缘背景色 | 网页表格有白底、浅灰和选中行背景，固定灰色会制造额外边界 |
| 颜色扰动 | brightness=0.12、contrast=0.15、saturation=0.10、hue=0.02 | majorbio 的 0.3/0.3/0.3/0.1 对浅蓝链接、灰字和彩色状态可能过强 |
| 几何增强 | 不启用随机水平裁剪及先缩放再等比缩放 | 保留完整 PDF 长行和单元格文字；原随机缩放随后被固定尺寸缩放抵消 |
| 保存 | 独立输出目录，Top-3＋最近 5 份，每 5 轮常规保存 | 避免混入原 PDF 权重，减少磁盘占用；网络参数格式不变，外层附输入配置 |

不默认切片、不改变表格框的粒度、不将网格线当文字。先用 1024 训练并检查验证集小字召回；1280 的输入像素量多约 56%，目前没有精度对比证明必须使用它。若 1024 的漏检集中在细小文字，再保持数据划分不变，设置 `INPUT_SIZE = 1280` 做对照。不能仅凭缩放字高就保证 1024 足够或 1280 更好。

默认前 50 轮每 10 轮验证，第 50 轮起逐轮验证；早停耐心 100 次验证，min_delta=1e-5。默认 ImageNet 预训练，无缓存需要下载；离线检查设置 `PRETRAINED = False`；轻量 CPU 检查可设置 `DEVICE = "cpu"`、`NUM_WORKERS = 0`、`LOG_GRAPH = False`。`--weights /path/to/model.pth` 仅初始化原 EAST 权重，优化器和轮次重新开始。检查点不含完整续训状态。

## 推理、评估与导出

```bash
.venv/bin/python detection_larger/inference.py \
  --model detection_larger/checkpoints/<run>/<model>.pth \
  --input detection_larger/data/val_images \
  --output detection_larger/outputs/val_v1 --save-crops

.venv/bin/python detection_larger/components/eval_metrics.py \
  --model detection_larger/checkpoints/<run>/<model>.pth \
  --image-dir detection_larger/data/val_images \
  --label-dir detection_larger/data/val_labels

.venv/bin/python detection_larger_export_onnx.py \
  --checkpoint detection_larger/checkpoints/<run>/<model>.pth \
  --output detection_larger/artifacts/baseline.onnx
```

训练、推理共用整图预处理，推理一次前向后恢复原图坐标；模型记录自身输入尺寸。评估报告总体及 JPG/PNG 等格式分组的 precision/recall/F1，不将扩展名当作通用业务分类。可单独指定 PDF 或网页验证目录进行来源评估。

完整尺寸统计及验证记录见[核对报告](../docs/detection/detection-larger-assessment.md)。
