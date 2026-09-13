# PDF 页面与网页表格文字检测

本包直接检测完整页面图片中的水平文字框，不直接读取 PDF，不恢复表格行列结构。
模型、数据处理、后处理、配置与权重均独立于 `detection`。
详细接口见[大图检测架构](../docs/detection/larger-architecture.md)。

## 训练

先准备 `detection_larger/data/train_images`、`train_labels`、`val_images`、`val_labels`。
支持 JPG/JPEG/PNG/TIFF/BMP，扩展名不区分大小写。训练标签默认
`x_min y_min x_max y_max [文本]`，支持浮点数；`###` 是忽略区域，空文件是无文字样本。

```bash
.venv/bin/python detection_larger/train.py
# 初始化权重，输出到新的空目录；不是完整续训
.venv/bin/python detection_larger/train.py --weights /path/to/model.pth --output detection_larger/checkpoints/new_run
```

| 顶部配置 | 当前值及含义 |
| --- | --- |
| INPUT_SIZE / BATCH_SIZE | 640×640 像素 / 每批 4 张整图 |
| EPOCHS | 最多 620 轮 |
| 学习率 | Adam 构造值 1e-4，OneCycleLR 峰值 3e-4 |
| 验证 | 前 50 轮每 10 轮验证，第 50 轮起逐轮验证 |
| 早停 | 连续 100 次验证未改善超过 1e-5 |
| 保存 | 每 5 轮及末轮常规保存，保留最近 5 份；另保留最低验证损失 Top-3 |
| PRETRAINED | True，无初始权重时使用 ImageNet 参数；无本地缓存会下载 |
| NUM_WORKERS / DEVICE | 4 个加载子进程 / 优先 CUDA，否则 CPU |

输入也可设为 `(高, 宽)`，各边至少 64 且为 32 的倍数。独立验证集需事先检查重复样本；
入口只检查目录路径不同。[早期 1024、batch 2 基线记录](../docs/detection/larger-baseline-history.md)不代表当前默认值，真实精度需另行评估。

## 推理与导出

这三个入口均使用顶部常量，不提供命令行参数。路径相对于运行目录，以下在仓库根目录执行：

```bash
.venv/bin/python detection_larger/inference.py
.venv/bin/python detection_larger_export_onnx.py
.venv/bin/python detection_larger_inference_onnx.py
```

包内推理 `DEVICE=auto`，根目录多后端入口默认 `BACKEND=openvino`、CPU、预热 3 次。
两个推理入口均默认关闭裁片；需要时将各自的 `SAVE_CROPS` 改为 `True`。
预览与标签保存在输出目录的 `images/`、`labels/`，可选裁片平铺为 `<stem>-00001.png` 起。
重复运行覆盖对应结果，开启裁片时清理该原图多余旧序号；关闭时不改裁片目录。

导出默认 `INPUT_SIZE=640`、`BENCHMARK_ENABLED=True`，除数值校验外还执行 CPU 测速。
推理与导出尺寸常量必须匹配，目标 ONNX 已存在时拒绝覆盖。参数契约和测速口径见架构页。

## 评估与测试

```bash
.venv/bin/python detection_larger/components/eval_metrics.py --model /path/to/document-model.pth
.venv/bin/python -m pytest tests/test_detection_larger.py tests/test_detection_larger_onnx.py tests/test_detection_isolation.py -q
```

单元测试不代表当前本地权重精度；待修复的测试开关及数据校验边界见[待办](../docs/code-doc-todolist.md)。
