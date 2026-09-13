# PDF 与网页表格文字检测

以 `/home/weiqiang/AI/ocr-majorbio/detection/train_resnet.py` 为训练基线，直接读取已经准备好的图片和标签目录。数据不重新划分、不自动排除样本、不复制或移动文件。

模型、图像变换、真值、损失及连通域后处理均在本目录内独立维护，不导入 `detection` 或根目录的小图检测入口；两套检测的数据、权重和配置分别管理。

## 直接训练

```bash
.venv/bin/python detection_larger/train.py
```

训练输入尺寸修改 `detection_larger/train.py` 的 `INPUT_SIZE`，训练入口显式将尺寸传给预处理和检查点保存函数，推理尺寸独立修改 `detection_larger/inference.py` 顶部的 `INPUT_SIZE`（默认 640，即 640×640），不从检查点配置取值，也不导入训练入口。支持整数或 `(高, 宽)`，各边至少 64 且为 32 的倍数；预处理和预热使用同一尺寸。ONNX/OpenVINO 的导出尺寸必须与该常量一致，否则加载时报错；推理入口不保存或读取 `self.config`，ONNX/OpenVINO 直接检查模型实际输入形状；加载辅助函数仍校验模型元数据格式。数据路径、batch、轮数、学习率、加载进程数和设备在训练入口顶部常量配置：

| 配置 | 默认路径 | 当前图片数 |
| --- | --- | ---: |
| TRAIN_IMAGE_DIR | detection_larger/data/train_images | 593：465 JPG＋128 PNG |
| TRAIN_LABEL_DIR | detection_larger/data/train_labels | 593 份标签 |
| VAL_IMAGE_DIR | detection_larger/data/val_images | 189：157 JPG＋32 PNG |
| VAL_LABEL_DIR | detection_larger/data/val_labels | 189 份标签 |

训练命令行仅保留 `--weights`（初始权重）和 `--output`（本次检查点空目录）。例如：

```bash
.venv/bin/python detection_larger/train.py \
  --weights /path/to/model.pth --output detection_larger/checkpoints/new_run
```

图片同时加载 JPG/JPEG/PNG（扩展名不区分大小写），也支持 TIFF/BMP。标签按 stem 配对，支持 `gt_<stem>.txt` 或 `<stem>.txt`，歧义或缺失标签报错。训练固定采用本批数据及 majorbio 使用的 bbox4 格式：`x_min y_min x_max y_max [文本]`。保留浮点坐标，避免网页小字因取整损失边界；`###` 区域忽略，空标签表示确认无文字。

## 相对 majorbio 的必要调整及原因

| 项目 | 保留或调整 | 原因 |
| --- | --- | --- |
| 网络、真值、损失 | 原 ResNet18 EAST、r=0.35 收缩、log1p 几何、Dice＋3×Smooth L1 | 独立维护网络、真值和损失实现，固定数值回归测试验证迁移前后行为一致 |
| 优化及训练循环 | Adam、OneCycleLR 峰值 3e-4 / pct_start=0.05、梯度裁剪 5、620 轮、早停与 Top-3 | 保持 PDF 基线训练方法 |
| JPG 与 PNG | 同时枚举 | majorbio 先找 JPG，仅无 JPG 时找 PNG，会漏掉混合目录里的网页图 |
| 透明 PNG | 加载时合成可见背景并转 RGB | 防止透明背景直接转 RGB 后黑字消失；当前数据全部为 RGB |
| 输入尺寸 | 默认 1024 方形、batch 2；整图等比缩放 | 表格中位字高：640 输入约 4.6px，1024 约 7.4px，1280 约 9.2px；网络输出步长为 4 |
| 居中补边 | 保留原居中布局，补边改为边缘背景色 | 网页表格有白底、浅灰和选中行背景，固定灰色会制造额外边界 |
| 颜色扰动 | brightness=0.12、contrast=0.15、saturation=0.10、hue=0.02 | majorbio 的 0.3/0.3/0.3/0.1 对浅蓝链接、灰字和彩色状态可能过强 |
| 几何增强 | 不启用随机水平裁剪及先缩放再等比缩放 | 保留完整 PDF 长行和单元格文字；原随机缩放随后被固定尺寸缩放抵消 |
| 保存 | 独立输出目录，Top-3＋最近 5 份，每 5 轮常规保存 | 避免混入原 PDF 权重，减少磁盘占用；网络参数格式不变，外层附输入配置 |

不默认切片、不改变表格框的粒度、不将网格线当文字。先用 1024 训练并检查验证集小字召回；1280 的输入像素量多约 56%，目前没有精度对比证明必须使用它。若 1024 的漏检集中在细小文字，再保持数据划分不变，在 `detection_larger/train.py` 设置 `INPUT_SIZE = 1280` 做对照。不能仅凭缩放字高就保证 1024 足够或 1280 更好。

默认前 50 轮每 10 轮验证，第 50 轮起逐轮验证；早停耐心 100 次验证，min_delta=1e-5。默认 ImageNet 预训练，无缓存需要下载；离线检查设置 `PRETRAINED = False`；轻量 CPU 检查可设置 `DEVICE = "cpu"`、`NUM_WORKERS = 0`、`LOG_GRAPH = False`。`--weights /path/to/model.pth` 仅初始化原 EAST 权重，优化器和轮次重新开始。检查点不含完整续训状态。

## 推理、评估与导出

推理不使用命令行参数，先修改 `inference.py` 顶部常量：`MODEL_PATH`（文档专用 `.pth` 或 `.onnx`）、`IMAGE_PATH`（图片或目录）、`OUTPUT_DIR`（允许覆盖同名结果的输出目录）、`DEVICE`、`SCORE_THRESHOLD`、`NMS_IOU_THRESHOLD`、`SAVE_CROPS` 和 `CROP_DIR`。路径相对于运行目录，建议在仓库根目录执行。默认使用 `detection_larger/data/test_images` 图片目录，`DEVICE = "auto"` 让 PyTorch 优先使用 CUDA，无可用 GPU 时回退到 CPU；也可指定 `"cuda:0"` 或 `"cpu"`。ONNX 在 `auto` 下使用 CPU，也可显式设置 `cpu`。需要保存文字裁片时设置 `SAVE_CROPS = True`，裁片平铺在 `CROP_DIR`（默认 `detection_larger/data/crops/test_images`）下，不再按原图建子目录；文件名为 `<原图 stem>-<5 位序号>.png`，序号从 1 开始，例如原图 `du20260910-000001.png` 的首个裁片是 `du20260910-000001-00001.png`。裁片名和标签都以原图 stem 为前缀，所以输入图片的 stem 必须唯一，同名不同扩展名会直接报错。重复运行覆盖本次输入对应的图片、标签及裁片，并清理该原图超出本次框数的旧序号裁片，保留其他文件和其他原图的结果；关闭裁片开关时不创建或修改裁片目录。

推理按 `SCORE_THRESHOLD` 过滤连通域，并通过 `NMS_IOU_THRESHOLD` 抑制重复框，不限制候选框数量；保留单个高置信度特征像素，不按连通域长宽比过滤，避免缩小后的编号、日期和小字漏框。输出目录默认为 `detection_larger/data/test_results`：`images/<原图文件名>` 保存画框后的原尺寸图片，`labels/gt_<原图 stem>.txt` 保存训练使用的 bbox4 标签，每行 `x_min y_min x_max y_max`，保留浮点原图坐标，不附置信度；无框时生成空标签。输入图片的 stem 必须唯一，避免不同扩展名共用标签。图片是画框预览；用于训练时应搭配对应原图。原先每图子目录中的 `overlay.png` 和 `boxes.json` 不再生成。

```bash
.venv/bin/python detection_larger/inference.py

.venv/bin/python detection_larger/components/eval_metrics.py \
  --model detection_larger/checkpoints/<run>/<model>.pth \
  --image-dir detection_larger/data/val_images \
  --label-dir detection_larger/data/val_labels

.venv/bin/python detection_larger_export_onnx.py

.venv/bin/python detection_larger_inference_onnx.py
```

导出脚本无命令行参数；运行前修改顶部的 `CHECKPOINT_PATH`（输入检查点）和 `OUTPUT_PATH`（新的 ONNX 文件路径）常量。ONNX 导出优先使用 Dynamo，失败自动回退到 TorchScript，两者使用 opset 18。安装了 `onnxsim` 时尝试简化图；简化不可用或失败则保留原图。导出需要 `onnx`、`onnxruntime`，Dynamo 另需 `onnxscript`（均已列入根目录依赖）。输入名为 `image`，输出名为 `score`、`geometry`，高宽固定为检查点配置，batch 动态变化（Dynamo 声明范围为 1～32 张）。导出强制比较 batch 1、2 的 PyTorch 与 ONNX CPU 输出，通过后才生成单个 ONNX 文件，并写入 `docdet_config` 元数据；目标存在时拒绝覆盖。大图检查点必须含 `config` 和 `model`，不接受裸权重或旧切片模型。

根目录 `detection_larger_inference_onnx.py` 支持 ONNX Runtime、OpenVINO 和 PyTorch CPU，无命令行参数。修改顶部的模型、图片、输出目录、阈值和裁片常量后运行；默认读取上述导出文件，输出至 `detection_larger/data/test_results/onnx`，裁片至 `detection_larger/data/crops/onnx`。`BACKEND = "auto"` 按后缀选择：`.onnx` 使用 ONNX Runtime，`.pth` 使用 PyTorch；设置 `"openvino"` 可用 OpenVINO 编译 ONNX。显式后端与文件类型不匹配时报错，不会自动换模型文件。OpenVINO 是可选运行依赖，使用 FP32 和单请求延迟优化，按 `score`、`geometry` 名称获取输出。CPU 线程预算集中在 `components/onnx_runtime.py` 的 `CPU_THREADS`，默认 4；ONNX Runtime 开启全部图优化、内存复用，OpenVINO 使用同一线程预算。

`WARMUP_RUNS = 3` 在正式处理前预热，设为 0 可关闭。每张图打印前向耗时与检测耗时：前向包含模型执行及必要设备传输/同步，检测另外包含预处理和后处理；汇总报告图片数、框数、平均前向/检测耗时、整体处理时间和吞吐量。整体时间包含图片读取和结果保存，不含模型加载及预热。默认保存原图画框预览、bbox4 标签和裁片，命名与覆盖规则同 `inference.py`。`EXPAND_BOX` 默认关闭；启用时，左右各增加框高乘 `EXPAND_RATIO_W`，上下各增加框高乘 `EXPAND_RATIO_H`，默认比例均为 0.10，并裁到原图边界；标签、预览与裁片统一使用扩展后的框。密集表格先检查扩框是否带入邻格文字。

导出默认额外运行 CPU 速度对比（`BENCHMARK_ENABLED`），PyTorch 和 ONNX Runtime 均使用 `CPU_THREADS`，batch=1、预热 5 次、计时 20 次，输出平均毫秒数、实测加速比和最终文件大小；相关次数在导出入口常量配置。测速不含预处理与结果保存，不保证固定倍数提速。导出数值检查始终执行，不受测速开关影响。

### 与小图入口的功能核对

| 功能 | 大图处理及原因 |
| --- | --- |
| Dynamo、传统导出回退、图简化 | 已支持；保留大图输入名 `image`、输出名 `score`/`geometry` 和配置元数据，兼容现有大图调用方 |
| 数值验证、动态 batch | 强制比较 batch 1、2，失败不生成目标文件；不能像参考脚本一样仅警告后报告成功 |
| CPU Benchmark、模型大小 | 已补充同线程预算的预热、平均耗时和实测加速比；不照搬未经测量的提速宣传 |
| ONNX FP32/INT8 | 同一个 ONNX Runtime 后端支持保持大图契约和 FP32 输入的量化图；不依赖文件名猜精度。测试验证 QDQ INT8 可运行，实际量化精度需代表性文档校准与验证，导出入口不自动量化 |
| OpenVINO、PyTorch、auto | 已补充；auto 按后缀选择，不宣称能自动判断哪种后端最快，也不静默改用其他权重 |
| CPU 线程与内存优化 | 已支持；不照搬占满全部逻辑核心的设置，避免线程争用，默认预算 4 线程，可集中调整 |
| 预热、单图耗时、整体吞吐 | 已补齐，区分前向、检测和包含文件 I/O 的整体耗时 |
| 单图/目录、裁片、可选扩框 | 已支持，并保留原图浮点 bbox4 标签；裁片仍使用大图已有的 5 位序号，避免改变下游路径契约 |
| 推理输入尺寸 | 修改 `inference.py` 的 `INPUT_SIZE` 常量，支持方形/矩形；ONNX/OpenVINO 必须匹配导出尺寸，不增加命令行覆盖 |
| 小图面积、长宽比过滤 | 不迁移；大图已有回归用例要求保留单像素、细长编号和日期区域，过滤会重新引入漏框 |
| 小图预处理和缩放图预览 | 使用大图共享预处理及原图预览，确保透明图背景、坐标和裁片与现有大图链路一致；不重复建立预处理实现 |
| 坏图跳过、重复覆盖模型 | 不迁移；坏图应带路径报错，导出拒绝覆盖已有模型，符合仓库数据约束 |
| 输入尺寸、阈值等命令行覆盖 | 保持顶部常量配置；导出只保留本次检查点和输出路径，避免同一配置同时存在常量与命令行两套入口 |

训练、推理共用整图预处理，推理一次前向后恢复原图坐标；模型记录训练输入尺寸，推理使用独立的 `INPUT_SIZE` 常量。评估报告总体及 JPG/PNG 等格式分组的 precision/recall/F1，不将扩展名当作通用业务分类。可单独指定 PDF 或网页验证目录进行来源评估。

完整尺寸统计及验证记录见[核对报告](../docs/detection/detection-larger-assessment.md)。

漏框修复依据、验证集指标和对比结果见[推理漏框核查](../docs/detection/inference-missing-boxes-2026-09-12.md)。
