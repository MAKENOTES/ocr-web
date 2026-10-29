# 大图检测架构

本文说明大图文字检测的数据加载、模型训练、整图推理与导出契约，供核对各入口的行为。

本页以 `detection_larger` 与两个同名前缀的根目录入口为准，操作见[包说明](../../detection_larger/README.md)。
本包不导入统一 `detection` 和小图检测包；当前仅复用 recognition `preprocessing.py` 的 `load_rgb_image` 做透明合成，
不调用识别去红章、缩放或模型。术语见[术语表](../terminology.md)。

## 数据、模型与训练

直接读取已准备好的独立训练、验证目录，不划分、不搬移数据。图片和标签按 stem 配对，
缺失或同时存在两种候选标签均报错；标签坐标须有限、非退化且在图内。
默认 bbox4，显式 `quad8` 只接受水平矩形。`###` 用独立训练掩码排除，空标签表示无文字。
入口只检查 train/val 目录路径不同，不检查两目录中是否复制了同一图片。

模型为独立 ResNet18 EAST，输出置信度 `[B,1,H/4,W/4]` 与几何 `[B,4,H/4,W/4]`。
几何顺序为上、右、下、左，以特征图像素计并使用 `log1p` 真值；模型几何头为线性输出。
预处理是 RGB、整图居中等比缩放、边缘中位色补边、ImageNet 归一化。
训练额外使用 brightness=0.12、contrast=0.15、saturation=0.10、hue=0.02 的颜色扰动。
损失为 Dice + 3 × 正样本 Smooth L1；不切片、不恢复表格结构。

检查点只保存 `config` 和 `model`，配置格式为 `document-east-resnet18-full-image-v2`。
格式常量、配置构造与校验、`save_checkpoint` 和 `load_checkpoint` 都在
`detection_larger/utils/checkpoint.py`，由 `detection_larger/utils/__init__.py` 汇总导出；
`detection_larger/models/east_model.py` 只定义网络。训练、推理、导出和 ONNX 元数据校验共用同一份契约。
`--weights` 可初始化纯参数或含 model 的权重，但不恢复优化器和轮次。
训练、推理、导出各自的 `INPUT_SIZE` 独立配置，当前均为 640；推理不从检查点继承尺寸。
ONNX/OpenVINO 还会检查模型实际输入形状与推理常量一致。

后处理使用四连通域，保留单个高置信度像素，不按区域长宽比过滤；
小区域用全像素几何中位数，大区域用边缘统计，随后还原原图坐标并执行 NMS。
仍有解码框宽高过滤，因此“保留单像素连通域”不表示无条件输出任何尺寸框。

评估按置信度一对一匹配，默认匹配 IoU=0.5；预测主要落在忽略区域时不计误检。
输出总体和扩展名分组的 TP/FP/FN、Precision、Recall、F1，不把 JPG/PNG 推断为业务来源。

## 推理、评估与导出


推理不使用命令行参数，先修改 `../../detection_larger/inference.py` 顶部常量：`MODEL_PATH`（文档专用 `.pth` 或 `.onnx`）、`IMAGE_PATH`（图片或目录）、`OUTPUT_DIR`（允许覆盖同名结果的输出目录）、`DEVICE`、`SCORE_THRESHOLD`、`NMS_IOU_THRESHOLD`、`SAVE_CROPS` 和 `CROP_DIR`。路径相对于运行目录，建议在仓库根目录执行。默认使用 `detection_larger/data/test_images` 图片目录，`DEVICE = "auto"` 让 PyTorch 优先使用 CUDA，无可用 GPU 时回退到 CPU；也可指定 `"cuda:0"` 或 `"cpu"`。ONNX 在 `auto` 下使用 CPU，也可显式设置 `cpu`。需要保存文字裁片时设置 `SAVE_CROPS = True`，裁片平铺在 `CROP_DIR`（默认 `detection_larger/data/crops/test_images`）下，不再按原图建子目录；文件名为 `<原图 stem>-<5 位序号>.png`，序号从 1 开始，例如原图 `du20260910-000001.png` 的首个裁片是 `du20260910-000001-00001.png`。裁片名和标签都以原图 stem 为前缀，所以输入图片的 stem 必须唯一，同名不同扩展名会直接报错。重复运行覆盖本次输入对应的图片、标签及裁片，并清理该原图超出本次框数的旧序号裁片，保留其他文件和其他原图的结果；关闭裁片开关时不创建或修改裁片目录。

推理按 `SCORE_THRESHOLD` 过滤连通域，并通过 `NMS_IOU_THRESHOLD` 抑制重复框，不限制候选框数量；保留单个高置信度特征像素，不按连通域长宽比过滤，避免缩小后的编号、日期和小字漏框。输出目录默认为 `../../detection_larger/data/test_results`：`images/<原图文件名>` 保存画框后的原尺寸图片，`labels/gt_<原图 stem>.txt` 保存训练使用的 bbox4 标签，每行 `x_min y_min x_max y_max`，保留浮点原图坐标，不附置信度；无框时生成空标签。输入图片的 stem 必须唯一，避免不同扩展名共用标签。图片是画框预览；用于训练时应搭配对应原图。原先每图子目录中的 `overlay.png` 和 `boxes.json` 不再生成。

```bash
.venv/bin/python detection_larger/inference.py

.venv/bin/python detection_larger/components/eval_metrics.py \
  --model detection_larger/checkpoints/<run>/<model>.pth \
  --image-dir detection_larger/data/val_images \
  --label-dir detection_larger/data/val_labels

.venv/bin/python detection_larger_export_onnx.py

.venv/bin/python detection_larger_inference_onnx.py
```

导出脚本无命令行参数；运行前修改顶部的 `CHECKPOINT_PATH`（输入检查点）、`OUTPUT_PATH`（新的 ONNX 文件路径）和 `INPUT_SIZE`（默认 640，支持整数或高宽二元组）常量。ONNX 导出优先使用 Dynamo，失败自动回退到 TorchScript，两者使用 opset 18。安装了 `onnxsim` 时尝试简化图；简化不可用或失败则保留原图。导出需要 `onnx`、`onnxruntime`，Dynamo 另需 `onnxscript`（均已列入根目录依赖）。输入名为 `image`，输出名为 `score`、`geometry`，高宽固定为导出脚本的 `INPUT_SIZE`，元数据同步记录本次导出尺寸；推理脚本的尺寸常量需与之保持一致，batch 动态变化（Dynamo 声明范围为 1～32 张）。导出强制比较 batch 1、2 的 PyTorch 与 ONNX CPU 输出，通过后才生成单个 ONNX 文件，并写入 `docdet_config` 元数据；目标存在时拒绝覆盖。大图检查点必须含 `config` 和 `model`，不接受裸权重或旧切片模型。

根目录 `../../detection_larger_inference_onnx.py` 支持 ONNX Runtime、OpenVINO 和 PyTorch CPU，无命令行参数。修改顶部的模型、图片、输出目录、阈值和裁片常量后运行；默认读取上述导出文件，输出至 `detection_larger/data/test_results/onnx`，裁片至 `detection_larger/data/crops/onnx`。当前 `BACKEND = "openvino"`；改为 `"auto"` 时按后缀选择：`.onnx` 使用 ONNX Runtime，`.pth` 使用 PyTorch；设置 `"openvino"` 可用 OpenVINO 编译 ONNX。显式后端与文件类型不匹配时报错，不会自动换模型文件。OpenVINO 是可选运行依赖，使用 FP32 和单请求延迟优化，按 `score`、`geometry` 名称获取输出。CPU 线程预算集中在 `detection_larger/components/onnx_runtime.py` 的 `CPU_THREADS`，默认 4；ONNX Runtime 开启全部图优化、内存复用，OpenVINO 使用同一线程预算。

`WARMUP_RUNS = 3` 在正式处理前预热，设为 0 可关闭。每张图打印前向耗时与检测耗时：前向包含模型执行及必要设备传输/同步，检测另外包含预处理和后处理；汇总报告图片数、框数、平均前向/检测耗时、整体处理时间和吞吐量。整体时间包含图片读取和结果保存，不含模型加载及预热。默认保存原图画框预览和 bbox4 标签；`SAVE_CROPS=False`，仅开启后保存裁片，命名与覆盖规则同 `../../detection_larger/inference.py`。`EXPAND_BOX` 默认关闭；启用时，左右各增加框高乘 `EXPAND_RATIO_W`，上下各增加框高乘 `EXPAND_RATIO_H`，默认比例均为 0.10，并裁到原图边界；标签、预览与裁片统一使用扩展后的框。密集表格先检查扩框是否带入邻格文字。

导出默认执行 CPU 速度对比（`BENCHMARK_ENABLED = True`）。需要跳过测速时设为 `False`；数值校验始终执行。测速会额外执行共 50 次 CPU 前向。启用时，PyTorch 和 ONNX Runtime 均使用 `CPU_THREADS`，batch=1、预热 5 次、计时 20 次，输出平均毫秒数、实测加速比和最终文件大小；相关次数在导出入口常量配置。测速不含预处理与结果保存，不保证固定倍数提速。导出数值检查始终执行，不受测速开关影响。

