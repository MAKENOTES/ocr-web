# 小图检测架构（历史实现，持续维护训练与报告）

本文说明 `detection_small` 的数据、张量、模型、增强、训练指标、评估和报告契约，
供核对实现和评估改动影响。统一 `detection` 是当前主链路；本包保持独立实现，
不导入 `detection`、`detection_larger` 或 `recognition`。

操作入口见 [detection_small/README.md](../../detection_small/README.md)，术语见[术语表](../terminology.md)。

## 边界与目录

训练调用关系为
`train.py → components/dataset.py → components/transforms.py → components/augmentation.py
→ components/ground_truth.py`，模型来自 `models/east_model.py`。

`components/` 只放训练和训练期评估需要的模块：

| 文件 | 职责 |
| --- | --- |
| `image.py` | RGB 读取、EXIF 方向、透明通道合成、边缘背景色 |
| `preprocessing.py` | 尺寸校验、固定/动态缩放补边、ImageNet 归一化、坐标还原 |
| `augmentation.py` | 29 种按框安全增强与轮次课程 |
| `dataset.py` | 严格标签配对、bbox4/quad8 解析、`###` 标记 |
| `transforms.py` | 增强接线、真值生成、张量化 |
| `ground_truth.py` | 收缩置信度、四方向 log1p 距离、忽略掩码 |
| `loss.py` | Dice 置信度损失和正样本 Smooth L1 |
| `postprocess.py` | 四连通域解码、NMS、扩框和裁片命名 |
| `eval_metrics.py` | 框级匹配和 P/R/F1 |

`utils/` 放检查点、配置契约、训练产物读取和报告；`runtime/` 是 ONNX/OpenVINO
部署快照，生成文件和手写入口的边界见下文。

## 数据与真值

数据集枚举图片目录，按 stem 配对唯一标签：

- `gt_<stem>.txt` 和 `<stem>.txt` 恰好存在一份；缺失或歧义立即报错。
- `auto` 模式下，8 个坐标且构成轴对齐矩形时按 `quad8` 解析，否则按 `bbox4`；
  需要强制格式时可显式传入 `label_format`。
- 坐标必须有限、`x_min < x_max`、`y_min < y_max`，且落在图片范围内；
  错误带文件路径和行号，不静默跳过。
- 行尾 `###` 表示忽略区域，四边形保留在 `ignored` 数组中，但不产生正样本。
- 空标签文件表示已确认的无文字图片。

图片经 `load_rgb_image` 应用 EXIF 方向；带透明通道时按可见前景亮度选择黑或白背景，
完全透明图使用白底。训练、训练期指标和报告都走同一读取路径。

| 张量 | 形状及单位 |
| --- | --- |
| 图片 | `[B,3,H,W]`，默认 H=256、W=768 |
| 置信度真值 | 数据集 `[H/4,W/4]`，训练增加通道维后 `[B,1,H/4,W/4]` |
| 几何真值/输出 | `[B,4,H/4,W/4]`；上、右、下、左距离，以特征图像素计，真值取 `log1p` |
| 忽略掩码 | `[B,H/4,W/4]`；`###` 覆盖的网格为 0，其余为 1 |

真值先在输出网格收缩四边形（`r=0.35`），再给正样本网格写入水平外接矩形距离；
重叠区域保留面积更小的框。模型没有旋转角度通道，旋转标注直接拒绝。

## 数据增强

增强集中在 `components/augmentation.py`，共 29 种，分三类：

- 几何（3）：随机补边、随机宽高补边、轻微各向异性缩放；
- 光度（23）：亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、
  色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、JPEG 压缩、
  低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色；
- 遮挡（3）：避开文字框的随机擦除、随机表格线、随机斑点。

执行顺序固定为几何 → 光度 → 遮挡。几何增强在 `ResizeAndPad` 之前的原图上执行，
光度/遮挡增强在缩放补边后的固定输入上执行。几何操作后统一裁剪四边形并删除宽或高
不超过 1 像素的退化框；旋转、透视和剪切会破坏水平框契约，因此不提供。

小图数据已经是紧凑 UI 裁剪，随机裁剪会系统性截断文字，因此几何段不包含随机裁剪；
旧 `RandomHorizontalCrop` 同样没有进入默认流水线。

批次执行由阶段权重控制：每个样本从 `light`、`standard`、`strong` 中采样一个阶段，
三个阶段分别采样几何 0-1/0-2/1-2 种、像素 2-3/3-4/4-6 种操作，概率与幅度随阶段放大。
轮次课程前 15% 以轻增强为主，15%-70% 以标准增强为主，最后 30% 提高强增强比例；
`train.py` 只在阶段边界重建训练 DataLoader。设置
`USE_AUGMENTATION_CURRICULUM=False` 可固定为标准增强。验证集不启用任何增强。

## 训练监控与检查点

TensorBoard 只保留可行动指标：epoch 级 loss/score/geo、学习率、增强阶段权重、
验证集 Precision/Recall/F1、最佳 F1，以及 epoch 训练耗时和吞吐量。验证损失按
`VAL_INTERVAL`/`START_DENSE_VAL_EPOCH` 调度；框级指标每 `METRIC_VAL_INTERVAL` 轮
直接在当前内存模型上评估，不重新加载检查点。

`evaluate_detection` 的输入尺寸默认为训练尺寸，阈值 `METRIC_SCORE_THRESHOLD=0.7`，
匹配 IoU `METRIC_MATCH_IOU=0.5`。指标在还原到原图坐标后计算，因此不同输入尺寸和
补边方式可以直接比较。

高开销记录已降频或移除：batch loss 每 `BATCH_LOG_INTERVAL` 个 batch 记录一次，
只记录总梯度范数，不再写逐参数梯度直方图；验证可视化每 `VIS_INTERVAL` 轮写一次。

每次训练在 `checkpoints/ocr_experiment_<时间戳>/` 下保存：

- `best_detection_epoch_*_loss_*.pth`：验证损失 Top-3；
- `best_f1_detection_epoch_*_f1_*.pth`：验证 F1 Top-3；
- `detection_epoch_*.pth`：常规检查点，保留最近 5 份；
- `detection_epoch_*_early_stop.pth`：早停时的模型；
- `run_config.json`：输入尺寸、数据目录、增强和指标口径。

检查点仍是纯 `state_dict`，与根目录部署入口和历史权重兼容。部署选型优先比较 F1，
不要只看验证损失。

## 模型与损失

ResNet18 的 `/4`、`/8`、`/16`、`/32` 四级特征通过转置卷积逐级融合，在步长 4
的特征图上输出 Sigmoid 置信度和四通道线性几何值。输入高宽必须是 32 的倍数且
至少 64。

损失为整个 batch 的 Dice 加 `lambda_iou=3` 的正样本 Smooth L1；`###` 掩码同时作用于
置信度和几何损失。纯背景 batch 的几何损失为零，Dice 平滑项保证损失有限。

## 评估、后处理与报告

`components/postprocess.py` 集中实现后处理，训练期指标、报告和部署 runtime 共用：

- 对置信度严格大于阈值的网格做四连通域；
- 最小面积 2；小区域面积按输入像素面积缩放，1024×1024 时为 15，256×768 时为 3；
- 小区域用全像素几何中位数出框并做长宽比过滤；大区域用边缘窄带中位数；
- 几何输出先 `expm1`，再乘步长 4；预测框还原到原图坐标；
- 按置信度执行 NMS，默认 IoU 0.3；再按置信度从高到低一对一匹配，默认 IoU 0.5；
- 预测主要落在 `###` 内时不计误检。

`report_training.py` 读取检查点目录、同名 TensorBoard 运行目录和验证集，生成静态
HTML 报告与 `summary.json`。报告比较训练尺寸和放大 1.25 倍的额外尺寸，置信度阈值
默认 0.7 和 0.85；错误图片使用最佳工作点绘制，橙色为漏检真值、红色为误检预测。
报告同时检查推理入口的 `MODEL_PATH`、`INPUT_SIZE` 和 ONNX 导出尺寸是否与主指标一致。

## ONNX 导出与部署

根目录 `detection_small_export_onnx.py` 导出自包含 ONNX：

- 默认固定 `INPUT_SIZE=(256, 768)`；`DYNAMIC_SPATIAL=True` 时导出动态高宽；
- 输入名 `image`，输出名 `score` 和 `geometry`，opset 18，动态 batch；
- 元数据键 `ocrsmall_config`，由 `utils/config.py` 的 `checkpoint_config` 生成，
  记录格式版本、输入尺寸上限、步长、几何编码、预处理版本和 `dynamic_spatial`；
- 先写 `<stem>.tmp.onnx`，PyTorch 与 ONNX Runtime 在 batch 1/2（动态模型再加不同
  高宽）上逐输出比较，全部通过后才替换目标；失败时删除临时文件，不发布模型。

`detection_small/runtime/` 是自包含部署包：

| 文件 | 来源/职责 |
| --- | --- |
| `image.py`、`preprocessing.py`、`postprocess.py`、`config.py` | 由 `tools/sync_detection_small_runtime.py` 从 `components/`、`utils/` 生成 |
| `backends.py` | ONNX Runtime 和 OpenVINO 会话、元数据/输入输出校验 |
| `engine.py` | 单图/批量推理、NMS、原图坐标还原、预览/标签/裁片输出 |
| `cli.py`、`__main__.py`、`__init__.py` | 命令行和稳定 Python 接口 |

输入尺寸由 ONNX 元数据决定：固定导出的静态 shape 必须等于 `config.input_size`；
动态导出的高宽维为 `None`，按原图宽高比计算 32 倍数尺寸并补边。runtime 不导入
训练、报告或仓库其他包，只使用相对导入。

修改 `components/image.py`、`components/preprocessing.py`、`components/postprocess.py`
或 `utils/config.py` 后必须运行同步脚本；`tools/build_detection_small_bundle.py`
先校验 runtime 同步，再构建包含 runtime、`detection_small_inference_onnx.py`、模型、
依赖清单和 manifest 的独立目录，并在隔离路径下真实推理一次。

## 与部署入口的边界

根目录 `detection_small_inference_onnx.py` 只保留默认常量和 `main`，推理全部委托给
`detection_small/runtime`。旧的 `detection_small_inference_640.py` 保留 PyTorch/
ONNX/OpenVINO 多后端实验行为，已适配新的 ONNX 输入输出名称，但不使用模型元数据和
NMS，不作为当前部署默认入口。

`detection_small_quantize_int8.py` 按纯 `state_dict` 或 ONNX 权重工作；训练检查点格式
保持不变。部署前用报告复核输入尺寸和检查点，不要手工拼接权重与配置。
