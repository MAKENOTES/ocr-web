# 统一检测架构

本文说明统一 `detection` 的数据、张量、模型、训练、推理和导出契约，供核对实现和评估改动。
操作入口见 [detection/README.md](../../detection/README.md)；历史实现见
[小图检测架构](../detection_small/architecture.md)和[大图检测架构](../detection_larger/larger-architecture.md)。

## 边界与隔离

统一检测只使用包内 `components/`、`models/` 和 `utils/`，不导入 `detection_small`、
`detection_larger` 或 `recognition`。`components/` 只放训练链路需要的模块：图片读取、
数据加载、增强、变换、真值、损失和训练期指标；推理后处理、评估与报告统一放在
`utils/`，ONNX Runtime 会话配置在根目录导出入口内定义。图片读取、透明通道合成、
缩放补边、归一化、后处理和 ONNX/OpenVINO 加载各自独立实现；
`tests/test_detection_isolation.py` 同时检查导入和运行时文件中的旧目录引用。

## 数据

`tools/prepare_unified_detection_data.py` 是一次性迁移工具，从两个历史检测目录读取
`train/val` 图片和标签，按图片内容 MD5 去重，然后对每个来源域分别用固定种子
做 80/20 划分。迁移结果在 `detection/data/`：

| 目录 | 内容 |
| --- | --- |
| `train_images/`、`train_labels/` | 训练图片和唯一配对标签 |
| `val_images/`、`val_labels/` | 验证图片和唯一配对标签 |
| `test_images/` | 推理演示图片；部分内容与 train/val 重叠，不能作为独立测试集 |
| `split_manifest.csv` | 样本、划分、来源路径和内容 MD5 |

当前唯一图片 6217 张，训练 4974 张、验证 1243 张。文件名以 `ui_` 或 `doc_` 前缀区分
来源；同一内容只保留一份代表样本，因此不会同时进入 train 和 val。清单是复核
归属和追溯来源的依据；新增数据应按同样规则追加，不能把验证图片手工复制进训练目录。

图片与标签按 stem 配对，`gt_<stem>.txt` 和 `<stem>.txt` 必须恰好存在一份，缺失或
同时存在均立即报错。加载器支持：

- 水平 `bbox4`：`x_min y_min x_max y_max [文本]`；
- 水平 `quad8`：四个顶点坐标，要求外接矩形恰好由四个顶点组成；
- `###`：忽略区域，框仍进入四边形列表，但训练掩码排除对应网格；
- 空标签：已确认的无文字图片。

坐标带文件路径和行号的错误提前停止。图片经 `load_rgb_image` 应用 EXIF 方向；
带透明通道时按可见前景亮度选择黑或白背景，完全透明图使用白底。

## 张量与真值

| 张量 | 形状及单位 |
| --- | --- |
| 图片 | `[B,3,H,W]`，训练默认 H=W=768 |
| 置信度真值 | 数据集 `[H/4,W/4]`，训练增加通道维后 `[B,1,H/4,W/4]` |
| 几何真值/输出 | `[B,4,H/4,W/4]`；上、右、下、左距离，以特征图像素计，真值取 `log1p` |

训练预处理先执行 `components/augmentation.py` 的按框安全增强，再保持宽高比缩放到
目标尺寸、边缘中位色居中补边、ImageNet 归一化；验证不做增强。真值先在输出网格
把四边形收缩 `r=0.35`，再给正样本网格写入水平外接矩形距离；重叠区域保留面积
更小的框。模型没有旋转角度通道，因此旋转标注直接拒绝。

输入高宽必须是 32 的倍数且至少 64。训练使用固定尺寸保证 batch 可直接堆叠；
推理可把 `INPUT_SIZE` 当作最大尺寸，按原图宽高比选择尺寸：最长边不超过上限，
每条边向上取整到 32 的倍数，较短边最少 64，不足部分补边缘背景色。

## 数据增强与批次执行

增强全部在 `components/augmentation.py`，共 30 种，分三类执行：

- 几何（4）：随机裁剪、随机补边、随机宽高补边、轻微各向异性缩放；
- 光度（23）：亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、
  色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、JPEG 压缩、
  低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色；
- 遮挡（3）：避开文字框的随机擦除、随机表格线、随机斑点。

执行顺序固定为几何 → 光度 → 遮挡。几何增强在 `ResizeAndPad` 之前的原图上执行，
光度/遮挡增强在缩放补边后的固定输入上执行，避免在大尺寸 PDF 页面上重复做昂贵的
像素操作。几何操作后统一裁剪四边形并删除宽或高不超过 1 像素的退化框；旋转、透视
和剪切会破坏水平框契约，因此不提供。

批次执行由阶段权重控制：每个样本从 `light`、`standard`、`strong` 中采样一个阶段，
三个阶段合计采样 2-4、3-6、5-8 种操作（几何 0-1/0-2/1-2 种，像素 2-3/3-4/4-6 种），
概率与幅度随阶段放大。轮次课程前 15% 以轻增强为主，15%-70% 以标准增强为主，
最后 30% 提高强增强比例；`train.py` 只在阶段边界重建训练 DataLoader。设置
`USE_AUGMENTATION_CURRICULUM=False` 可固定为标准增强。验证集不启用任何增强，
保持指标可复现。

## 训练监控与检查点

TensorBoard 只保留可行动指标：epoch 级 loss/score/geo、学习率、增强阶段权重、
验证集 Precision/Recall/F1（总体及 `ui_`/`doc_` 分组）、最佳 F1，以及 epoch 训练
耗时和吞吐量。验证 F1 每 `METRIC_VAL_INTERVAL` 轮直接在当前内存模型上评估，
不重新加载检查点；默认使用 `METRIC_INPUT_SIZE=768` 按当前部署最佳尺寸推理，
阈值、NMS 和匹配 IoU 由 `METRIC_*` 常量控制。

高开销记录已降频或移除：batch loss 每 `BATCH_LOG_INTERVAL` 个 batch 记录一次，
只记录总梯度范数，不再写逐参数梯度直方图；验证可视化每 `VIS_INTERVAL` 轮写一次
输入图和预测置信度图。旧的逐参数 `Gradients/*`、`Gradient_Norm/*`、逐 batch
score/geo loss 和重复 `epoch/loss_train` 组不再产生。

检查点同时按验证损失和验证 F1 保留 Top-3：`best_detection_*_loss_*.pth` 和
`best_f1_detection_*_f1_*.pth`。F1 与 loss 可能不同步，部署选型应优先比较 F1。

## 模型与损失

ResNet18 的 `/4`、`/8`、`/16`、`/32` 四级特征通过转置卷积逐级融合，在步长 4
的特征图上输出 Sigmoid 置信度和四通道线性几何值。网络是纯卷积结构，因此
PyTorch 可接受任意 32 倍数高宽；ONNX 可导出动态高宽。

损失为整个 batch 的 Dice 加 3 × 正样本 Smooth L1；`###` 掩码同时作用于置信度和
几何损失。纯背景 batch 的几何损失为零，Dice 平滑项保证损失有限。

检查点格式为 `unified-east-resnet18-v1`，保存 `config` 和 `model`。`config` 记录
输入尺寸上限、步长、几何编码、预处理版本和是否动态导出；训练、推理、导出和
ONNX 元数据共用 `detection/utils/checkpoint.py` 的校验。

## 推理、评估与导出

后处理使用四连通域，保留单个高置信度像素，不按连通域长宽比过滤。小连通域用
全像素几何中位数，大连通域用边缘像素统计；几何预测先 `expm1`，再按步长 4
还原到输入图像素。高度至少 8 个特征像素的大连通域默认按低占用列拆分：连续低
占用列宽度需同时达到绝对最小值和连通域高度的 18%，避免把字符内部空隙拆开；
`split_word_gaps=False` 可保留整段输出。目标粒度是词/短语，整行 GT 需要先合并
为一致粒度。坐标减去补边偏移、除以缩放比例后裁到原图边界，再按
`NMS_IOU_THRESHOLD` 抑制重复框，按从上到下、从左到右排序。

`DocumentDetector` 只加载 PyTorch `.pth` 检查点，默认固定方形画布（768×768），
与训练输入一致；`DYNAMIC_INPUT=True` 时才按宽高比选择尺寸。ONNX/OpenVINO
部署统一走单文件入口，由导出文件读取实际高宽：动态模型使用配置的尺寸上限，
固定模型自动改为导出尺寸，并校验元数据、输入输出名称和 `dynamic_spatial` 一致。
`inference.py` 保存画框预览、bbox4 标签和可选裁片；`SAVE_CROPS` 关闭时不创建
或修改裁片目录。

CPU 部署入口 `detection_inference_onnx.py` 是单文件实现：预处理、连通域解码、
NMS 和输出辅助全部内联，不导入 `detection` 包，只依赖 numpy、Pillow、OpenCV
和 ONNX Runtime（可选 OpenVINO）；部署时复制脚本和 `.onnx` 模型即可。其输出
契约与包内实现保持一致，修改任一侧后需运行 ONNX 对比测试。

根目录 `detection_export_onnx.py` 默认导出动态 batch 和固定 768×768 高宽；设置
`DYNAMIC_SPATIAL=True` 时才导出动态高宽。导出先执行 PyTorch 与
ONNX Runtime 的数值比较，固定导出覆盖 batch 1/2，动态导出额外覆盖不同高宽；
直接写入目标文件并校验，目标已存在时覆盖，不创建临时目录。`SIMPLIFY_MODEL`
只影响图简化，不影响强制数值校验。

`components/eval_metrics.py` 直接读取图片和标签目录，按置信度一对一匹配，
默认匹配 IoU 0.5；预测主要落在 `###` 区域内时不计误检，输出总体和按扩展名
分组的 TP/FP/FN、Precision、Recall、F1。扩展名只用于分组，不推断业务来源。
`detection/report_inference.py` 调用两个部署入口在训练/验证集上生成预测，按同一
0.5 IoU 口径复核 P/R/F1、两后端框级一致性和入口耗时，并与训练报告摘要对照；
同时输出 `errors_train.csv` 和 `errors_val.csv` 逐图错误明细。
`detection/report_size_policy.py` 对比全固定 768 与小图 256 两种输入策略的精度和耗时，
只做评估，不修改部署默认值。

加 `--error-previews` 时，入口再对这两个 CSV 中的错误图片调用包内 PyTorch
`DocumentDetector` 重新推理，并复用 `inference.py` 的 `draw_predictions` 画出红色
预测框，按原尺寸、原格式和原文件名写入 `error_previews/<split>/`。本轮推理的
TP/FP/FN、耗时和相对预览路径写入同层的 `error_previews_<split>.csv`，用于复查
预览与指标的对应关系。已有预览 CSV 且图片齐全时跳过，`--force-previews` 只重算
预览，`--force` 在重跑四次推理时一并重算预览。
