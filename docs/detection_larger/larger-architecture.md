# 大图检测架构

本文说明大图文字检测的数据、增强、训练、推理、报告和导出契约，供核对实现和改动。
操作入口见[包说明](../../detection_larger/README.md)；术语见[术语表](../terminology.md)。

## 边界与隔离

`detection_larger` 只使用自身 `components/`、`models/`、`runtime/` 和 `utils/`，
不导入 `detection`、`detection_small` 或 `recognition`。图片读取、EXIF 方向、
透明通道合成、缩放补边、归一化、后处理和 ONNX/OpenVINO 加载都由本包独立实现。

`components/` 只放训练链路需要的模块：图片读取、推理预处理、数据加载、增强、
变换、真值、损失和训练期指标；推理后处理、配置契约和报告放在 `utils/`。
`runtime/` 是部署运行包：图片读取、预处理、后处理和配置契约由
`tools/sync_detection_larger_runtime.py` 从源文件生成，`backends.py`、`engine.py`
和 `cli.py` 为手写模块。

## 数据契约

直接读取已准备好的独立训练、验证目录，不划分、不搬移数据。图片和标签按 stem
配对，`gt_<stem>.txt` 和 `<stem>.txt` 必须恰好存在一份；缺失或同时存在立即报错。
标签坐标须有限、非退化且在图内。默认 bbox4，显式 `quad8` 只接受水平矩形；
`###` 用独立训练掩码排除，空标签表示无文字。

图片经 `components/image.py` 的 `load_rgb_image` 应用 EXIF 方向；带透明通道时按
可见前景亮度选择黑或白背景，完全透明图使用白底。训练入口只检查 train/val 目录
路径不同，不检查两目录中是否复制了同一图片。

## 张量与真值

| 张量 | 形状及单位 |
| --- | --- |
| 图片 | `[B,3,H,W]`，训练默认 H=W=640 |
| 置信度真值 | 数据集 `[H/4,W/4]`，训练增加通道维后 `[B,1,H/4,W/4]` |
| 几何真值/输出 | `[B,4,H/4,W/4]`；上、右、下、左距离，以特征图像素计，真值取 `log1p` |

输入高宽必须是 32 的倍数且至少 64。训练使用固定尺寸保证 batch 可直接堆叠；
PyTorch 推理默认也使用固定方形画布，`DYNAMIC_INPUT=True` 时才按原图宽高比选择
尺寸：最长边不超过上限，每条边向上取整到 32 的倍数，较短边最少 64。

## 数据增强与批次执行

增强全部在 `components/augmentation.py`，共 30 种，分三类执行：

- 几何（4）：随机裁剪、随机补边、随机宽高补边、轻微各向异性缩放；
- 光度（23）：亮度、对比度、饱和度、色相、Gamma、自动对比度、直方图均衡、
  色调分离、曝光过度、锐化、高斯/运动/中值模糊、高斯/椒盐噪声、JPEG 压缩、
  低分辨率重建、CLAHE、通道增益、阴影渐变、背景染色、暗角、反色；
- 遮挡（3）：避开文字框的随机擦除、随机表格线、随机斑点。

执行顺序固定为几何 → 光度 → 遮挡。几何增强在 `ResizeAndPad` 之前的原图上执行，
光度/遮挡增强在缩放补边后的固定输入上执行，避免在大尺寸 PDF 页面上重复做昂贵的
像素操作。几何操作后统一裁剪四边形并删除宽或高不超过 1 像素的退化框；旋转、
透视和剪切会破坏水平框契约，因此不提供。

批次执行由阶段权重控制：每个样本从 `light`、`standard`、`strong` 中采样一个阶段，
三个阶段合计采样 2-4、3-6、5-8 种操作（几何 0-1/0-2/1-2 种，像素 2-3/3-4/4-6 种）。
轮次课程前 15% 以轻增强为主，15%-70% 以标准增强为主，最后 30% 提高强增强比例；
`train.py` 只在阶段边界重建 DataLoader。验证集不启用任何增强，保持指标可复现。

## 模型、损失与检查点

模型为独立 ResNet18 EAST：`/4`、`/8`、`/16`、`/32` 四级特征通过转置卷积逐级融合，
在步长 4 的特征图上输出 Sigmoid 置信度和四通道线性几何值。损失为整个 batch 的
Dice 加 3 × 正样本 Smooth L1；`###` 掩码同时作用于置信度和几何损失。

检查点只保存 `config` 和 `model`，格式为 `document-east-resnet18-full-image-v2`。
`config` 记录输入尺寸上限、步长、几何编码、预处理版本和 `dynamic_spatial`；
缺少该字段的历史 v2 检查点按固定高宽读取。配置构造与校验在
`utils/config.py`，读写和模型构建在 `utils/checkpoint.py`，训练、推理、导出和
ONNX 元数据共用同一份契约。训练、推理、导出各自的 `INPUT_SIZE` 独立配置，
固定高宽导出的 ONNX/OpenVINO 模型自动使用导出尺寸。

## 训练监控

TensorBoard 保留 epoch 级 loss/score/geo、学习率、增强阶段权重、验证集
Precision/Recall/F1（总体及按扩展名分组）、最佳 F1，以及 epoch 训练耗时和吞吐量。
验证 F1 每 `METRIC_VAL_INTERVAL` 轮直接在当前内存模型上评估，默认使用
`METRIC_INPUT_SIZE=640`（与训练和部署一致），阈值、NMS 和匹配 IoU 由 `METRIC_*`
常量控制。

高开销记录已降频或移除：batch loss 每 `BATCH_LOG_INTERVAL` 个 batch 记录一次，
只记录总梯度范数，不再写逐参数梯度直方图；验证可视化每 `VIS_INTERVAL` 轮写一次
输入图和预测置信度图。检查点同时按验证损失和验证 F1 保留 Top-3，
部署选型优先比较 F1。

## 推理、评估与报告

后处理使用四连通域，保留单个高置信度像素，不按连通域长宽比过滤。小连通域用
全像素几何中位数，大连通域用边缘像素统计；几何预测先 `expm1` 再按步长 4 还原到
输入图像素。页面真值通常是整行框，默认不按词间空隙拆分；`split_word_gaps=True`
只用于对照实验。坐标减去补边偏移、除以缩放比例后裁到原图边界，再按
`NMS_IOU_THRESHOLD` 抑制重复框，按从上到下、从左到右排序。

`DocumentDetector` 默认按后缀选择 PyTorch 或 ONNX Runtime；显式 `openvino` 使用
OpenVINO CPU。ONNX/OpenVINO 的尺寸来自导出元数据并校验输入输出名称和
`dynamic_spatial` 一致。`inference.py` 保存画框预览、bbox4 标签和可选裁片；
`SAVE_CROPS` 关闭时不创建或修改裁片目录。

`components/eval_metrics.py` 直接读取图片和标签目录，按置信度一对一匹配，默认
匹配 IoU 0.5；预测主要落在 `###` 区域内时不计误检，输出总体和按扩展名分组的
TP/FP/FN、Precision、Recall、F1。扩展名只用于分组，不推断业务来源。

`report_training.py` 读取某个运行目录的检查点、`run_config.json` 和 TensorBoard
事件，在验证集上重新评估主检查点、最佳 loss 和最后一轮，比较训练尺寸、+128
更大输入和不同阈值的工作点，并输出错误集中度、页面方向拆分和最差图片可视化。
报告还会检查推理入口和导出入口的默认检查点、尺寸是否与主检查点一致。报告是
本地产物，不提交。

## 部署与导出

CPU 部署入口 `detection_larger_inference_onnx.py` 只保留默认常量，推理实现在
`detection_larger/runtime`。runtime 中的图片读取、预处理、后处理和配置契约由
`tools/sync_detection_larger_runtime.py` 从源文件生成；runtime 不导入训练、报告或
PyTorch 推理模块。部署包由 `tools/build_detection_larger_bundle.py` 生成，包含
runtime、入口和 `.onnx` 模型，并在隔离路径下完成一次推理自检；共享源文件修改后
未重新同步时，构建脚本会拒绝生成。

根目录 `detection_larger_export_onnx.py` 默认导出动态 batch 和固定 640×640 高宽；
`DYNAMIC_SPATIAL=True` 时才导出动态高宽。导出优先 Dynamo，失败回退 TorchScript，
opset 18；`SIMPLIFY_MODEL` 只影响图简化。导出直接覆盖目标文件，随后比较
PyTorch 与 ONNX Runtime 的 batch 1/2 输出，动态导出额外覆盖不同高宽；
`BENCHMARK_ENABLED` 控制可选的 CPU 测速。目标 ONNX 不会自动跟随检查点更新，
修改权重或尺寸后必须重新导出。
