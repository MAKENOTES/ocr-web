# 小图检测架构（历史实现）

本文说明历史小图检测的数据格式、模型张量、训练与推理契约，供核对实现和评估改动影响。
统一 `detection` 是当前主链路，本文不描述统一检测的行为。

本页对应 `../../detection_small/train.py`、`detection_small_inference_640.py` 的历史实现。
操作入口见 [detection_small/README.md](../../detection_small/README.md)，术语见[术语表](../terminology.md)。

## 数据与张量

训练调用关系为 `train_resnet.py → components/dataset.py → components/transforms.py → components/ground_truth.py`。
模型来自 `models/east_model.py`。
训练仅枚举 PNG；`OCRDataset` 类本身默认支持 JPG/JPEG/PNG。

加载器按 stem 选择标签，存在 `gt_<stem>.txt` 时优先使用它。`require_labels=True`
检查缺标签；默认关闭时，缺标签作为空框样本返回。当前解析会跳过坏行和退化框并打印警告。
`###` 只影响文字列表，没有在真值中屏蔽对应框。这些是与仓库要求的差距，见[待办](../code-doc-todolist.md)。

坐标支持 bbox4 与 quad8，按连续数字字段数量判断，保留浮点值。
正文若以数字开头，可能干扰格式判断，新数据应明确使用水平 bbox4，迁移前单独检查标签。
图像直接转 RGB，再做轻微颜色扰动、居中等比缩放、边缘中位色补边和 ImageNet 归一化。
验证不做随机颜色扰动；`RandomScale`、`RandomHorizontalCrop` 不在默认流水线中。

| 张量 | 形状及单位 |
| --- | --- |
| 图片 | `[B,3,H,W]`，默认 H=256、W=768 |
| 置信度真值 | 数据集为 `[H/4,W/4]`，训练增加通道维后为 `[B,1,H/4,W/4]` |
| 几何真值/输出 | `[B,4,H/4,W/4]`；上、右、下、左距离，以特征图像素计，真值取 `log1p` |

真值先在输出网格收缩四边形（`r=0.35`），再给正样本网格写入水平外接矩形距离。
重叠区域按实现中的面积规则选择几何目标。模型没有旋转角度通道。

## 模型、损失与指标

ResNet18 四级特征通过转置卷积逐级融合，在步长 4 的特征图上输出 Sigmoid 置信度和四通道线性几何值。
`EASTLoss.forward` 使用 Dice 加加权 Smooth L1，训练入口的几何权重为 3；
`lambda_iou` 是历史参数名，当前前向不调用 IoU 辅助方法。纯背景批次几何损失为零，Dice 按全 batch 计算。

根目录推理用四连通域，几何预测先做 `expm1`，再按步长 4 还原到输入图坐标。
小区域采用全像素中位数，大区域采用边缘像素中位数；小区域阈值随输入面积缩放，
仍有最小面积和小区域长宽比过滤。坐标减去补边偏移、除以缩放比例后用于原图裁片。当前入口未调用 NMS。

`components/eval_metrics.py` 独立使用八连通域、固定小区域阈值，
用匈牙利匹配和 IoU 阈值统计 TP/FP/FN。标签辅助解析器只读取空白分隔 bbox4 并取整，
不是训练解析器；`evaluate_detection` 的尺寸路径仍按标量方形处理。
不能将该模块的指标称为与矩形部署完全一致，需按待办补对齐验证。

## 历史文件

`inference_v3resnet18_640.py` 保留历史推理实验配置。
`transforms.py` 显式导入 `ground_truth.py`，训练默认使用该真值实现。
