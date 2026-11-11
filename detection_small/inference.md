# 小图检测推理与 ONNX 部署

本文说明小图检测的 ONNX 导出、CPU 推理和独立部署方式。统一 `detection` 是当前主链路，
但小图专用模型仍可按本页导出和部署。

## 推荐入口

ONNX CPU 部署使用根目录 [detection_small_inference_onnx.py](../detection_small_inference_onnx.py)，
实现位于 [detection_small/runtime/](../detection_small/runtime/)：

```bash
.venv/bin/python detection_small_inference_onnx.py \
    --model_path best_detection_small.onnx \
    --image_path detection_small/data/test/4 \
    --output_dir detection_small/data/test_results/onnx/4 \
    --backend onnx
```

| 项目 | 当前行为 |
| --- | --- |
| 输入尺寸 | 读取 ONNX 元数据；固定导出使用静态高宽，动态导出按原图宽高比选择 32 倍数尺寸 |
| 后端 | `auto` 等价 `onnx`；`openvino` 需显式选择 |
| 置信度 | `--score_thr` 默认 0.7，`--nms_iou` 默认 0.3 |
| 预览 | `<output_dir>/images/<原文件名>`，原图尺寸画框 |
| 标签 | `<output_dir>/labels/gt_<stem>.txt`，bbox4 原图像素坐标；无框时为空文件 |
| 裁片 | 默认关闭；开启后按 `<stem>-<6 位序号>.png` 写入 `--crop_dir` |
| 扩框 | 根目录入口常量控制，默认关闭；开启后按框高比例扩边 |
| 透明图片 | runtime 使用共享 RGB 读取，透明 PNG 按可见前景亮度合成黑/白背景 |

输入输出名固定为 `image`、`score`、`geometry`；加载时会校验元数据格式、输入输出名、
静态尺寸和 `dynamic_spatial` 是否一致。`python -m detection_small.runtime`
提供同样的显式路径接口。

## 导出

导出配置集中在 [detection_small_export_onnx.py](../detection_small_export_onnx.py) 顶部：

```python
CHECKPOINT_PATH = Path("detection_small/checkpoints/<本次运行>/best_f1_... .pth")
OUTPUT_PATH = Path("best_detection_small.onnx")
INPUT_SIZE = (256, 768)
DYNAMIC_SPATIAL = False
```

```bash
.venv/bin/python detection_small_export_onnx.py
```

导出先写临时文件，PyTorch 与 ONNX Runtime 在 batch 1/2（动态模型再加不同高宽）上比较
`score` 和 `geometry`；全部通过后才覆盖目标。失败时不会留下目标或临时模型。
`ocrsmall_config` 元数据记录格式版本、输入尺寸上限、步长、几何编码、预处理版本和
`dynamic_spatial`，推理入口据此决定固定或动态尺寸。

导出后如需同步新的输入输出名，旧多后端入口已适配，但其 PyTorch 分支和输出布局属于
历史实验行为。

## 独立部署包

```bash
.venv/bin/python tools/sync_detection_small_runtime.py
.venv/bin/python tools/build_detection_small_bundle.py \
    --model best_detection_small.onnx \
    --out dist/detection_small_runtime \
    --force
```

构建前会检查 runtime 与 `components/image.py`、`components/preprocessing.py`、
`components/postprocess.py`、`utils/config.py` 是否同步；产物只包含 runtime、薄入口、
模型、依赖清单和 `manifest.json`，并在隔离路径下实际推理一次。修改生成源文件后必须
重新同步并提交 `detection_small/runtime`。

## 历史入口

根目录 `detection_small_inference_640.py` 保留 PyTorch/ONNX/OpenVINO 多后端实验行为，
默认仍按 CLI/常量选择输入尺寸，不使用模型元数据、NMS 和新标签目录布局；
`detection_small/inference.py` 也是历史实验入口。需要与训练和报告一致的部署行为时，
使用上面的 `detection_small_inference_onnx.py`。

`detection_small_quantize_int8.py` 仍可用于 INT8 静态量化，但需在导出的 FP32 ONNX 上运行，
并重新验证量化模型的数值误差。
