# 小图检测推理

当前入口是仓库根目录的 [detection_inference_640.py](../detection_inference_640.py)，
支持 PyTorch、ONNX Runtime 和 OpenVINO。以下命令均在仓库根目录运行。

```bash
.venv/bin/python detection_inference_640.py --model_path best_detection.onnx --backend onnx --image_path detection/data/test/1 --output_dir detection/data/test_results/current
```

## 配置与输出

| 项目 | 当前行为 |
| --- | --- |
| 输入尺寸 | `--input_size 256x768`，顺序为高×宽；默认值与矩形训练配置一致 |
| 后端 | 默认 `openvino`；`auto` 根据后缀及依赖可用性选择，不会测量哪个后端最快 |
| 置信度 | `--score_thr` 默认 0.7；连通域后处理生成水平矩形框 |
| 预览 | `--output_dir/res_<原文件名>`，保存补边后的模型输入尺寸画框图，不是原尺寸图 |
| 裁片 | 默认关闭；`--save_crops` 开启后从原图裁片，写入顶部 `DEFAULT_CROP_DIR` |
| 裁片命名 | `<原图 stem>-<6 位序号>.png`，从 1 开始；覆盖同名文件，不清理旧多余序号 |
| 扩框 | `EXPAND_BOX=False`；开启后按框高乘宽/高扩边比例，各方向默认 0.10 |
| 计时 | `--benchmark` 显示前向耗时，不等于含图片读取、预处理、解码和保存的总耗时 |

入口只保存预览图和可选裁片，不生成训练标签。部署前检查模型实际输入尺寸，
不要只根据文件名判断；透明图片目前直接转 RGB，与 recognition 的透明合成不同。

## 导出

```bash
.venv/bin/python detection_export_onnx.py --model_path detection/checkpoints/detection_epoch_180.pth --output_path best_detection.onnx --input_size 256x768
```

导出使用本包 ResNet18 模型。现有脚本存在验证失败仅提示的路径，不能把文件已生成
视为数值验证已通过，见[待办](../docs/code-doc-todolist.md)。

包内 `inference.py` 和带版本号脚本是历史实验入口，使用各自顶部常量。
原文档中的 `test/inference.py` 不存在，不应继续使用该命令。
