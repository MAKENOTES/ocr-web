# 模型推理脚本使用说明

该目录下的 `inference.py` 脚本用于验证训练好的 EAST 模型。它会读取图像，检测文本区域，并将画有检测框的结果图像保存下来。

## 使用方法

在项目根目录下运行以下命令：

```bash
python test/inference.py --image_path <图像路径或目录> --model_path <模型权重路径>
```

### 参数说明

- `--image_path`: (必选) 输入图像的路径，或者包含图像的目录路径。支持 `.jpg` 和 `.png`。
- `--model_path`: (必选) 训练好的模型权重文件路径 (例如 `east_model_epoch_50.pth`)。
- `--output_dir`: (可选) 结果保存目录，默认为 `test_results`。
- `--input_size`: (可选) 输入图像的尺寸，默认为 `512`。请确保与训练时使用的尺寸一致。

### 示例

1. **检测单张图像**：
   ```bash
   python test/inference.py --image_path data/test_images/img_1.jpg --model_path east_model_epoch_50.pth
   ```

2. **检测整个目录**：
   ```bash
   python test/inference.py --image_path data/test_images --model_path east_model_epoch_50.pth --output_dir output_viz
   ```

## 输出

脚本将在 `output_dir` (默认 `test_results`) 中生成结果图像。结果图像的文件名将以 `res_` 开头，图像上会用绿色矩形框标出检测到的文本区域。
