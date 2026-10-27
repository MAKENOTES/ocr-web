# Recognition 部署

结论：部署只拷贝构建产物 `dist/recognition_runtime/`（下文简称 bundle）。产物自带
运行代码、模型、字典、依赖清单和 `manifest.json`，不依赖仓库目录、训练数据或当前工作目录。

模型输入、真实宽度和字典契约见[当前架构](../recognition.md)；训练、导出和 CPU 量化
仍按[运行说明](running.md)执行。

## 1. 构建产物

`recognition/runtime/` 是单向生成的部署快照：预处理、推理数据集、模型、解码和检查点工具
来自 `components/`、`models/`、`utils/`，由同步脚本复制并把导入改写为相对路径；只有
`engine.py`、`backends.py`、`cli.py`、`loader.py` 是手写。修改源文件后先同步：

```bash
.venv/bin/python tools/sync_recognition_runtime.py          # 重新生成 runtime
.venv/bin/python tools/sync_recognition_runtime.py --check  # 只检查
```

构建脚本会在开始前检查同步状态，runtime 与源文件不一致时直接失败，避免把过期代码
打进部署包。

在仓库根目录执行：

```bash
.venv/bin/python tools/build_recognition_bundle.py \
  --model best_recognition_layer.onnx \
  --keys recognition/keys.txt \
  --out dist/recognition_runtime --zip
```

参数：

| 参数 | 用途 |
| --- | --- |
| `--model` | 识别模型，支持 `.onnx`、`.xml`（OpenVINO IR）和 `.pth` |
| `--keys` | 必须与模型导出时使用的 `keys.txt` 完全一致 |
| `--out` | 输出目录，默认 `dist/recognition_runtime` |
| `--force` | 覆盖空目录或含 `manifest.json` 的旧产物，其他非空目录拒绝覆盖 |
| `--zip` | 同时生成同名压缩包 |
| `--skip-verify` | 跳过隔离环境自检；仅在构建机缺少运行依赖时使用 |

构建完成后默认做一次隔离自检：在临时工作目录、只把产物加入 `PYTHONPATH` 的条件下
启动子进程，确认 `recognition` 来自产物并实际完成一次推理。缺少运行模块、模型或字典
会在这里直接失败，不会生成“看起来成功”的目录。

## 2. 产物结构

```text
dist/recognition_runtime/
├── recognition/            # 自包含 runtime 包：预处理、数据集、模型、解码与后端
├── models/model.onnx       # 构建时拷贝的模型
├── keys.txt                # 与模型配套的字典
├── requirements.txt        # numpy、pillow、opencv-python-headless、torch、torchvision
├── requirements-onnx.txt   # 基础依赖 + onnxruntime
├── requirements-openvino.txt
├── requirements-pytorch.txt
├── example.py              # 最小调用示例
├── README.md
└── manifest.json
```

产物中的 `recognition/` 只包含 `runtime/` 和空的 `__init__.py`，不包含训练数据集、
`components/`、`models/` 或 `utils/`。`runtime/` 内部全部使用相对导入，因此也可以
只把 `recognition/runtime/` 拷进其他 Python 项目并按包重命名（例如 `ocr_runtime`），
不需要仓库其余代码。

`manifest.json` 记录 `format_version`、构建时间、git commit、模型和字典的 SHA256，
以及 ONNX 导出元数据中的 `num_layers`、`num_channels`、`keys_sha256` 和 `output_type`。
OpenVINO IR 的 `.bin` 与 ONNX 外部数据 `.data` 会随模型一起复制。
部署方可用模型、字典哈希核对现场文件；ONNX 后端启动时还会按 `keys_sha256` 校验字典，
防止类别顺序不一致造成静默错字。

## 3. 安装依赖

按实际后端选择一份；CPU 部署优先 ONNX Runtime：

```bash
pip install -r requirements-onnx.txt       # ONNX Runtime
pip install -r requirements-openvino.txt   # OpenVINO
pip install -r requirements-pytorch.txt    # PyTorch checkpoint
```

runtime 目录自包含预处理、数据集、模型、解码和后端，不引用训练代码。基础清单包含
`torch`/`torchvision`，因为推理管线仍用它们做张量承载和 DataLoader；ONNX/OpenVINO
部署也需要安装。去掉这层依赖属于后续瘦身工作，不在当前产物范围内。

## 4. 调用

Python API：

```python
from recognition.runtime import RecognitionEngine, collect_image_paths

engine = RecognitionEngine(
    model_path="models/model.onnx",
    keys_path="keys.txt",
    backend="auto",     # auto | onnx | openvino | pytorch
    num_threads=1,
)
report = engine.recognize(
    collect_image_paths("/path/to/images"),
    batch_size=8,
    sort_by_width=True,
)
for result in report:
    print(result.path, result.text)
print(report.infer_ms, report.wall_seconds)
```

结果按传入图片顺序返回；`sort_by_width=True` 只改变组 batch 的顺序，不改变输出顺序。
路径按进程当前工作目录解析，引擎不会把相对路径自动拼到产物目录。

命令行：

```bash
python example.py /path/to/image_or_dir
python -m recognition.runtime \
  --model_path models/model.onnx \
  --keys_path keys.txt \
  --image_path /path/to/images \
  --batch_size 8
```

仓库内的 `recognition_inference.py` 和 `recognition_inference_layer.py` 仍保留原命令、
原参数和旧默认值，内部调用同一套 runtime；新接入应直接使用 `python -m recognition.runtime`
或产物内的 `example.py`。

### 直接嵌入其他 Python 项目

不需要 bundle 时，只复制 `recognition/runtime/` 即可。runtime 内部全部使用相对导入，
可以按目标项目的包名重命名，例如复制为 `my_project/ocr_runtime/`：

```python
from ocr_runtime import RecognitionEngine

engine = RecognitionEngine("model.onnx", "keys.txt", backend="onnx")
```

`python -m ocr_runtime --model_path ... --keys_path ... --image_path ...` 同样可用。
这种方式只带运行代码，模型、字典和依赖仍需由目标项目自行管理；需要可核对的产物时
仍建议使用 bundle。手动拷贝时排除 `__pycache__`，bundle 构建会自动清理。

## 5. 升级与回滚

- 模型、字典和运行代码必须同版本构建；不要单独替换产物里的模型或 `keys.txt`。
- 升级时重新运行构建脚本并保留上一个产物目录，回滚只需切回旧目录。
- 构建脚本只按白名单拷贝 `recognition/runtime/`，不会带出训练数据、`components/`、
  `models/` 或 `utils/`。
- `format_version` 变化表示产物布局或 manifest 字段不兼容，接入方需要同步调整。

## 6. 使用边界

- 输入是完整 detection 文字框；runtime 不二次裁剪正文。
- `keys.txt` 的内容和顺序属于模型 ABI，改字典必须重新训练、导出并重建产物。
- batch>1 依赖模型包含 `img_widths` 输入和 GRU 的 `sequence_lens`；旧版导出模型会打印
  警告，批量结果可能与逐张推理不一致。
- 带透明通道的 PNG 由共享预处理合成背景，不需要调用方先转 RGB。
- CPU 线程数建议在目标机器实测 1/2/4；不要直接把逻辑 CPU 总数当作最优值。
