# OCR Web 仓库协作指南

## 项目范围

这是一个两阶段 OCR 项目：`detection` 定位文字区域，`recognition` 将文字框识别为 Unicode 文本。根目录中的脚本负责主要的推理、导出和量化入口。

## 目录与入口

- `detection/`：EAST 风格检测模型、数据集、损失、训练和历史推理实现。
- `recognition/`：CRNN 识别模型、统一图片预处理、训练、评估和历史记录。
- `detection_inference_640.py`：当前矩形检测推理入口。
- `recognition_inference.py`：PyTorch、ONNX、OpenVINO 多后端识别入口。
- `detection_export_onnx.py`、`recognition_export_onnx.py`：模型导出入口。
- `docs/recogniton/recognition.md`：Recognition 当前架构和数据契约；`recognition/train.md` 只保存历史训练记录。
- `tests/`：检测数据划分、矩形检测链路和识别预处理测试。

## 开发环境

- 使用 Python 3.10 和仓库内的 `.venv`。
- 安装运行依赖：`.venv/bin/python -m pip install -r requirements.txt`；运行测试前另需安装 `pytest`（当前未列入 `requirements.txt`）。
- 优先用 `.venv/bin/python` 运行脚本，避免调用到系统环境中的不同版本依赖。
- 完整训练、模型量化和大数据集评估会消耗较长时间或 GPU 资源，除非任务明确要求，否则只运行针对性单元测试和轻量检查。

## 常用检查

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_recognition_preprocess.py -q
.venv/bin/python -m pytest tests/test_recognition_batch_invariance.py -q
.venv/bin/python -m pytest tests/test_detection_rectangular_pipeline.py -q
.venv/bin/python -m pytest tests/test_detection_split_dataset.py -q
.venv/bin/python -m py_compile recognition/components/dataset.py recognition/models/model.py recognition/train.py recognition/evaluate_val.py recognition_export_onnx.py recognition_inference.py
```

改动哪个子系统，至少运行对应的测试文件。涉及共享预处理、张量 layout、动态 padding 或模型输入通道时，还要运行全部测试。

## 数据契约

- Recognition 训练图片和标注路径统一使用 `.png`。标注格式是 `relative/path.png, "文本"`，路径相对于标注文件所在目录。
- 标注路径必须与磁盘文件精确一致，不要增加扩展名猜测、同名文件回退或缺图 dummy tensor。
- Recognition 使用完整 detection 文字框；训练、验证和推理共享按局部叠印关系去红章、等比缩放与动态 padding，不做二次文字裁剪；去章联合纸面背景、正文穿插和错位关系判断，逐连通区域保护同行红字及无正文支持的红色内容，禁止全图无差别去红或膨胀删除掩码。
- 新识别模型默认使用 3 通道 RGB。带透明通道的 PNG 必须通过共享的 `load_rgb_image` 合成背景，避免黑字或白字消失。
- `img_widths` 表示不含 CNN 右侧保护区和 batch padding 的真实像素宽度。collate 必须为每张图片保留固定的右侧背景保护区，BiGRU 必须通过 `PackedSequence` 跳过 padding；修改 CNN 下采样或 padding 时，必须同步检查 CTC 序列长度和 batch 一致性。
- `recognition/keys.txt` 的字符内容与顺序属于模型 ABI；修改后必须重新训练、导出并验证模型。
- Detection 图片与标签按 stem 配对，标签可以是 `gt_<stem>.txt` 或 `<stem>.txt`。数据划分以 manifest 为准，不能让同一样本同时进入 train 和 val。

## 修改原则

- 优先复用 `recognition/components/dataset.py` 中的加载和预处理逻辑，确保训练、评估、PyTorch、ONNX 和 OpenVINO 行为一致。
- 图片或标注错误应尽早给出包含具体路径的异常，不能静默跳过会污染训练的数据。
- 修复当前主链路时不要顺手重写带版本号的历史实验脚本；只有在任务明确涉及它们时才同步修改。
- 行为发生变化时同步更新测试和当前架构文档；历史训练结果保留原意，并明确标注为历史记录。
- 保持现有中文注释和文档风格；新增 Python 标识符、类型标注和异常信息应清晰、具体。

## 数据与产物

`detection/data/`、`recognition/data/`、训练日志、checkpoints、推理输出和大模型产物都不应提交。不要删除、移动或批量改写用户的本地数据集和权重，除非任务明确要求并已核对目标路径。
