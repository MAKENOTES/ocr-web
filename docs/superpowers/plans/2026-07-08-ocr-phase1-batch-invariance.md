# OCR 阶段 1：批次一致性实施计划

> **给智能执行者：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐项执行。本计划使用 checkbox（`- [ ]`）跟踪进度。

**目标：** 同一张 OCR 图片在 `batch_size=1` 和混合宽度 batch 推理时，必须解码出完全一致的文本。

**架构：** 先加回归测试，再修 `CRNN.forward()` 中的序列长度处理；随后让 CTC 解码布局显式化，并用 batch size 和类别数规范后端输出布局，移除不可靠的 shape 启发式判断。

**技术栈：** Python、PyTorch、ONNX Runtime、OpenVINO、现有 `recognition` 包。

---

## 文件结构

- 修改：`recognition/models/model.py` — 实现长度感知的 RNN 处理。
- 修改：`recognition/components/converter.py` — 增加显式 `layout="TBC" | "BTC"` 解码。
- 修改：`recognition_inference.py` — 确定性处理后端输出布局。
- 新建：`recognition/utils/tensor_layout.py` — 共享输出布局规范化工具。
- 新建：`tests/test_batch_invariance.py` — 同图 batch 一致性回归测试。
- 新建：`tests/test_converter_layout.py` — 解码布局测试。
- 新建：`tests/test_tensor_layout.py` — 后端输出布局测试。

---

### 任务 1：增加 batch invariance 回归测试

- [ ] 新建 `tests/test_batch_invariance.py`，测试 `recognition/data/test_images1/menu-item-current_000.jpg` 单独推理和与 `recognition/data/test_images1/crop.png` 混合 batch 推理时，第一个图片的解码文本必须一致。
- [ ] 运行 `python -m pytest tests/test_batch_invariance.py -q`，确认修复前失败。
- [ ] 提交：`git commit -m "test: add OCR batch invariance regression"`。

### 任务 2：让 decoder 布局显式化

- [ ] 新建 `tests/test_converter_layout.py`，包含两个测试：`[T, B, C]` 和 `[B, T, C]`。
- [ ] 修改 `recognition/components/converter.py` 中的 `LabelConverter.decode(self, preds, layout="TBC")`。
- [ ] 如果 `layout == "TBC"`，转成 `[B, T, C]`；如果 `layout == "BTC"`，保持不变；其它值抛 `ValueError`。
- [ ] 修改 `recognition/train.py`、`recognition/evaluate_val.py`、`recognition_inference.py` 的调用，使用 `converter.decode(preds, layout="TBC")`。
- [ ] 运行 `python -m pytest tests/test_converter_layout.py -q`。
- [ ] 提交：`git commit -m "fix: make CTC decode layout explicit"`。

### 任务 3：修复 BiGRU 长度处理

- [ ] 在 `recognition/models/model.py` 中，用 `torch.nn.utils.rnn.pack_padded_sequence` 替换 “GRU 之前将 padding 区域置零” 的逻辑。
- [ ] 使用 `_calc_cnn_output_width()` 计算每张图片的真实序列长度。
- [ ] GRU 后使用 `pad_packed_sequence(..., total_length=seq_len)` 恢复序列。
- [ ] 输出 mask 只在 `padding_mask is not None` 时执行。
- [ ] 运行 `python -m pytest tests/test_batch_invariance.py -q`。
- [ ] 运行 `python -m py_compile recognition/models/model.py`。
- [ ] 提交：`git commit -m "fix: make CRNN RNN length-aware"`。

### 任务 4：规范后端输出布局

- [ ] 新建 `recognition/utils/tensor_layout.py`，实现 `ensure_tbc_output(preds, batch_size, num_classes)`。
- [ ] 新建 `tests/test_tensor_layout.py`，覆盖：保留 `[T, B, C]`、转置 `[B, T, C]`、拒绝非法 shape。
- [ ] 在 `recognition_inference.py` 中，decode 前调用 `ensure_tbc_output()`。
- [ ] 删除 OpenVINO 中的启发式判断：`if out.ndim == 3 and out.shape[0] < out.shape[1]`。
- [ ] 运行：

```bash
python -m pytest tests/test_tensor_layout.py tests/test_converter_layout.py tests/test_batch_invariance.py -q
python recognition_inference.py --backend onnx --model_path best_recognition.onnx --image_path recognition/data/test_images1 --batch_size 1
python recognition_inference.py --backend onnx --model_path best_recognition.onnx --image_path recognition/data/test_images1 --batch_size 32
```

- [ ] 期望：batch 1 和 batch 32 对同一图片输出一致。
- [ ] 提交：`git commit -m "fix: normalize OCR backend output layout"`。

---

## 自检

- 覆盖 BiGRU padding 根因、回归测试、decoder 布局、后端输出布局。
- 没有模糊或延后任务。
- 模型和后端输出统一使用 `layout="TBC"`。

---

## 执行交接

计划完成。执行时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。
