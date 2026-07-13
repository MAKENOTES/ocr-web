# OCR 阶段 3：预处理接入训练实施计划

> **给智能执行者：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐项执行。本计划使用 checkbox（`- [ ]`）跟踪进度。

**目标：** 让训练阶段也见到推理阶段白字裁剪所覆盖的 UI 白字、小文字、大背景分布。

**架构：** 确定性裁剪保留在 `recognition/utils/white_text_crop.py`。训练阶段在 `recognition/components/dataset.py` 增加随机 UI padding/background 增强。验证和推理保持确定性。

**技术栈：** PIL、NumPy、TorchVision transforms、现有 dataset pipeline。

---

## 文件结构

- 修改：`recognition/utils/white_text_crop.py` — 暴露 bbox helper。
- 修改：`recognition/components/dataset.py` — 增加 `RandomUIPadding` 和 dataset 开关。
- 修改：`recognition/train.py` — 只对训练 loader 启用增强。
- 修改：`recognition/doc/train.md` — 记录预处理契约。
- 新建：`tests/test_preprocess_training_integration.py`。

---

### 任务 1：暴露白字 bbox helper

- [ ] 为 `find_white_text_bbox()` 写测试：没有白字时返回 `None`；有白字时返回 `(left, top, right, bottom)`。
- [ ] 在 `recognition/utils/white_text_crop.py` 实现 `find_white_text_bbox()`。
- [ ] 重构 `auto_crop_white_text_image()` 使用该 helper，但保持当前裁剪输出不变。
- [ ] 运行 `python -m pytest tests/test_preprocess_training_integration.py tests/test_recognition_inference_crop.py -q`。
- [ ] 提交：`git commit -m "refactor: expose white text bbox helper"`。

### 任务 2：增加 UI padding 增强

- [ ] 增加测试：从 `recognition.components.dataset` 导入 `RandomUIPadding`，验证图像 mode 保持不变，canvas 尺寸不缩小。
- [ ] 实现 `RandomUIPadding(p=0.35, min_scale=0.55, max_scale=0.9, background=(30, 100, 170))`。
- [ ] 增强逻辑：随机缩小文本图，贴到更大的有色背景画布中，返回 PIL image。
- [ ] 运行 `python -m pytest tests/test_preprocess_training_integration.py -q`。
- [ ] 提交：`git commit -m "feat: add UI padding augmentation"`。

### 任务 3：只在训练中接入增强

- [ ] 给 `BaseTextDataset`、`TextRecognitionDataset`、`create_dataloader()` 增加 `use_ui_augmentation=False` 参数。
- [ ] 在训练 transform 中，仅当 `is_train=True` 且 `use_ui_augmentation=True` 时前置 `RandomUIPadding(...)`。
- [ ] 在 `recognition/train.py` 增加 `USE_UI_AUGMENTATION = True`。
- [ ] train loader 传 `use_ui_augmentation=USE_UI_AUGMENTATION`，validation loader 传 `False`。
- [ ] 运行：

```bash
python -m py_compile recognition/components/dataset.py recognition/train.py
python - <<'PY'
from recognition.components.dataset import TextRecognitionDataset
D = TextRecognitionDataset('recognition/data/train_annotation.txt', 32, 1800, 1, is_train=True, use_ui_augmentation=True)
img, label = D[0]
print(tuple(img.shape), label[:20])
PY
```

- [ ] 提交：`git commit -m "feat: enable UI augmentation for OCR training"`。

### 任务 4：记录预处理契约

- [ ] 在 `recognition/doc/train.md` 追加说明：推理使用确定性 white-text crop；训练使用随机 UI padding 增强。
- [ ] 明确说明 validation 不允许使用随机增强。
- [ ] 提交：`git commit -m "docs: document OCR preprocessing contract"`。

---

## 自检

- 覆盖训练/推理分布对齐和白字 UI 增强。
- 验证集保持确定性。
- 不修改原始数据。

---

## 执行交接

计划完成。执行时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。
