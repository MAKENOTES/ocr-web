# OCR 阶段 5：输入高度与模型消融实施计划

> **给智能执行者：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐项执行。本计划使用 checkbox（`- [ ]`）跟踪进度。

**目标：** 在切换更重模型之前，先验证 `IMG_HEIGHT=48` 是否能改善小 UI 文本识别。

**架构：** 先让高度可配置，再验证 CRNN 支持 48px 输入；随后在相同 clean split 上做 32 vs 48 对照实验，并按场景评估。

**技术栈：** Python、PyTorch、现有 CRNN 和训练脚本。

---

## 文件结构

- 修改：`recognition/train.py` — config dataclass 和 CLI 参数。
- 修改：`recognition/models/model.py` — 必要时支持 height 48。
- 修改：`recognition_inference.py` — 如果导出模型高度不同，推理高度也要可配置。
- 新建：`scripts/run_height_ablation.sh`。
- 新建：`scripts/evaluate_recognition_groups.py`。
- 新建：`tests/test_img_height_config.py`。

---

### 任务 1：让训练高度可配置

- [ ] 在 `recognition/train.py` 新建 `RecognitionTrainConfig` dataclass，字段包括 `img_height`、`img_width`、`batch_size`、`epochs`、`experiment_name`。
- [ ] 增加测试：默认高度是 32；自定义高度 48 可被接受。
- [ ] 保留兼容常量，并让它们从 config 读取。
- [ ] 运行 `python -m pytest tests/test_img_height_config.py -q`。
- [ ] 提交：`git commit -m "feat: add OCR training config"`。

### 任务 2：增加训练 CLI 参数

- [ ] 在 `recognition/train.py` 增加 `parse_args()`，支持 `--img_height`、`--experiment_name`、`--epochs`、`--batch_size`。
- [ ] 在 `main()` 中用 args 构建 `RecognitionTrainConfig`。
- [ ] model、dataloader、training loop 使用 `config.img_height`、`config.batch_size`、`config.epochs`。
- [ ] 运行 `python -m py_compile recognition/train.py`。
- [ ] 提交：`git commit -m "feat: expose OCR training height CLI"`。

### 任务 3：验证 CRNN 支持 height 48

- [ ] 增加测试：创建 `CRNN(img_height=48, nc=1, n_class=10)`，forward 输入 `[2, 1, 48, 160]`。
- [ ] 如果 CNN 输出高度不是 1，则在 feature extractor 中加入 adaptive height pooling。
- [ ] 运行 `python -m pytest tests/test_img_height_config.py -q`。
- [ ] 提交：`git commit -m "feat: support OCR input height ablation"`。

### 任务 4：增加消融实验脚本

- [ ] 新建可执行脚本 `scripts/run_height_ablation.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
python recognition/train.py --img_height 32 --experiment_name ocr_h32 --epochs 120 --batch_size 64
python recognition/train.py --img_height 48 --experiment_name ocr_h48 --epochs 120 --batch_size 48
```

- [ ] 运行 `chmod +x scripts/run_height_ablation.sh`。
- [ ] 提交：`git commit -m "chore: add OCR height ablation runner"`。

### 任务 5：按场景评估

- [ ] 新建 `scripts/evaluate_recognition_groups.py`。
- [ ] 输出 JSON 分组：`overall`、`menu-white-text`、`small-text-or-sparse`、`has-arrow`、`long-text`。
- [ ] 每个分组必须包含：`accuracy`、`cer`、`count`。
- [ ] 分别评估 h32 和 h48，输出到 `recognition/data/reports/eval_h32.json` 和 `recognition/data/reports/eval_h48.json`。
- [ ] 提交：`git commit -m "feat: evaluate OCR height ablations by scenario"`。

---

## 自检

- 先测试高度配置，再改模型。
- 先评估 48px，再考虑更重架构。
- 使用阶段 2 的场景指标。

---

## 执行交接

计划完成。执行时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。
