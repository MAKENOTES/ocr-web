# OCR 阶段 4：Hard Case 数据管线实施计划

> **给智能执行者：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐项执行。本计划使用 checkbox（`- [ ]`）跟踪进度。

**目标：** 建立经过人工审核的 hard case 数据管线，覆盖菜单白字、小文字稀疏图、箭头/图标、当前失败样本。

**架构：** hard case 先进入独立 manifest。validator 检查 label、文件、字符集。merge 脚本生成可过采样的训练 manifest，不修改原始标注文件。

**技术栈：** Python、PIL、现有标注格式。

---

## 文件结构

- 新建：`recognition/data/hard_cases/README.md` — 标注规则。
- 新建：`recognition/data/hard_cases/hard_cases_annotation.txt` — 人工审核 manifest。
- 新建：`scripts/validate_hard_cases.py` — 校验 CLI。
- 新建：`scripts/merge_hard_cases.py` — 合并训练 manifest 生成器。
- 新建：`tests/test_hard_case_pipeline.py`。

---

### 任务 1：编写 hard case 标注规则

- [ ] 新建 `recognition/data/hard_cases/README.md`。
- [ ] 包含分类：`menu-white-text`、`small-text-sparse`、`has-arrow`、`current-failures`。
- [ ] 包含 label 策略：必须和业务输出完全一致；箭头是否纳入 label 必须保持一致。
- [ ] 提交：`git commit -m "docs: define OCR hard-case labeling rules"`。

### 任务 2：增加 hard case 校验器

- [ ] 在 `tests/test_hard_case_pipeline.py` 写测试，覆盖缺失文件和未知字符报告。
- [ ] 实现 `scripts/validate_hard_cases.py`，提供 `validate_hard_cases(annotation_path, keys_path)`。
- [ ] 报告必须包含：`total_rows`、`missing_files`、`empty_labels`、`unknown_chars`。
- [ ] CLI 在任意错误列表非空时 exit 1。
- [ ] 运行 `python -m pytest tests/test_hard_case_pipeline.py -q`。
- [ ] 提交：`git commit -m "feat: validate OCR hard-case annotations"`。

### 任务 3：初始化当前失败样本 manifest

- [ ] 新建 `recognition/data/hard_cases/hard_cases_annotation.txt`，只写入人工确认过的 label。
- [ ] 初始样本：

```text
../test_images1/menu-item-current_000.jpg,首页274
../test_images1/crop.png,首页74
../test_images1/menu-item_013.jpg,样品74
```

- [ ] 运行 `python scripts/validate_hard_cases.py --annotation recognition/data/hard_cases/hard_cases_annotation.txt --keys recognition/keys.txt`。
- [ ] 提交：`git commit -m "data: seed OCR hard-case manifest"`。

### 任务 4：合并 hard case 到训练 manifest

- [ ] 为 `merge_annotations(base, hard, output, hard_repeat=3)` 写测试，期望输出 1 行 base + 3 行 hard case。
- [ ] 实现 `scripts/merge_hard_cases.py`。
- [ ] 运行：

```bash
python -m pytest tests/test_hard_case_pipeline.py -q
python scripts/merge_hard_cases.py --base recognition/data/splits/train_annotation.txt --hard recognition/data/hard_cases/hard_cases_annotation.txt --output recognition/data/splits/train_with_hard_cases.txt --hard_repeat 3
```

- [ ] 提交：`git commit -m "feat: merge OCR hard cases into training data"`。

---

## 自检

- 覆盖 hard case label 规则、校验器、初始 manifest、过采样合并。
- 不修改原始标注。
- label 策略明确。

---

## 执行交接

计划完成。执行时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。
