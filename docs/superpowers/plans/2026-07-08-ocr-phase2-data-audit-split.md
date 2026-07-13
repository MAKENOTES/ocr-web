# OCR 阶段 2：数据审计与重新划分实施计划

> **给智能执行者：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐项执行。本计划使用 checkbox（`- [ ]`）跟踪进度。

**目标：** 重建 OCR 的 train/validation/test 标注清单，消除 exact duplicate 泄漏和相同图片不同 label 的冲突。

**架构：** 原始图片不动。新增脚本解析标注、计算像素哈希、输出冲突报告、标记业务场景，并把干净划分写到 `recognition/data/splits/`。

**技术栈：** Python 标准库、PIL、hashlib、CSV/JSON。

---

## 文件结构

- 新建：`recognition/utils/data_audit.py` — 标注解析、像素哈希、重复/冲突检测、场景标签。
- 新建：`scripts/audit_recognition_data.py` — JSON 审计报告 CLI。
- 新建：`scripts/build_recognition_splits.py` — 无泄漏数据划分生成器。
- 新建：`tests/test_data_audit.py` — 工具函数测试。
- 修改：`recognition/train.py` — 验证集改用 `recognition/data/splits/val_annotation.txt`。

---

### 任务 1：增加标注解析和像素哈希

- [ ] 写测试：`parse_annotation_file()` 能保留 label 中的逗号；`image_sha256()` 基于像素内容生成哈希。
- [ ] 在 `recognition/utils/data_audit.py` 实现 `AnnotationRow`、`parse_annotation_file(annotation_path: Path)`、`image_sha256(path: Path)`。
- [ ] 运行 `python -m pytest tests/test_data_audit.py -q`。
- [ ] 提交：`git commit -m "feat: add OCR data audit utilities"`。

### 任务 2：检测重复泄漏和 label 冲突

- [ ] 增加 `DuplicateGroup` 和 `find_duplicate_groups(rows)` 测试，使用两张像素相同但 label 不同的图片。
- [ ] 实现 `DuplicateGroup.has_conflict`，按 `image_sha256()` 分组。
- [ ] 运行 `python -m pytest tests/test_data_audit.py -q`。
- [ ] 提交：`git commit -m "feat: detect OCR duplicate label conflicts"`。

### 任务 3：增加业务场景标签

- [ ] 增加 `menu-white-text`、`small-text-or-sparse`、`has-arrow`、`long-text` 的测试。
- [ ] 实现 `infer_scenario_tags(label, width, height, white_ratio)`。
- [ ] 运行 `python -m pytest tests/test_data_audit.py -q`。
- [ ] 提交：`git commit -m "feat: tag OCR samples by scenario"`。

### 任务 4：增加审计 CLI

- [ ] 新建 `scripts/audit_recognition_data.py`，支持 `--annotations` 和 `--output` 参数。
- [ ] 输出 JSON 字段：`total_rows`、`missing_files`、`duplicate_groups`、`conflict_groups`、`scenario_counts`、`conflicts`。
- [ ] 运行：

```bash
python scripts/audit_recognition_data.py \
  --annotations recognition/data/train_annotation.txt recognition/data/val_annotation.txt \
  --output recognition/data/reports/audit_summary.json
```

- [ ] 提交：`git commit -m "feat: add OCR dataset audit report"`。

### 任务 5：生成干净数据划分

- [ ] 新建 `scripts/build_recognition_splits.py`，按像素哈希分组后做确定性 80/10/10 划分。
- [ ] exact duplicate 必须留在同一个 split 中。
- [ ] 冲突重复组写入 `recognition/data/reports/conflicts_to_fix.csv` 并从训练划分中排除。
- [ ] 生成：

```text
recognition/data/splits/train_annotation.txt
recognition/data/splits/val_annotation.txt
recognition/data/splits/test_annotation.txt
```

- [ ] 修改 `recognition/train.py`：

```python
TRAIN_ANNOTATION = "recognition/data/splits/train_annotation.txt"
VALID_ANNOTATION = "recognition/data/splits/val_annotation.txt"
```

- [ ] 运行 `python -m py_compile scripts/build_recognition_splits.py recognition/train.py`。
- [ ] 提交：`git commit -m "feat: build leakage-free OCR data splits"`。

---

## 自检

- 覆盖 exact duplicate 泄漏、冲突 label、场景标签、`train.py` 验证集路径。
- 原始数据不被修改。
- 输出路径明确。

---

## 执行交接

计划完成。执行时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。
