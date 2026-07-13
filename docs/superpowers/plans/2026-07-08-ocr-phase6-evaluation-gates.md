# OCR 阶段 6：评测门禁实施计划

> **给智能执行者：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 按任务逐项执行。本计划使用 checkbox（`- [ ]`）跟踪进度。

**目标：** 建立固定 OCR 发布门禁，让后续模型质量由可重复指标判断，而不是靠少量手工样例。

**架构：** 定义 JSON 指标 schema，增加 gate checker CLI，把分组评估输出接入 schema，并提供一个 CI 友好的统一检查命令。

**技术栈：** Python、JSON、shell script、前面阶段的数据审计/评估脚本。

---

## 文件结构

- 新建：`recognition/evaluation/metrics_schema.md`。
- 新建：`scripts/check_ocr_release_gates.py`。
- 新建：`tests/test_ocr_release_gates.py`。
- 修改：`scripts/evaluate_recognition_groups.py`。
- 新建：`scripts/check_ocr_quality.sh`。

---

### 任务 1：定义指标 schema

- [ ] 新建 `recognition/evaluation/metrics_schema.md`。
- [ ] 要求 JSON 字段：`overall`、`menu-white-text`、`small-text-or-sparse`、`has-arrow`、`long-text`、`batch_invariance`、`data_quality`。
- [ ] 每个识别分组包含：`accuracy`、`cer`、`count`。
- [ ] 门禁阈值：

```text
batch_invariance.match_rate >= 1.0
menu-white-text.accuracy >= 0.98
small-text-or-sparse.accuracy >= 0.95
overall.accuracy >= 0.97
data_quality.duplicate_leakage == 0
data_quality.conflict_labels == 0
```

- [ ] 提交：`git commit -m "docs: define OCR release metric schema"`。

### 任务 2：增加 release gate checker

- [ ] 新建 `tests/test_ocr_release_gates.py`，覆盖：指标通过、`menu-white-text` 不达标、duplicate leakage 不为 0。
- [ ] 新建 `scripts/check_ocr_release_gates.py`，实现 `check_gates(metrics: dict) -> list[str]`。
- [ ] CLI 支持 `--metrics`，打印失败项；任意 gate 失败时 exit 1。
- [ ] 运行 `python -m pytest tests/test_ocr_release_gates.py -q`。
- [ ] 提交：`git commit -m "feat: add OCR release gate checker"`。

### 任务 3：生成符合门禁 schema 的评估报告

- [ ] 修改 `scripts/evaluate_recognition_groups.py`，确保输出包含：

```python
report["batch_invariance"] = {"match_rate": batch_match_rate, "count": batch_count}
report["data_quality"] = {"duplicate_leakage": duplicate_leakage, "conflict_labels": conflict_labels}
```

- [ ] 生成报告：

```bash
python scripts/evaluate_recognition_groups.py \
  --model_path best_recognition.onnx \
  --backend onnx \
  --annotation recognition/data/splits/test_annotation.txt \
  --audit_report recognition/data/reports/audit_summary.json \
  --output recognition/data/reports/release_gate_report.json
```

- [ ] 运行 `python scripts/check_ocr_release_gates.py --metrics recognition/data/reports/release_gate_report.json`。
- [ ] 提交：`git commit -m "feat: generate OCR release gate report"`。

### 任务 4：增加 CI 友好的统一命令

- [ ] 新建可执行脚本 `scripts/check_ocr_quality.sh`，依次运行数据审计、分组评估、release gate checker。
- [ ] 运行 `chmod +x scripts/check_ocr_quality.sh`。
- [ ] 运行 `./scripts/check_ocr_quality.sh`。
- [ ] 提交：`git commit -m "chore: add OCR quality gate command"`。

---

## 自检

- 覆盖固定指标、硬阈值、数据泄漏、冲突标签、batch 一致性、场景准确率。
- schema、checker、evaluator 的指标 key 保持一致。
- gate 命令可用于 CI。

---

## 执行交接

计划完成。执行时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。
