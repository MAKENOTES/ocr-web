# 代码与文档核对待办

本文汇总代码与文档核对中尚未解决的问题，并列出后续检查方法和完成标准。

核对日期：2026-09-13。统一 `detection` 已独立实现严格数据契约；本页“小图”条目保留为历史 `detection_small` 的待办。本轮修订注释、帮助文字及文档，不改训练配置、模型算法、数据集或权重。
以下区分已确认的实现差距和仍需数据证明的结论；完成前不要把目标要求写成已有保障。

## 数据与评估正确性

- [x] **小图训练的验证集隔离。** 2026-11-11 已完成：[train.py](../detection_small/train.py)
  默认使用 `data/train_*` 和 `data/val_*`，并在启动时拒绝训练/验证图片目录相同；
  `run_config.json` 记录本次实际目录，报告会复核两者。样本交集仍按 `split_manifest.csv` 人工核对。
- [x] **小图标签严格加载。** 2026-11-11 已完成：[dataset.py](../detection_small/components/dataset.py)
  要求 `gt_<stem>.txt` 和 `<stem>.txt` 恰好一份，坏行、退化框、越界框和旋转 quad8 带文件和行号报错；
  `###` 进入 `ignored` 数组，训练掩码排除对应网格，不贡献置信度或几何梯度。
  现有 16517 个标签文件已用严格解析器全量扫描，无坏行。
- [x] **小图评估与部署解码对齐。** 2026-11-11 已完成：[eval_metrics.py](../detection_small/components/eval_metrics.py)
  改为四连通域、按输入面积缩放小区域阈值、保留浮点标签，并在原图坐标系匹配；
  训练期 F1、报告和 ONNX runtime 共用 [postprocess.py](../detection_small/components/postprocess.py)
  的解码/NMS。矩形输入 `256×768` 的小区域阈值为 3，NMS IoU 默认 0.3。
- [x] **小图透明图片一致性。** 2026-11-11 已完成：训练、训练期指标、报告和新的
  ONNX runtime 都使用 [image.py](../detection_small/components/image.py) 的 alpha 合成读取；
  旧 `detection_small_inference_640.py` 仍保留历史 OpenCV/Pillow 读取，但不再是推荐部署入口。
- [ ] **大图独立目录内的重复数据。** [train.py](../detection_larger/train.py) 只拒绝相同目录路径，
  不检查两目录中复制的同一图片。先只读生成图片内容摘要与标签配对清单，报告跨目录重复。
  完成标准：确认重复处理规则后再调整数据，独立验证指标不包含训练样本。
- [ ] **识别超宽输入的处理约定。** [preprocessing.py](../recognition/components/preprocessing.py) 在超出 max_width 时横向压缩；
  训练及训练期验证上限 1800，独立评估/推理默认无上限。
  先统计缩放后超过 1800 px 的数量，再比较两种路径的张量、有效长度和 CER。
  完成标准：选择并记录统一策略，覆盖超宽与窄图测试，再决定是否需要重新训练。
- [ ] **识别独立评估的样本标识与报告。** [evaluate_val.py](../recognition/evaluate_val.py) 用 stem 映射标签，
  同名不同路径会互相覆盖；空输入会除零；零错误时保留上次错误报告。
  用两个目录的同 stem 图片、空标注和先错后全对的两次评估复现。
  完成标准：预测与真值逐条对应、空输入明确报错、报告能确定属于本次评估。

## 字典与清理

- [ ] **清理范围与当前训练清单不一致。** 字符统计及两种清理工具默认只核对 train、val、synthesis1/2，
  [train.py](../recognition/train.py) 还使用 synthesis3 和五份专项标注。
  先对全部训练清单生成字符覆盖及图片引用报告，特别检查待删 keys 是否仍被其他清单使用。
  完成标准：明确清理应覆盖的范围，跨清单保留引用受到保护；未经核对不执行删除。
- [ ] **识别字典校验统一。** [LabelConverter](../recognition/components/converter.py) 未严格拒绝重复或多字符行；
  包内推理/评估只依靠权重形状，不能发现同长度字典的顺序错误。
  用重复字、多字符行和交换两个字符的字典验证各入口；完成标准为错误提前报出且不改变类别 ABI。
- [ ] **损坏图片预检。** recognition 全局允许 Pillow 加载截断图，桶预扫描异常先用宽高比 1。
  用截断 PNG/JPEG 验证哪些情况能被实际加载器捕获，再确定是否收紧全局开关或提前统一检查。
  完成标准：数据错误有具体路径，不产生未报告的损坏训练样本。

## 测试、入口与历史证据

- [x] **ONNX 裁片测试显式配置。** [test_onnx_entry_saves_images_labels_and_crops](../tests/test_detection_larger_onnx.py)
  已在测试中显式设置 `SAVE_CROPS=True`，产品入口默认值仍为关闭。
- [x] **小图导出成功条件。** 2026-11-11 已完成：[detection_small_export_onnx.py](../detection_small_export_onnx.py)
  改为临时文件导出，PyTorch/ONNX Runtime 数值校验失败时抛异常并删除临时文件，不覆盖目标；
  固定/动态高宽、batch 1/2、ONNX 元数据和动态标志一致性都有回归测试。
- [ ] **默认写入与命令行收敛。** 旧分隔符转换默认覆盖；部分入口还有固定常量的同义参数。
  本轮只如实标明行为。后续逐入口确认调用方和必要运行参数，补预览/执行回归测试后再改默认或删参数。
  完成标准：帮助、默认值、调用方、测试和文档同时更新，不破坏已使用的入口。
- [ ] **合成标注输出尚未统一。** [generate_lines.py](../recognition/utils/synthesis/generate_lines.py) 仍生成逗号版及
  `_annotation_training.txt` Tab 版，其他主要生成器已使用单份 Tab 标注。
  先核对现有调用方，再决定是否统一；完成标准：训练文档指向实际 Tab 产物，旧产物迁移可追溯。
- [ ] **未确认的标点与形近字规则。** 旧数据维护笔记中的 `已/己`、`杳/杏`、`间/问`、`日/目`、`余/佘`、
  `µ/μ`、双引号及“横杠是否归一为一”没有统一结论。
  结合原图、业务语义及现有 keys 逐条定标，不在加载器全局替换；完成标准为可复核的标注规则和样本清单。
- [ ] **恢复尾部实验依据。** 原架构引用的 `errdata20260912-tail-experiment.md`、
  `errdata20260912-tail-followup.md` 当前不存在。先从 Git 历史或原实验产物定位报告，
  核对权重、数据清单和评价口径；无法恢复时保留缺证据标记，不把历史文字当作当前验收依据。
- [ ] **历史精度和性能记录复核。** 历史文档中的模型速度倍数、数据数量、红章触发率和合成恢复数量，
  本轮没有重新训练或测量。需要用于模型选型时，记录权重/字典/数据摘要、预处理版本、CPU/GPU、线程及批量，
  在独立数据上报告 CER、整行准确率和完整流水线耗时；训练/验证参考图用于调参后不能再称独立发布测试集。

## 本轮验证

2026-09-13 基线：源码语法和去除 docstring 后的 AST 对照用于确认未更改执行逻辑；唯一预期的非注释变化是
`dictionary.py` 的帮助文字从旧 `train_annotation.txt` 修正为实际默认的 `append1_annotation.txt`。
全量测试命令 `.venv/bin/python -m pytest -q --tb=short`：348 通过、1 失败、4 条警告，
耗时 37.93 秒。失败仅为上方已记录的 ONNX 裁片测试开关，未出现收集错误。
本页待办均未作为算法修复执行。

2026-10-29 统一检测最终评估与日志调整：全量测试 `.venv/bin/python -m pytest -q` 为 528 通过、0 失败、14 条警告，
耗时 61.96 秒。新增词/短语拆分回归、训练期 F1 评估、F1 检查点和降频日志；历史检测测试保留。
ONNX 裁片测试已显式开启裁片并通过。

2026-11-11 小图检测训练与报告改造：新增 29 种按框安全增强、轮次课程、严格标签加载、
`###` 训练掩码、训练期 F1/P/R、F1 检查点、按运行目录保存检查和
[report_training.py](../detection_small/report_training.py) 静态 HTML 报告；评估解码改为与矩形部署入口
一致的四连通域和原图坐标匹配。新增测试 `tests/test_detection_small_augmentation.py`、
`tests/test_detection_small_report.py`，并保留 `tests/test_detection_rectangular_pipeline.py` 的
`ResizeAndPad` 兼容性。全量数据标签已用严格解析器扫描通过；本轮未重新训练或重新导出权重。

2026-11-11 小图检测 ONNX 部署改造：新增 `detection_small/runtime` 自包含部署包、
薄入口 `detection_small_inference_onnx.py`、`ocrsmall_config` 元数据契约、固定/动态高宽导出、
临时文件原子发布、runtime 同步和部署包构建工具；解码/NMS 统一到
`components/postprocess.py`，训练指标和报告同时使用。新增
`tests/test_detection_small_onnx.py`，覆盖导出往返、动态/固定、元数据错配、失败不发布、
runtime 一致性、同步状态和 bundle 自检。本轮未用真实大数据集重新导出部署模型。
