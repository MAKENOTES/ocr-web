# OCR Web 仓库协作指南

本文规定 OCR Web 仓库的协作方式、数据契约、代码与文档规范，以及修改和测试要求。

## 项目范围

这是一个两阶段 OCR 项目：`detection` 统一检测 UI 小图、PDF 页面和网页表格文字，`recognition` 将文字框识别为 Unicode 文本。`detection_small` 和 `detection_larger` 是迁移前的两个独立检测系统，只作为历史实现保留。根目录中的脚本负责统一检测和识别的主要推理、导出入口，历史小图的量化入口保留在 `detection_small_quantize_int8.py`。

## 目录与入口

- `detection/`：当前统一检测包。ResNet18 EAST 模型、严格数据加载、真值/损失、训练、推理和评估都在包内独立实现，不导入 `detection_small`、`detection_larger` 或 `recognition`；`detection/inference.py` 面向 PyTorch/GPU 推理，根目录 `detection_inference_onnx.py` 只保留部署常量，ONNX CPU 推理实现在 `detection/runtime/`（生成快照由 `tools/sync_detection_runtime.py` 维护），ONNX 导出入口在根目录 `detection_export_onnx.py`，独立部署包由 `tools/build_detection_bundle.py` 生成。
- `detection/components/`：只放训练链路需要的模块，包括图片读取、推理预处理、数据加载、增强、变换、真值、损失和训练期指标；推理后处理、ONNX/OpenVINO 运行时、评估报告等放到 `detection/utils/`。
- `tools/sync_detection_runtime.py`、`tools/build_detection_bundle.py`：从仓库源文件同步 `detection/runtime` 并组装可独立部署的检测运行目录，不参与训练和常规推理。
- `tools/sync_detection_small_runtime.py`、`tools/build_detection_small_bundle.py`：同步 `detection_small/runtime` 并组装小图检测独立部署目录。
- `detection_small/`：迁移前的 UI 小图检测，保留独立实现、数据和历史脚本，不再作为主链路；训练入口 `train.py`、报告入口 `report_training.py` 和 ONNX 部署 runtime 仍维护。
- `detection_larger/`：迁移前的 PDF 页面和网页表格检测，保留原有实现、数据和历史脚本，不再作为主链路。
- `recognition/`：CRNN 识别模型、统一图片预处理、训练、评估和历史记录。
- `detection_export_onnx.py`、`detection_inference_onnx.py`：统一检测的 ONNX 导出和独立 ONNX Runtime/OpenVINO CPU 推理入口。
- `detection_small_inference_onnx.py`：小图检测薄 ONNX CPU 推理入口，实现位于 `detection_small/runtime/`。
- `detection_small_inference_640.py`、`detection_small_export_onnx.py`、`detection_small_quantize_int8.py`：小图检测多后端历史入口、ONNX 导出和 INT8 量化入口。
- `detection_larger_export_onnx.py`、`detection_larger_inference_onnx.py`：历史大图检测入口。
- `recognition_inference.py`：PyTorch、ONNX、OpenVINO 多后端识别入口。
- `docs/detection/architecture.md`：统一检测的当前架构和数据契约；`docs/detection_small/`、`docs/detection_larger/` 保留历史说明。
- `tools/prepare_unified_detection_data.py`：一次性数据迁移工具，只读取两个历史检测目录并写入 `detection/data/`，不属于统一检测运行时。
- `tests/`：统一检测、历史检测链路和识别预处理测试。

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
.venv/bin/python -m pytest tests/test_detection_unified.py -q
.venv/bin/python -m pytest tests/test_detection_augmentation.py -q
.venv/bin/python -m pytest tests/test_detection_unified_onnx.py -q
.venv/bin/python -m pytest tests/test_detection_rectangular_pipeline.py -q
.venv/bin/python -m pytest tests/test_detection_small_augmentation.py tests/test_detection_small_report.py tests/test_detection_small_onnx.py -q
.venv/bin/python -m pytest tests/test_detection_larger.py -q
.venv/bin/python -m py_compile detection_small/train.py detection_small/report_training.py detection_small/components/*.py detection_small/models/*.py detection_small/utils/*.py detection_small/runtime/*.py detection_small_export_onnx.py detection_small_inference_onnx.py tools/sync_detection_small_runtime.py tools/build_detection_small_bundle.py
.venv/bin/python -m py_compile detection/train.py detection/inference.py detection/report_training.py detection/components/*.py detection/models/*.py detection/runtime/*.py detection/utils/*.py detection_export_onnx.py detection_inference_onnx.py tools/sync_detection_runtime.py tools/build_detection_bundle.py recognition/components/dataset.py recognition/models/model.py recognition/train.py recognition/evaluate_val.py recognition_export_onnx.py recognition_inference.py
```

改动哪个子系统，至少运行对应的测试文件。涉及共享预处理、张量 layout、动态 padding 或模型输入通道时，还要运行全部测试。

## 数据契约

- Recognition 训练图片和标注路径统一使用 `.png`。标注格式是 `relative/path.png<TAB>文本`（`<TAB>` 为真实制表符），路径相对于标注文件所在目录；加载器不自动剥离正文引号或归一化标点。
- 标注路径必须与磁盘文件精确一致，不要增加扩展名猜测、同名文件回退或缺图 dummy tensor。
- Recognition 使用完整 detection 文字框；训练、验证和推理共享按局部叠印关系去红章、等比缩放与动态 padding，不做二次文字裁剪；去章联合纸面背景、正文穿插和错位关系判断，逐连通区域保护同行红字及无正文支持的红色内容，禁止全图无差别去红或膨胀删除掩码。
- 新识别模型默认使用 3 通道 RGB。带透明通道的 PNG 必须通过共享的 `load_rgb_image` 合成背景，避免黑字或白字消失。
- `img_widths` 表示不含 CNN 右侧保护区和 batch padding 的真实像素宽度。collate 必须为每张图片保留固定的右侧背景保护区，BiGRU 必须通过 `PackedSequence` 跳过 padding；修改 CNN 下采样或 padding 时，必须同步检查 CTC 序列长度和 batch 一致性。
- `recognition/keys.txt` 的字符内容与顺序属于模型 ABI；修改后必须重新训练、导出并验证模型。
- Detection 图片与标签按 stem 配对，标签可以是 `gt_<stem>.txt` 或 `<stem>.txt`，同一张图片必须恰好存在一份标签；缺失或歧义立即报错。支持水平 `bbox4` 和水平 `quad8`，旋转四边形直接拒绝；`###` 表示忽略区域，空标签表示已确认的无文字图片。
- 统一 `detection/` 的数据位于 `detection/data/{train,val}_{images,labels}`，由 `tools/prepare_unified_detection_data.py` 从两个历史检测目录按图片内容 MD5 去重后生成，并用 `split_manifest.csv` 固定归属。同一内容不能同时进入 train 和 val；新增数据应重新生成清单或按同样规则追加，禁止把验证图片复制进训练目录。
- 统一检测训练使用固定输入尺寸，默认 768×768，可由 `INPUT_SIZE` 改为 `(高, 宽)`；推理默认 `DYNAMIC_INPUT=False`，使用同样的固定方形画布。`DYNAMIC_INPUT=True` 时按原图宽高比选择 32 的倍数尺寸，但超宽图短边会被压到 64、显著降低细条指标，只建议实验使用。ONNX 导出默认固定高宽（`DYNAMIC_SPATIAL=False`），动态导出时对应动态高宽；固定尺寸导出的 ONNX/OpenVINO 模型自动使用导出尺寸。
- 统一检测训练增强集中在 `detection/components/augmentation.py`，默认 30 种；几何增强在原图执行，光度/遮挡增强在缩放补边后的固定输入执行。几何增强后必须裁剪四边形并删除退化框，旋转/透视/剪切不得进入默认流程。验证集不启用随机增强，批次强度课程只在训练集生效。
- 统一检测最终框粒度为词/短语：整行 GT 必须在数据层合并为一致粒度，解码器默认按低占用列拆分粘连连通域；`split_word_gaps=False` 仅用于对照或兼容旧模型。训练默认 `BATCH_SIZE=8`、`NUM_WORKERS=8`，batch 翻倍时同步调整学习率并重新核对显存。
- `detection_small/` 和 `detection_larger/` 的数据契约按各自 README/架构文档执行；历史目录中的重复样本和加载差异不自动同步到统一检测。

## 修改原则

- **三个检测子系统必须隔离**：`detection`、`detection_small`、`detection_larger` 的代码、配置、数据、权重、训练、推理、导出和测试分别维护，不允许相互导入、引用或通过根目录入口间接依赖。统一 `detection` 还禁止导入或读取 `recognition` 的任何文件；透明图片合成、缩放和归一化必须在包内独立实现。修改任一子系统不得要求其他子系统同步修改或提供运行资源。
- **部署快照必须同步**：修改 `detection/components/image.py`、`detection/components/preprocessing.py`、`detection/utils/postprocess.py` 或 `detection/utils/config.py` 后，必须运行 `tools/sync_detection_runtime.py` 并提交生成的 `detection/runtime` 文件；部署包统一用 `tools/build_detection_bundle.py` 构建，不能手工拼接 runtime 与入口。
- **小图部署快照必须同步**：修改 `detection_small/components/image.py`、`components/preprocessing.py`、`components/postprocess.py` 或 `utils/config.py` 后，必须运行 `tools/sync_detection_small_runtime.py` 并提交生成的 `detection_small/runtime` 文件；部署包统一用 `tools/build_detection_small_bundle.py` 构建。
- 统一 `detection` 内部优先复用自身的 `components/preprocessing.py`、`components/transforms.py`、`components/image.py` 和 `utils/postprocess.py`，确保训练、评估、PyTorch、ONNX 和 OpenVINO 使用同一套加载、缩放、补边和归一化。
- 图片或标注错误应尽早给出包含具体路径的异常，不能静默跳过会污染训练的数据。
- 修复当前主链路时不要顺手重写带版本号的历史实验脚本；只有在任务明确涉及它们时才同步修改。
- 行为发生变化时同步更新测试和当前架构文档；历史训练结果保留原意，并明确标注为历史记录。
- 注释和文档使用中文，遵循下方表达规范；新增 Python 标识符、类型标注和异常信息应清晰、具体。

## 注释与文档

- **主动同步，无需提醒**：任何代码改动完成后，必须主动检查并同步受影响的文档、注释、docstring 和命令示例，不需要用户在每次任务中提醒。
- **先说结论**：先写用途、结果或建议，再补必要依据；一句话能说清楚就不写一段，避免重复、口号和无关背景。
- **不确定也要可行动**：无法下结论时，明确已知事实、缺少的证据，并给出具体检查方法和判定标准，不停留在“继续观察”“可能有问题”。
- **按读者需要分层**：入口文档只放用途、常用操作和关键约束；架构、专题说明和历史记录分开，通过链接连接，避免一个文档包揽所有内容。标题按职责或主题分类，重要信息放前面。
- **术语集中解释**：优先使用易懂的中文；必要术语首次出现时给一句解释，详细定义、例子及项目含义统一维护到 `docs/terminology.md`（需要时创建），其他文档引用，不重复展开。
- **代码与说明同步**：修改行为、参数、默认值、路径或数据格式时，检查并同步相关注释、docstring、命令示例和当前文档。现状以实现和验证结果为依据，发现冲突应核实修正；历史记录保留原意并标明适用版本或日期。
- **注释解释关键点**：写清目的、约束、原因和容易误用的边界，不逐行翻译代码。函数说明按需补充输入输出、单位、张量形状和副作用，不堆砌显而易见的信息。

## 编码与测试

- **代码按步骤成块**：一个函数做几件事，就按这些步骤组织代码；例如五个主要步骤分成五个清晰的逻辑块，用空行和简短的目的注释分隔。紧密相关的语句保持连续，不逐句插空行，不使用大段装饰性分隔线；定义之间遵循 Python 常规空行规范。
- **按职责适度拆分**：长函数出现独立职责、重复逻辑或难以单独验证的步骤时，提取有明确名称的函数；文件承担多个独立主题时，按主题拆文件。不要仅为满足行数限制拆分，也不要把简单流程拆成需要反复跳转的多层包装。
- **抽象必须可追踪**：优先显式调用和清晰的数据传递。使用注册表、动态导入、反射、回调分发等编辑器难以直接追踪的机制时，在入口简要说明选择规则、实际实现位置和调用关系；不为尚不存在的需求搭建通用框架。
- **同类功能保持一致**：新增实现前先查找已有同类代码，沿用命名、参数顺序、返回结构、错误处理和配置方式。相同规则复用公共实现；确有差异时明确原因，避免无理由并存多套写法。
- **常量集中且就近**：重复使用且含义相同的路径、字段名、状态字符串和阈值提为具名常量，同一规则只保留一个定义。专属常量放所属模块，跨模块共享的才放公共位置；不要仅因字面值相同就合并不同含义的常量。
- **常量必须有中文注释**：新增或修改常量时，每个常量都应有同行或紧邻的中文注释，说明用途；涉及数值时写清单位或计数口径，布尔开关写清启用行为。存在覆盖关系、适用条件或容易误解的边界时一并说明，不能只写分组标题代替逐项注释。修改常量值时同步更新注释和相关文档。
- **命令行保持精简**：已有常量控制的配置直接使用常量，不再提供同义命令行覆盖；只有一个有效选择或由当前流程固定的选项不暴露为参数。仅保留每次运行确有必要指定的输入，例如本次输出目录。区分固定的输出根目录常量与本次运行目录，不为固定配置再建立一套参数映射。删减参数时同步入口调用、帮助文本、测试和文档。
- **训练日志按可行动性分级**：保留 epoch 级 loss/F1/学习率/增强阶段和耗时；逐 batch、逐参数梯度和每轮可视化必须降频或关闭。检查点同时保留最佳 loss 和最佳 F1，部署选型优先比较 F1，不能只用 loss 删除检查点。
- **脚本开箱可用**：常用路径和配置提供合理默认值，遵循上述命令行精简规则；保留的参数写清用途、单位和默认行为。无法合理推断的必要输入明确要求提供，涉及数据删除或覆盖的操作默认只预览。
- **必要测试必须补齐**：新增或修改行为时，覆盖正常路径、关键边界和错误输入；修复缺陷时补能复现问题的回归测试。共享预处理和多后端契约需验证一致性，重构需确认行为不变；测试断言可观察结果，避免绑定内部实现或重复覆盖同一行为。
- **提交前核对**：检查改动是否易读、分块是否清晰、同类风格是否一致、说明是否同步，并运行上方要求的相关测试。未完成的验证要说明原因和可执行的后续步骤，不能宣称已通过。

## 数据与产物

`detection/data/`、`detection_small/data/`、`detection_larger/data/`、`recognition/data/`、训练日志、checkpoints、推理输出和大模型产物都不应提交。不要删除、移动或批量改写用户的本地数据集和权重，除非任务明确要求并已核对目标路径。
