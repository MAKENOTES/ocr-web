# Recognition CPU 导出、推理与量化报告

> 历史分析：结论限于原文注明的代码、数据及权重版本，本次未重做训练或性能测量。当前行为见[架构](../../recognition.md)和[运行说明](../../guides/running.md)。

完成日期：2026-09-09。测试 CPU：Intel Core i5-12490F，6 核 / 12 线程。

## 1. 结论与推荐

推荐 **ONNX Runtime FP32，4 个推理线程，关闭线程空转**。单图使用 batch=1；离线目录
识别使用 **batch=4 + 按宽度分组**。在全部 **9,553 张、88,196 个文字字符**的验证集上，
推荐配置的 CER 为 **0.9354%**，整行准确率 **94.8184%**，与 PyTorch 的识别文字
**9,553/9,553 完全一致**。

- batch=1：完整数据遍历约 **205.7 张/秒**，模型调用平均 **2.860 ms/张**。
- batch=4：完整数据遍历约 **257.2 张/秒**，模型调用摊销 **2.104 ms/张**；全量耗时 **37.15 秒**。
- INT8 的误差增幅很小，但本机单图速度仍落后于 ONNX FP32；本次不将其作为准确率优先的默认方案。
- `logits` 相比 `log_probs` 的单次全量吞吐仅提升约 1%，接近计时波动，不能把它宣传为主要加速来源。
  需要 log 概率或后续概率计算的调用方可以保留默认 `log_probs`；当前 greedy 解码可使用 logits。

以下是已生成、可直接运行的推荐模型：

```bash
# 单图 / 顺序识别
.venv/bin/python recognition_inference.py \
  --model_path outputs/recognition_cpu_20260908/logits.onnx \
  --image_path recognition/data/val_images/000003-000002.png \
  --backend auto --num_threads 4 --batch_size 1

# 离线批处理；结果仍按原始文件名排序
.venv/bin/python recognition_inference.py \
  --model_path outputs/recognition_cpu_20260908/logits.onnx \
  --image_path recognition/data/val_images \
  --backend auto --num_threads 4 --batch_size 4 --sort_by_width \
  --output_file outputs/recognition_cpu_20260908/results.txt
```

## 2. 模型与数据可追溯性

用户指定权重：`recognition/checkpoints/1788878430best_cer_0.0093_epoch_120.pth`。
首次成功使用该文件导出 `baseline.onnx`。随后原路径消失，现场仍在更新训练 checkpoint；
`recognition/train.py` 的 Top-K 保存逻辑会移除旧 best 文件。使用同一轮 `epoch_120.pth`
复制得到 `outputs/recognition_cpu_20260908/epoch_120.pth` 后继续测试。

这不是换用其他训练轮次：原始 ONNX 与副本导出 ONNX 的 **41 项初始化参数内容全部一致**
（按 dtype、shape、数值字节 SHA256 比较，忽略导出节点编号变化）；另做随机混合宽度输出
对照，最大绝对误差为 **1.812×10⁻⁵**。证据见 `checkpoint_equivalence.json`、
`baseline_export.log`。原始 best 文件已无法补算文件哈希，副本哈希如下。

| 文件 | SHA256 |
| --- | --- |
| 固定 epoch_120.pth 副本 | `57c7f5b17338fec213a48b42dc8f62acd69c6166e70ff72b511f84d7b1f4f050` |
| val_annotation.txt | `25b98b8072b78fdb36562417ab9e3f82bd387b37addb6216da911ebfd711805d` |
| keys.txt | `8c63758731378171f64a1d2c1d85d23c1b64754e1154d34867a644049cbee236` |

模型输入为 **3 通道 RGB、高度 32**，字典 3,009 个字符，加 blank/UNK 共 **3,011 类**。
评估严格复用当前图片加载、透明合成、保守去红章、完整文字框等比缩放、`[-1,1]` 归一化及
动态 padding。真实宽度不包括每张图右侧的 32px CNN 保护区，长图不被强制压缩到 256。

校准从训练目录的 **38,234 张图片中固定种子 20260908 抽取 300 张**，两种正式 INT8
候选使用同一名单；名单分别保存在 `cnn_int8/quantization.json`、`head_int8/quantization.json`。
训练与验证标注的图片文件名交集为 0。此检查不证明没有内容近重复，也不替代独立业务测试集。
未修改、删除训练图片、验证图片、字典或用户原有权重。

## 3. 全量 CPU 对照

前六行均使用 batch=1、4 个推理线程、num_workers=0。ONNX 关闭线程空转；非 PyTorch
后端的张量预处理和解码只使用一个 PyTorch 线程。OpenVINO 固定单 stream 和 FP32 浮点
计算提示，INT8 层按量化图执行。最后一行是推荐的 batch=4 全量复核。

| 配置 | CER↓ | 整行准确率↑ | 编辑距离总数 | 模型调用 ms/张↓ | 完整遍历 张/秒↑ | 与 PyTorch 不同的文本数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 0.9354% | 94.8184% | 825 | 5.609 | 131.9 | 0 |
| ONNX FP32 / log_probs | 0.9354% | 94.8184% | 825 | 2.918 | 203.7 | 0 |
| ONNX FP32 / logits | 0.9354% | 94.8184% | 825 | 2.860 | 205.7 | 0 |
| OpenVINO FP32（修复后） | 0.9354% | 94.8184% | 825 | 3.711 | 170.4 | 0 |
| OpenVINO INT8：仅 CNN | 0.9400% | 94.8079% | 829 | 3.624 | 174.1 | 118 |
| OpenVINO INT8：CNN + 分类层 | 0.9456% | 94.8079% | 834 | 3.162 | 190.4 | 115 |
| ONNX FP32 / logits，batch=4、宽度分组 | 0.9354% | 94.8184% | 825 | 2.104 | 257.2 | 0 |

CER = 全部 Levenshtein 编辑距离 / 全部标签字符数；整行准确率要求文字完全相等，不做
标点或 Unicode 宽松归一化。模型调用耗时不包含 greedy 解码；完整遍历包含图片读取、
预处理、collate、模型执行、解码和内存结果收集，**不含加载/编译、预热、宽高比预扫描及写报告**。
模型预热 5 次；动态形状第一次出现产生的额外执行开销计入数据遍历。

这是一台共享机器上的单次全量遍历，不是隔离 CPU、固定频率下的 SLA 测试。批处理复核在
9 月 9 日完成，其余主对照在 9 月 8 日完成。中途发现已中断测试的残留子进程造成 CPU 竞争，
停止这些测试进程后重测；受污染结果另存 `discarded_contended_onnx12.json`，不用于上述结论。
本报告没有停止用户训练。纯随机固定形状测试与真实流水线耗时不能混用。

### INT8 的具体代价

CNN INT8 比 FP32 增加 **4 个编辑距离**，CER 增加 **0.00454 个百分点**；
CNN+分类层 INT8 增加 **9 个编辑距离**，CER 增加 **0.01020 个百分点**。
虽然两者整行正确数都只净减少 1，但分别有 **118 / 115 张**的文本发生变化，
包含原错误被纠正和新增错误，不能表述为“仅一张有变化”或“完全无损”。

| 按预处理后真实像素宽度分组 | 图片数 | 标签字符数 | FP32 CER | CNN INT8 CER | CNN+分类层 INT8 CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤128 | 5790 | 16283 | 0.4238% | 0.3808% | 0.3930% |
| 129–256 | 1775 | 14615 | 0.4721% | 0.4653% | 0.4653% |
| 257–512 | 842 | 12605 | 1.1821% | 1.1900% | 1.1900% |
| >512 | 1146 | 44693 | 1.2038% | 1.2284% | 1.2351% |

当前数据没有完整的红章、合法红字、主题、字体、语言分组标签，本次未声称这些业务分组
分别通过上线门禁；也未用字典外字符集补充本次 CER。选型依据的是现有验证集，不代表
未见业务分布的最高准确率保证。

### 模型体积

ONNX 统计整个文件；OpenVINO 统计 XML + BIN，总体积而非仅权重文件。

| 产物 | 大小 |
| --- | ---: |
| ONNX FP32 | 9.445 MiB |
| OpenVINO FP32 | 9.515 MiB |
| CNN INT8 | 9.184 MiB |
| CNN+分类层 INT8 | 4.790 MiB |

仅量化 CNN 体积收益有限，因为 GRU 和分类层占了大部分权重。额外量化分类层才有明显
压缩，但本次仍达不到原脚本注释笼统宣称的 3–4 倍压缩与 1.5–3 倍加速。

## 4. 批量推理结果

固定种子随机抽取同一批 **512 张**验证图片，按宽度排序，各配置独立运行；单位为
完整数据遍历的张/秒。该子集 CER 低于全量集，不能用子集数字替代全量准确率。

| 后端 | batch=1 | batch=4 | batch=8 | batch=16 |
| --- | ---: | ---: | ---: | ---: |
| ONNX FP32 | 231.0 | 272.8 | 259.2 | 263.2 |
| OpenVINO FP32 | 167.0 | 212.3 | 266.2 | 265.6 |
| OpenVINO CNN+分类层 INT8 | 162.6 | 269.9 | 278.2 | 240.4 |

各后端自己的 batch=1/4/8/16 文本一致率均为 **100%**。进一步将 ONNX batch=4 扩展
到全部 9,553 张，文本与 PyTorch 单图输出仍完全一致；吞吐 **257.2 张/秒**。
因此推荐离线批处理先使用 batch=4，避免把小子集某次最高的数字视为普适最优。
单图请求仍建议 batch=1，以免为了凑批增加等待延迟。

## 5. 修复与优化说明

### recognition_export_onnx.py

新增 CLI 参数，使用明确的 checkpoint 路径，支持动态 batch/width 和可选 logits。
未改变 CNN、GRU 权重或字典。默认仍导出 log_probs；logits 对 greedy CTC 的 argmax
等价，但调用方不能直接把它当作 log 概率使用。

无论是否安装 onnxsim，都会检查 ONNX 及 GRU sequence_lens。数值校验同时覆盖 batch=1/4
和不同实际输入宽度；本模型导出对照的最大误差约 **1.7×10⁻⁵**。基准中的 PyTorch 和
ONNX 使用同一真实宽度与右侧保护区。模型写入字典 SHA256，ONNX 加载时核对字典，
避免字数相同但字符顺序变化仍被静默接受。

### recognition_inference.py 与 OpenVINO 长度修复

提供 CPU 线程数、预处理进程数、可选宽度分组；`.xml` 由 auto 正确选择 OpenVINO。
默认禁用 ONNX 线程空转，将其与 PyTorch 预处理线程分开管理。ORT 的线程数和空转是
独立配置项，详见 [ONNX Runtime 官方线程文档](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)。
本机固定形状扫描中，宽度 256 的 ONNX 4 线程约 2.03 ms，而更高线程数并不持续获益；
实际部署最终以带预处理的数据遍历为准。`--allow_spinning` 保留为显式实验选项。

本次发现比量化更严重的错误：**未修复的 OpenVINO FP32 CER 为 11.3543%，整行准确率
仅 1.6121%**，例如“投标文件”被识别成“投标文”。逐层对比显示 CNN 最大误差仅
**2.86×10⁻⁶**，GRU 收到的真实长度也正确（32），但 CPU 实际继续遍历总长 40 的
padding，反向状态出现约 2 的偏差，padding 处 GRU 输出也没有清零。
这是本机 OpenVINO 2026.0 CPU 的实测现象，并非推断所有版本都存在同样问题；
[GRUSequence 官方语义](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/gru-sequence-5.html)
要求使用每个样本的真实 sequence_lengths。

新增 `recognition/utils/openvino.py`：将双向 GRU 转为两个正向 GRU，用按 batch 的 Gather
显式反转有效前缀，再反转输出，最后沿用 blank mask。正向有效前缀不受后续 padding 影响，
同时保留完整 CNN 保护区。此转换用于推理加载和量化前处理，也写入生成的 IR。
已转换 IR 再次加载不会重复处理；消费最终 hidden state 的其他架构会显式报错。

修复后的 OpenVINO FP32 与 PyTorch 在全量验证集上逐图完全一致。
回归测试新增“对齐 PyTorch”断言，避免仅比较两个同样错误的 OpenVINO batch 输出。
量化还增加 ONNX→OpenVINO 的前 16 张真实校准输入数值门禁；实际校验最大误差
**3.033×10⁻⁴**，超出容差就拒绝继续量化。

### recognition_quantize_static.py

原脚本生成 `[1,1,32,256]` 灰度、`[0,1]` 输入；用指定模型实测直接报
“Got: 1 Expected: 3”。即使临时改为三通道，长图压缩、归一化、透明处理和 padding
仍与真实推理不一致。本次完全复用共享数据集和 collate，默认不限制宽度，固定抽样种子，
保留 GRU 与分类层 FP32，并以 `--quantize_head` 明确开启额外分类层量化。

原“精度控制”使用 FP32/INT8 输出余弦相似度，且大量 -1e9 padding 会掩盖真实文字差异。
现改为真实标注的 **1-CER**，在 NNCF 恢复敏感层后再次检查 CER 门禁。指标越大越好，
符合 [NNCF 精度控制 API](https://openvinotoolkit.github.io/nncf/autoapi/nncf/) 的要求。
`--max_drop 0.001` 表示最多增加 **0.1 个百分点 CER**；启用精度控制必须提供标签。

真实精度恢复冒烟测试使用 **64 张训练校准图 + 128 张 val 图、max_drop=0**。
NNCF 确实触发敏感层恢复，最终 FP32/INT8 都为 **10/1281 个编辑距离**，128 张文本
完全一致。这用于验证恢复流程，**不能作为全量无损 INT8 的证明**，该小样本模型未作为
推荐部署模型。可选 ONNX 直接静态量化也已用 32 张校准图验证生成 ONNX 和修复后的 IR。
命令行实测还发现 ORT CPU 的 QLinearConv 不接受逐通道激活零点；该可选路径已明确
使用逐张量激活量化，权重仍可逐通道量化，并在保存后执行真实不同宽度输入验证。
最终四张真实图片的 batch=4 命令行检查通过。它只完成兼容性冒烟测试，未作为上述全量
INT8 对照或推荐模型。

## 6. 验证与已有测试问题

- 识别预处理、batch 一致性及新增部署测试：**39 passed**。
- 三个目标脚本、新评估脚本、OpenVINO 修复模块：Python 编译检查通过；`git diff --check` 通过。
- 原生 PyTorch、ONNX、OpenVINO FP32 和两种 INT8：完成上述全量评估并保存逐图预测。
- 多 batch 小子集一致性和推荐 batch=4 全量一致性：通过。
- CLI 实图目录批处理、auto 选择 XML、输出恢复文件名顺序，以及可选 ONNX INT8 加载：通过。
- 全仓 pytest 被两个已有缺失模块阻断：`remove_annotation_outer_quotes`、`rename_synthesis_images`。
  排除对应两个测试文件后为 **172 passed / 17 failed**。17 项失败位于字符统计及按统计移除
  数据的旧测试，预期仍使用单个 `synthesis`，当前实现已采用 `synthesis1/synthesis2` 列；
  本次未改这些实现或测试，也未将全仓状态报告为全部通过。

## 7. 复现与产物

所有模型、校准名单、JSON、逐图预测及日志保存在
`outputs/recognition_cpu_20260908/`，属于 Git 忽略的本地产物。
`before_gru_fix_*` 和 `discarded_contended_*` 为诊断证据，不是推荐部署产物。

```bash
# 从已核对一致的固定副本导出；不覆盖用户根目录已有模型
.venv/bin/python recognition_export_onnx.py \
  --model_path outputs/recognition_cpu_20260908/epoch_120.pth \
  --output_path outputs/recognition_cpu_20260908/logits.onnx \
  --keys_path recognition/keys.txt --output_type logits --num_threads 4

# 全量复核推荐配置，生成 JSON 和同名 .predictions.jsonl
.venv/bin/python recognition_benchmark_cpu.py \
  --model_path outputs/recognition_cpu_20260908/logits.onnx \
  --annotation recognition/data/val_annotation.txt \
  --num_threads 4 --batch_size 4 --sort_by_width \
  --output outputs/recognition_cpu_20260908/final_onnx_batch4.json

# CNN 静态量化，使用带标注数据评估，validation_size=0 表示全量
.venv/bin/python recognition_quantize_static.py \
  --onnx_model outputs/recognition_cpu_20260908/logits.onnx \
  --calibration_dir recognition/data/train_images \
  --output_dir outputs/recognition_cpu_20260908/cnn_int8 \
  --subset_size 300 --validation_annotation recognition/data/val_annotation.txt \
  --validation_size 0 --num_threads 4

# 需要更严格控制 INT8 时，使用独立带标注调参集：
# 在上面的量化命令加 --accuracy_control --max_drop 0
# 更多分类层压缩另加 --quantize_head；恢复后仍需独立业务测试集复核。
```

核心原始文件：`summary.json`、各后端 `.json/.predictions.jsonl`、
`final_onnx_batch4.json`、`thread_sweep.json`、`cli_smoke.json`、
`accuracy_control_smoke.log`、`deployment_tests.log`、`all_tests_final.log`、
`available_tests_final.log`。环境为 Python 3.10.16、PyTorch 2.9.0、ONNX 1.20.1、
ONNX Runtime 1.23.2、OpenVINO 2026.0.0、NNCF 3.0.0，推理和校准均在 CPU 执行。
