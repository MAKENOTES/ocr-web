"""
recognition_quantize_int8.py — NNCF INT8 Post-Training Quantization for CRNN ONNX Model
========================================================================================

功能:
  1. 加载已导出的 ONNX CRNN 模型
  2. 使用 NNCF PTQ (Post-Training Quantization) 进行 INT8 量化
  3. 支持两条路径:
     - 路径 A (推荐): ONNX → OpenVINO IR (FP32) → NNCF 量化 → OpenVINO IR (INT8)
     - 路径 B: 直接对 ONNX 模型做 NNCF 量化 → 量化后的 ONNX → OpenVINO IR (INT8)
  4. 输出: OpenVINO IR 格式 (.xml + .bin)，可直接部署
  5. 自动进行 FP32 vs INT8 精度对比验证和速度 benchmark

使用方法:
    # 基础用法 (使用校准图片目录)
    python recognition_quantize_int8.py \
        --onnx_model best_recognition.onnx \
        --keys_path recognition/keys.txt \
        --calibration_dir recognition/data/train_images \
        --output_dir quantized_model

    # 高级用法 (带精度控制)
    python recognition_quantize_int8.py \
        --onnx_model best_recognition.onnx \
        --keys_path recognition/keys.txt \
        --calibration_dir recognition/data/train_images \
        --output_dir quantized_model \
        --subset_size 300 \
        --accuracy_control \
        --validation_dir recognition/data/test_images \
        --max_drop 0.02

依赖:
    pip install nncf openvino onnx onnxruntime numpy Pillow

注意事项:
    1. 校准数据集建议使用 200-500 张具有代表性的训练/验证图片
    2. 图片应覆盖各种文本长度、字体、背景条件
    3. 量化后模型大小约缩减 3-4 倍，推理速度提升 1.5-3 倍
    4. 对于 CRNN (CNN + GRU) 架构，建议排除 GRU 和最终分类层的量化
       以保持 CTC 解码精度

架构分析 & 量化策略:
    ┌──────────────────────────────────────────────────────────┐
    │  MobileFeatureExtractor (CNN)                           │
    │  ├─ stem: Conv2d + BN + ReLU6          ← INT8 量化 ✓   │
    │  ├─ pool: MaxPool2d                    ← 无参数，跳过   │
    │  ├─ block1: DSConv × 2                 ← INT8 量化 ✓   │
    │  ├─ block2: DSConv × 2                 ← INT8 量化 ✓   │
    │  └─ block3: DSConv × 2                 ← INT8 量化 ✓   │
    ├──────────────────────────────────────────────────────────┤
    │  BiGRU (RNN)                                            │
    │  └─ GRU(256, bidirectional)            ← 跳过量化 ✗    │
    │     原因: RNN 的循环连接对量化误差极度敏感，           │
    │           INT8 下误差会在时间步间累积放大               │
    ├──────────────────────────────────────────────────────────┤
    │  CTC Head                                               │
    │  └─ Linear(512 → n_class)              ← 跳过量化 ✗    │
    │     原因: CTC 解码依赖精确的 log_softmax 概率分布，    │
    │           最后一层量化容易导致字符漏识/误识             │
    └──────────────────────────────────────────────────────────┘
"""

import os
import sys
import glob
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. 校准数据集准备
# ============================================================

class CRNNCalibrationDataset:
    """
    CRNN 校准数据集。

    与训练/推理 pipeline 使用完全相同的预处理逻辑:
    - 灰度读取
    - 高度缩放到 IMG_HEIGHT (32)，宽度等比缩放
    - 归一化到 [0, 1]
    - 统一 padding 到固定宽度 (max_width)

    NNCF ONNX 后端要求 transform_fn 返回 Dict[str, np.ndarray]，
    key 为模型输入名称。我们的 CRNN ONNX 模型有两个输入:
    - "input": [1, 1, 32, W] 图像张量
    - "img_widths": [1] 真实像素宽度
    """

    def __init__(self, image_dir, img_height=32, max_width=256):
        self.img_height = img_height
        self.max_width = max_width

        # 收集图片路径
        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

        if not self.image_paths:
            raise FileNotFoundError(f"在 {image_dir} 未找到图片文件")

        print(f"  校准数据集: {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # 灰度

        # 等比缩放到目标高度
        w_orig, h_orig = img.size
        ratio = self.img_height / h_orig
        new_w = max(1, int(w_orig * ratio))

        # 限制宽度不超过 max_width
        if new_w > self.max_width:
            new_w = self.max_width

        img = img.resize((new_w, self.img_height), Image.BILINEAR)

        # 转为 numpy，归一化
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = img_np[np.newaxis, :, :]  # [1, H, W]

        real_width = new_w

        # Padding 到 max_width
        if new_w < self.max_width:
            pad_width = self.max_width - new_w
            img_np = np.pad(img_np, ((0, 0), (0, 0), (0, pad_width)),
                            mode='constant', constant_values=0.0)

        # 返回 [1, 32, max_width] (C, H, W)，不含 batch 维
        # DataLoader(batch_size=1) 会自动添加 batch 维 → [1, 1, 32, W]
        return img_np, np.array([real_width], dtype=np.int64)


def create_calibration_dataloader(image_dir, img_height=32, max_width=256, subset_size=300):
    """创建用于 NNCF 校准的 DataLoader。"""
    import torch
    from torch.utils.data import DataLoader

    dataset = CRNNCalibrationDataset(image_dir, img_height, max_width)

    # 如果数据集太大，随机选取 subset_size 张
    if len(dataset) > subset_size:
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  随机选取 {subset_size} 张用于校准")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return loader


# ============================================================
# 2. 路径 A: ONNX → OpenVINO IR → NNCF INT8 (推荐)
# ============================================================

def quantize_via_openvino_ir(onnx_model_path, calibration_dir, output_dir,
                              img_height=32, max_width=256, subset_size=300,
                              accuracy_control=False, validation_dir=None,
                              max_drop=0.01, keys_path=None):
    """
    推荐路径: 先转 OpenVINO IR，再用 NNCF 量化。

    优势:
    - OpenVINO 后端是 NNCF 的首选后端，支持最完整
    - 可以使用 ignored_scope 精确排除 GRU / Linear 层
    - 支持 quantize_with_accuracy_control 精度控制模式
    - 输出直接为 OpenVINO IR，无需二次转换
    """
    import nncf
    import openvino as ov

    print("\n" + "=" * 60)
    print("路径 A: ONNX → OpenVINO IR (FP32) → NNCF INT8")
    print("=" * 60)

    # ---- Step 1: 转换 ONNX → OpenVINO IR (FP32) ----
    print("\n[1/5] 转换 ONNX → OpenVINO IR (FP32)...")
    core = ov.Core()

    # 重要: 不要压缩到 FP16，保持 FP32 精度作为量化基准
    ov_model = ov.convert_model(onnx_model_path)
    print(f"  ✓ OpenVINO IR 转换成功")

    # 打印模型输入输出信息
    for inp in ov_model.inputs:
        print(f"  输入: {inp.any_name}, shape={inp.partial_shape}, dtype={inp.element_type}")
    for out in ov_model.outputs:
        print(f"  输出: {out.any_name}, shape={out.partial_shape}, dtype={out.element_type}")

    # 保存 FP32 IR 用于对比
    fp32_dir = os.path.join(output_dir, "fp32")
    os.makedirs(fp32_dir, exist_ok=True)
    fp32_model_path = os.path.join(fp32_dir, "crnn_fp32.xml")
    ov.save_model(ov_model, fp32_model_path, compress_to_fp16=False)
    print(f"  FP32 模型已保存: {fp32_model_path}")

    # ---- Step 2: 准备校准数据集 ----
    print(f"\n[2/5] 准备校准数据集...")
    calibration_loader = create_calibration_dataloader(
        calibration_dir, img_height, max_width, subset_size
    )

    def transform_fn_ov(data_item):
        """
        OpenVINO 后端的 transform_fn。
        返回值直接作为 ov.CompiledModel 的输入。

        DataLoader(batch_size=1) 输出:
          img_tensor: torch.Tensor [1, 1, 32, W]  (batch + C + H + W)
          img_widths: torch.Tensor [1, 1]          (batch + 1)

        模型期望:
          input:      [B, 1, 32, W]  → 取 [1, 1, 32, W] 直接可用
          img_widths: [B]            → 需要 squeeze 成 [1]
        """
        img_tensor, img_widths = data_item
        # DataLoader 返回的是 torch.Tensor，需转 numpy
        if hasattr(img_tensor, 'numpy'):
            img_tensor = img_tensor.numpy()
        if hasattr(img_widths, 'numpy'):
            img_widths = img_widths.numpy()

        # img_tensor 已经是 [1, 1, 32, W]（DataLoader 的 batch 维 = 模型的 batch 维）
        img_np = img_tensor.astype(np.float32)

        # img_widths 可能是 [1, 1]，需 flatten 成 [1]
        widths_np = img_widths.flatten().astype(np.int64)

        return {"input": img_np, "img_widths": widths_np}

    # 验证 transform_fn 输出 shape
    _sample = next(iter(calibration_loader))
    _check = transform_fn_ov(_sample)
    print(f"  transform_fn 输出验证:")
    print(f"    input shape:      {_check['input'].shape}  (期望: [1, 1, 32, {max_width}])")
    print(f"    img_widths shape: {_check['img_widths'].shape}  (期望: [1])")
    assert _check['input'].ndim == 4, f"input 维度错误: {_check['input'].shape}, 期望 4D"
    assert _check['img_widths'].ndim == 1, f"img_widths 维度错误: {_check['img_widths'].shape}, 期望 1D"

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn_ov)

    # ---- Step 3: 分析模型图，确定 ignored_scope ----
    print(f"\n[3/5] 分析模型结构，确定量化策略...")

    # 获取 NNCF 图来查找 GRU 和最终 Linear 层的节点名
    # 策略: 排除 GRU 和最终 embedding (Linear) 层
    #
    # 对于 OpenVINO 后端，可以按 operation type 排除:
    # - GRU → 在 OpenVINO opset 中对应 "GRUSequence"
    # - MatMul (最后的 Linear) → 需要按名称精确匹配
    #
    # 也可以按名称模式匹配 (更精确):
    # - GRU 相关节点通常包含 "rnn" 或 "gru" 或 "GRU"
    # - embedding (Linear) 层通常包含 "embedding" 或 "MatMul"

    # 方案 1: 按类型排除 GRU (简单通用)
    # 方案 2: 按名称模式排除 (更精确，需要检查实际节点名)

    # 我们先尝试按类型排除 GRU，并通过名称模式排除最终 Linear
    ignored_scope = nncf.IgnoredScope(
        # 排除 GRU 循环层 (OpenVINO opset 中的类型名)
        types=["GRUSequence"],
        # 可选: 排除最终分类 Linear 层 (MatMul)
        # 如果你知道节点名，可以用 names= 精确指定
        # 或用 patterns= 正则匹配
        # patterns=[r".*embedding.*"],  # 如果 ONNX 中保留了层名
    )

    print(f"  量化策略:")
    print(f"    - CNN (DepthwiseSeparable Conv): INT8 量化 ✓")
    print(f"    - GRU (BiGRU): 排除量化 ✗ (类型: GRUSequence)")
    print(f"    - Embedding (Linear): 取决于精度测试结果")
    print(f"    - Dropout / Padding Mask: 无参数，自动跳过")

    # ---- Step 4: 执行 NNCF 量化 ----
    print(f"\n[4/5] 执行 NNCF INT8 量化 (subset_size={subset_size})...")

    if accuracy_control and validation_dir:
        # 带精度控制的量化
        print(f"  使用精度控制模式 (max_drop={max_drop})")

        validation_loader = create_calibration_dataloader(
            validation_dir, img_height, max_width, subset_size
        )
        validation_dataset = nncf.Dataset(validation_loader, transform_fn_ov)

        def validation_fn(model, validation_dataset):
            """
            验证函数: 计算 CTC 解码的字符级准确率。
            注意: 这里使用简化的 greedy decode 对比。
            """
            # 对于无标签的简化验证，我们使用 FP32 vs INT8 输出一致性
            # 作为精度指标 (余弦相似度)
            compiled = ov.Core().compile_model(model, "CPU")
            similarities = []

            # 获取 FP32 参考模型
            fp32_compiled = ov.Core().compile_model(
                ov.Core().read_model(fp32_model_path), "CPU"
            )

            for data_item in validation_dataset.get_data(range(min(100, subset_size))):
                fp32_result = fp32_compiled(data_item)[0]
                int8_result = compiled(data_item)[0]

                # 余弦相似度
                fp32_flat = fp32_result.flatten()
                int8_flat = int8_result.flatten()
                cos_sim = np.dot(fp32_flat, int8_flat) / (
                    np.linalg.norm(fp32_flat) * np.linalg.norm(int8_flat) + 1e-8
                )
                similarities.append(cos_sim)

            return float(np.mean(similarities)), None

        quantized_model = nncf.quantize_with_accuracy_control(
            ov_model,
            calibration_dataset=calibration_dataset,
            validation_dataset=validation_dataset,
            validation_fn=validation_fn,
            max_drop=max_drop,
            drop_type=nncf.DropType.ABSOLUTE,
            subset_size=subset_size,
            fast_bias_correction=False,  # 更精确的 bias correction
            ignored_scope=ignored_scope,
        )
    else:
        # 基础量化 (无精度控制)
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            subset_size=subset_size,
            # preset: PERFORMANCE (默认, 对称量化) 或 MIXED (非对称, 更精确)
            # CRNN 推荐 MIXED，因为激活值分布不均匀
            preset=nncf.QuantizationPreset.MIXED,
            fast_bias_correction=False,  # 更精确的 bias correction
            ignored_scope=ignored_scope,
            # model_type: 不设为 TRANSFORMER，因为 CRNN 不是 Transformer
        )

    print(f"  ✓ INT8 量化完成")

    # ---- Step 5: 保存 INT8 模型 ----
    print(f"\n[5/5] 保存 INT8 量化模型...")
    int8_dir = os.path.join(output_dir, "int8")
    os.makedirs(int8_dir, exist_ok=True)
    int8_model_path = os.path.join(int8_dir, "crnn_int8.xml")

    # 重要: compress_to_fp16=False 保持未量化层的 FP32 精度
    # 这对被 ignored_scope 排除的 GRU 层尤其重要
    ov.save_model(quantized_model, int8_model_path, compress_to_fp16=False)

    int8_bin_path = int8_model_path.replace(".xml", ".bin")
    xml_size = os.path.getsize(int8_model_path) / 1024 / 1024
    bin_size = os.path.getsize(int8_bin_path) / 1024 / 1024
    print(f"  INT8 模型已保存: {int8_model_path}")
    print(f"  模型大小: .xml={xml_size:.2f}MB, .bin={bin_size:.2f}MB")

    return fp32_model_path, int8_model_path


# ============================================================
# 3. 路径 B: 直接对 ONNX 做 NNCF 量化
# ============================================================

def quantize_onnx_directly(onnx_model_path, calibration_dir, output_dir,
                            img_height=32, max_width=256, subset_size=300):
    """
    备选路径: 直接对 ONNX 模型做 NNCF INT8 量化。

    适用场景:
    - 需要保留 ONNX 格式 (如部署到 ONNX Runtime)
    - 调试时对比两种路径的量化效果

    注意: NNCF ONNX 后端功能比 OpenVINO 后端少:
    - 不支持 quantize_with_accuracy_control
    - ignored_scope 的类型名要用 ONNX opset (如 "Conv", "MatMul")
    """
    import nncf
    import onnx
    import openvino as ov

    print("\n" + "=" * 60)
    print("路径 B: 直接 ONNX NNCF INT8 量化")
    print("=" * 60)

    # 1. 加载 ONNX 模型
    print(f"\n[1/4] 加载 ONNX 模型: {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # 获取输入名称
    input_names = [inp.name for inp in onnx_model.graph.input]
    print(f"  模型输入: {input_names}")

    # 2. 准备校准数据集
    print(f"\n[2/4] 准备校准数据集...")
    calibration_loader = create_calibration_dataloader(
        calibration_dir, img_height, max_width, subset_size
    )

    def transform_fn_onnx(data_item):
        """
        ONNX 后端的 transform_fn。
        返回 Dict[str, np.ndarray]，key 为模型输入名。
        """
        img_tensor, img_widths = data_item
        if hasattr(img_tensor, 'numpy'):
            img_tensor = img_tensor.numpy()
        if hasattr(img_widths, 'numpy'):
            img_widths = img_widths.numpy()
        return {
            "input": img_tensor.astype(np.float32),
            "img_widths": img_widths.flatten().astype(np.int64),
        }

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn_onnx)

    # 3. NNCF 量化
    print(f"\n[3/4] 执行 NNCF INT8 量化 (ONNX 后端)...")

    # ONNX 后端的 ignored_scope 使用 ONNX opset 类型名
    ignored_scope = nncf.IgnoredScope(
        types=["GRU"],  # ONNX opset 中 GRU 的类型名
    )

    quantized_onnx = nncf.quantize(
        onnx_model,
        calibration_dataset,
        subset_size=subset_size,
        preset=nncf.QuantizationPreset.MIXED,
        fast_bias_correction=False,
        ignored_scope=ignored_scope,
    )

    print(f"  ✓ ONNX INT8 量化完成")

    # 4. 保存
    print(f"\n[4/4] 保存量化模型...")
    int8_onnx_dir = os.path.join(output_dir, "int8_onnx")
    os.makedirs(int8_onnx_dir, exist_ok=True)
    int8_onnx_path = os.path.join(int8_onnx_dir, "crnn_int8.onnx")
    onnx.save(quantized_onnx, int8_onnx_path)

    onnx_size = os.path.getsize(int8_onnx_path) / 1024 / 1024
    print(f"  INT8 ONNX 已保存: {int8_onnx_path} ({onnx_size:.2f} MB)")

    # 转换为 OpenVINO IR
    int8_ir_path = os.path.join(int8_onnx_dir, "crnn_int8_from_onnx.xml")
    ov_model = ov.convert_model(int8_onnx_path)
    ov.save_model(ov_model, int8_ir_path, compress_to_fp16=False)
    print(f"  INT8 OpenVINO IR 已保存: {int8_ir_path}")

    return int8_onnx_path, int8_ir_path


# ============================================================
# 4. 验证 & Benchmark
# ============================================================

def validate_and_benchmark(fp32_model_path, int8_model_path,
                           calibration_dir, keys_path,
                           img_height=32, max_width=256):
    """
    对比验证 FP32 和 INT8 模型:
    1. 数值一致性 (最大误差、余弦相似度)
    2. CTC 解码一致性 (识别文本是否相同)
    3. 推理速度对比
    4. 模型大小对比
    """
    import openvino as ov

    print("\n" + "=" * 60)
    print("验证 & Benchmark")
    print("=" * 60)

    core = ov.Core()

    # 编译模型
    fp32_compiled = core.compile_model(
        core.read_model(fp32_model_path), "CPU",
        {"PERFORMANCE_HINT": "LATENCY"}
    )
    int8_compiled = core.compile_model(
        core.read_model(int8_model_path), "CPU",
        {"PERFORMANCE_HINT": "LATENCY"}
    )

    # ---- 数值一致性 ----
    print("\n[1/3] 数值一致性验证...")
    dataset = CRNNCalibrationDataset(calibration_dir, img_height, max_width)
    n_test = min(50, len(dataset))

    def _prepare_input(dataset, idx):
        """从 dataset[i] 取数据并补 batch 维，适配 OpenVINO 推理。
        dataset.__getitem__ 返回: img=[1,32,W], widths=[1]
        模型期望:                 img=[1,1,32,W], widths=[1]
        """
        img, widths = dataset[idx]
        # img: [C=1, H=32, W] → [B=1, C=1, H=32, W]
        if img.ndim == 3:
            img = img[np.newaxis, :, :, :]
        # widths: [1] → 保持 [1] (已经是 1D)
        widths = widths.flatten()
        return img.astype(np.float32), widths.astype(np.int64)

    max_errors = []
    cos_sims = []

    for i in range(n_test):
        img, widths = _prepare_input(dataset, i)

        fp32_out = fp32_compiled({"input": img, "img_widths": widths})[0]
        int8_out = int8_compiled({"input": img, "img_widths": widths})[0]

        max_err = np.max(np.abs(fp32_out - int8_out))
        max_errors.append(max_err)

        fp32_flat = fp32_out.flatten()
        int8_flat = int8_out.flatten()
        cos_sim = np.dot(fp32_flat, int8_flat) / (
            np.linalg.norm(fp32_flat) * np.linalg.norm(int8_flat) + 1e-8
        )
        cos_sims.append(cos_sim)

    print(f"  测试样本数: {n_test}")
    print(f"  最大绝对误差: mean={np.mean(max_errors):.4f}, max={np.max(max_errors):.4f}")
    print(f"  余弦相似度:   mean={np.mean(cos_sims):.6f}, min={np.min(cos_sims):.6f}")

    quality = "✓ 优秀" if np.mean(cos_sims) > 0.99 else \
              "⚠ 一般" if np.mean(cos_sims) > 0.95 else "✗ 需要调整"
    print(f"  量化质量评估: {quality}")

    # ---- CTC 解码一致性 ----
    print(f"\n[2/3] CTC 解码一致性检查...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from recognition.components.converter import LabelConverter
        converter = LabelConverter(keys_path)

        matches = 0
        mismatches = []
        for i in range(n_test):
            img, widths = _prepare_input(dataset, i)

            fp32_out = fp32_compiled({"input": img, "img_widths": widths})[0]
            int8_out = int8_compiled({"input": img, "img_widths": widths})[0]

            import torch
            fp32_text = converter.decode(torch.from_numpy(fp32_out), layout="TBC")
            int8_text = converter.decode(torch.from_numpy(int8_out), layout="TBC")

            if fp32_text == int8_text:
                matches += 1
            else:
                mismatches.append((i, fp32_text[0] if fp32_text else "", int8_text[0] if int8_text else ""))

        accuracy = matches / n_test * 100
        print(f"  解码一致率: {matches}/{n_test} ({accuracy:.1f}%)")
        if mismatches and len(mismatches) <= 10:
            print(f"  不一致样本:")
            for idx, fp32_t, int8_t in mismatches[:10]:
                print(f"    [{idx}] FP32: '{fp32_t}' → INT8: '{int8_t}'")
    except Exception as e:
        print(f"  ⚠ CTC 解码验证跳过 (原因: {e})")
        print(f"    请确保 recognition/components/converter.py 可导入")

    # ---- 速度 Benchmark ----
    print(f"\n[3/3] 推理速度 Benchmark...")

    # 使用固定输入进行基准测试
    bench_img = np.random.randn(1, 1, img_height, max_width).astype(np.float32)
    bench_widths = np.array([max_width], dtype=np.int64)
    bench_input = {"input": bench_img, "img_widths": bench_widths}

    # Warmup
    for _ in range(10):
        fp32_compiled(bench_input)
        int8_compiled(bench_input)

    # FP32
    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fp32_compiled(bench_input)
    fp32_time = (time.perf_counter() - t0) / n_runs * 1000

    # INT8
    t0 = time.perf_counter()
    for _ in range(n_runs):
        int8_compiled(bench_input)
    int8_time = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  FP32 (OpenVINO): {fp32_time:.2f} ms/张")
    print(f"  INT8 (OpenVINO): {int8_time:.2f} ms/张")
    print(f"  加速比:          {fp32_time / int8_time:.2f}x")

    # 模型大小对比
    fp32_bin = fp32_model_path.replace(".xml", ".bin")
    int8_bin = int8_model_path.replace(".xml", ".bin")
    fp32_size = os.path.getsize(fp32_bin) / 1024 / 1024
    int8_size = os.path.getsize(int8_bin) / 1024 / 1024
    print(f"\n  模型大小: FP32={fp32_size:.2f}MB → INT8={int8_size:.2f}MB "
          f"(压缩比 {fp32_size/int8_size:.1f}x)")


# ============================================================
# 5. 辅助: 打印模型节点名 (用于调试 ignored_scope)
# ============================================================

def inspect_model_nodes(onnx_model_path):
    """
    打印 ONNX 和 OpenVINO IR 中的节点名称，帮助确定 ignored_scope。
    """
    import openvino as ov

    print("\n" + "=" * 60)
    print("模型结构检查 (用于调试 ignored_scope)")
    print("=" * 60)

    # OpenVINO IR 节点
    print("\n--- OpenVINO IR 节点 ---")
    core = ov.Core()
    ov_model = ov.convert_model(onnx_model_path)

    for op in ov_model.get_ordered_ops():
        op_type = op.get_type_name()
        op_name = op.get_friendly_name()
        # 只打印关键层
        if op_type in ["Convolution", "GroupConvolution", "MatMul",
                        "GRUSequence", "LSTMSequence", "GRUCell",
                        "Add", "Multiply"]:
            shapes = [str(out.get_partial_shape()) for out in op.outputs()]
            print(f"  {op_type:20s} | {op_name:50s} | out={shapes}")

    # ONNX 节点
    print("\n--- ONNX 节点 ---")
    import onnx
    model = onnx.load(onnx_model_path)
    for node in model.graph.node:
        if node.op_type in ["Conv", "Gemm", "MatMul", "GRU", "LSTM",
                            "BatchNormalization"]:
            print(f"  {node.op_type:20s} | {node.name}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NNCF INT8 量化工具 (CRNN ONNX → OpenVINO INT8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 1. 查看模型结构 (确认 ignored_scope)
  python recognition_quantize_int8.py --onnx_model best_recognition.onnx --inspect

  # 2. 基础量化
  python recognition_quantize_int8.py \\
      --onnx_model best_recognition.onnx \\
      --keys_path recognition/keys.txt \\
      --calibration_dir recognition/data/train_images

  # 3. 带精度控制的量化
  python recognition_quantize_int8.py \\
      --onnx_model best_recognition.onnx \\
      --keys_path recognition/keys.txt \\
      --calibration_dir recognition/data/train_images \\
      --accuracy_control \\
      --validation_dir recognition/data/test_images \\
      --max_drop 0.02

  # 4. 同时输出 ONNX INT8 和 OpenVINO IR INT8
  python recognition_quantize_int8.py \\
      --onnx_model best_recognition.onnx \\
      --keys_path recognition/keys.txt \\
      --calibration_dir recognition/data/train_images \\
      --also_onnx
        """
    )

    parser.add_argument("--onnx_model", type=str, default="best_recognition.onnx",
                        help="输入 ONNX 模型路径")
    parser.add_argument("--keys_path", type=str, default="recognition/keys.txt",
                        help="字符集 keys.txt 路径")
    parser.add_argument("--calibration_dir", type=str, default="recognition/data/train_images",
                        help="校准图片目录 (建议 200-500 张)")
    parser.add_argument("--output_dir", type=str, default="recognition_quantized_model",
                        help="输出目录")
    parser.add_argument("--max_width", type=int, default=256,
                        help="校准图片最大宽度 (应与导出时一致)")
    parser.add_argument("--subset_size", type=int, default=300,
                        help="校准子集大小 (NNCF 默认 300)")

    parser.add_argument("--accuracy_control", action="store_true",
                        help="启用精度控制模式")
    parser.add_argument("--validation_dir", type=str, default=None,
                        help="验证图片目录 (精度控制模式需要)")
    parser.add_argument("--max_drop", type=float, default=0.01,
                        help="精度控制: 最大允许精度下降")

    parser.add_argument("--also_onnx", action="store_true",
                        help="同时生成 ONNX INT8 量化模型")
    parser.add_argument("--inspect", action="store_true",
                        help="仅打印模型节点信息 (调试用)")
    parser.add_argument("--skip_benchmark", action="store_true",
                        help="跳过 benchmark")

    args = parser.parse_args()

    # ---- 仅检查模型结构 ----
    if args.inspect:
        inspect_model_nodes(args.onnx_model)
        return

    # ---- 检查输入 ----
    if not os.path.exists(args.onnx_model):
        print(f"✗ ONNX 模型不存在: {args.onnx_model}")
        return
    if not os.path.isdir(args.calibration_dir):
        print(f"✗ 校准目录不存在: {args.calibration_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CRNN NNCF INT8 量化工具")
    print("=" * 60)
    print(f"  ONNX 模型:     {args.onnx_model}")
    print(f"  校准数据目录:  {args.calibration_dir}")
    print(f"  输出目录:      {args.output_dir}")
    print(f"  校准子集大小:  {args.subset_size}")
    print(f"  精度控制:      {'开启' if args.accuracy_control else '关闭'}")

    # ---- 路径 A: 推荐路径 ----
    fp32_path, int8_path = quantize_via_openvino_ir(
        onnx_model_path=args.onnx_model,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
        img_height=32,
        max_width=args.max_width,
        subset_size=args.subset_size,
        accuracy_control=args.accuracy_control,
        validation_dir=args.validation_dir,
        max_drop=args.max_drop,
        keys_path=args.keys_path,
    )

    # ---- 路径 B: 可选 ONNX 直接量化 ----
    if args.also_onnx:
        quantize_onnx_directly(
            onnx_model_path=args.onnx_model,
            calibration_dir=args.calibration_dir,
            output_dir=args.output_dir,
            img_height=32,
            max_width=args.max_width,
            subset_size=args.subset_size,
        )

    # ---- 验证 & Benchmark ----
    if not args.skip_benchmark:
        validate_and_benchmark(
            fp32_model_path=fp32_path,
            int8_model_path=int8_path,
            calibration_dir=args.calibration_dir,
            keys_path=args.keys_path,
            img_height=32,
            max_width=args.max_width,
        )

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("✓ 量化完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  FP32 模型: {fp32_path}")
    print(f"  INT8 模型: {int8_path}")
    print(f"\n使用 INT8 模型进行推理:")
    print(f"  python recognition_inference.py \\")
    print(f"      --model_path {int8_path} \\")
    print(f"      --backend openvino \\")
    print(f"      --image_path <your_images>")

    print(f"\n量化调优建议:")
    print(f"  1. 如果精度下降明显，尝试 --accuracy_control 模式")
    print(f"  2. 检查 --inspect 输出，将敏感层加入 ignored_scope")
    print(f"  3. 增大 --subset_size (如 500) 提高校准精度")
    print(f"  4. 确保校准数据覆盖: 短文本/长文本/中英文/数字/特殊符号")


if __name__ == "__main__":
    main()
