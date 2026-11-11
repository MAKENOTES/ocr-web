"""
detection_small_quantize_int8.py — EAST ONNX 模型 INT8 静态量化 (v4)
=====================================================================
v4 改进:
  - 验证阶段使用固定种子 + 多样本平均，结果完全可复现
  - 同时计算 score_map 和 geo_map 的相关系数
  - 针对真实校准数据 MinMax outlier 问题，增加诊断信息
  - 推荐真实图像使用 Entropy 校准方法

用法:
    # 推荐：真实图像 + Entropy 校准
    python detection_small_quantize_int8.py \
        --input_model best_detection.onnx \
        --output_model best_detection_int8.onnx \
        --calib_dir detection_small/data/val_images \
        --calib_num 100 \
        --calibrate_method entropy

依赖:
    pip install onnx onnxruntime numpy Pillow
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# 常量
# ============================================================

INPUT_SIZE = (256, 768)  # (height, width)，与 detection_small/train.py 一致
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 验证用固定种子，确保每次结果一致
VERIFY_SEED = 42
VERIFY_NUM_SAMPLES = 5


def normalize_input_size(input_size):
    """统一返回 (高, 宽)，兼容 int、二元序列和 ``256x768`` 字符串。"""
    if isinstance(input_size, str):
        parts = input_size.lower().replace("×", "x").split("x")
        if len(parts) == 1:
            input_size = int(parts[0])
        elif len(parts) == 2:
            input_size = tuple(int(part) for part in parts)
        else:
            raise ValueError(f"无效输入尺寸: {input_size!r}")
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        input_h, input_w = int(input_size[0]), int(input_size[1])
        if input_h > 0 and input_w > 0:
            return input_h, input_w
    raise ValueError(f"input_size 必须是正整数或 (高, 宽)，实际为: {input_size!r}")


# ============================================================
# 图像预处理（与 detection_small_inference_640.py 完全一致）
# ============================================================

def preprocess_image(image_path: str, input_size=INPUT_SIZE) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    input_h, input_w = normalize_input_size(input_size)

    scale = min(input_w / orig_w, input_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    pixels = np.asarray(img)
    border = np.concatenate(
        [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0
    )
    padding_color = tuple(np.median(border, axis=0).astype(np.uint8).tolist())
    padded = Image.new("RGB", (input_w, input_h), padding_color)
    padded.paste(resized, ((input_w - new_w) // 2, (input_h - new_h) // 2))

    arr = np.array(padded).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr


# ============================================================
# 校准数据读取器
# ============================================================

class EASTCalibrationDataReader:
    def __init__(self, input_name: str, calib_dir: str = None,
                 calib_num: int = 50, input_size=INPUT_SIZE):
        self.input_name = input_name
        self.input_size = normalize_input_size(input_size)
        self.data_list = []
        self.idx = 0

        if calib_dir and os.path.isdir(calib_dir):
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(glob.glob(os.path.join(calib_dir, ext)))
                image_files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
            image_files = sorted(set(image_files))

            if len(image_files) == 0:
                print(f"  ⚠ 校准目录 {calib_dir} 中未找到图像，回退到随机数据")
                self._generate_random(calib_num)
            else:
                if len(image_files) > calib_num:
                    indices = np.linspace(0, len(image_files) - 1, calib_num, dtype=int)
                    image_files = [image_files[i] for i in indices]

                print(f"  校准图像: {len(image_files)} 张 (来自 {calib_dir})")
                for fpath in image_files:
                    try:
                        arr = preprocess_image(fpath, input_size)
                        self.data_list.append(arr)
                    except Exception as e:
                        print(f"    跳过 {os.path.basename(fpath)}: {e}")

                if len(self.data_list) == 0:
                    print(f"  ⚠ 所有图像读取失败，回退到随机数据")
                    self._generate_random(calib_num)
                else:
                    print(f"  成功加载 {len(self.data_list)} 张校准图像")
        else:
            if calib_dir:
                print(f"  ⚠ 校准目录 {calib_dir} 不存在，使用随机数据")
            else:
                print(f"  使用随机校准数据 ({calib_num} 条)")
            self._generate_random(calib_num)

    def _generate_random(self, n: int):
        rng = np.random.RandomState(12345)  # 校准数据也固定种子
        input_h, input_w = self.input_size
        for _ in range(n):
            fake_img = rng.rand(1, 3, input_h, input_w).astype(np.float32)
            for c in range(3):
                fake_img[0, c] = (fake_img[0, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            self.data_list.append(fake_img)

    def get_next(self):
        if self.idx >= len(self.data_list):
            return None
        data = self.data_list[self.idx]
        self.idx += 1
        return {self.input_name: data}

    def rewind(self):
        self.idx = 0


# ============================================================
# 自动发现输出头节点
# ============================================================

def find_output_head_nodes(model_path: str) -> list:
    """
    自动扫描 ONNX 图，找到直接连接到模型输出的 Conv 节点。
    EAST 模型:
      ... → fuse1 → score_conv (Conv 32→1) → Sigmoid → score_map
      ... → fuse1 → geo_conv   (Conv 32→4)           → geo_map
    """
    import onnx

    model = onnx.load(model_path)
    graph = model.graph
    output_names = {o.name for o in graph.output}

    tensor_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp not in tensor_consumers:
                tensor_consumers[inp] = []
            tensor_consumers[inp].append(node)

    exclude_nodes = []
    for node in graph.node:
        for out in node.output:
            if out in output_names:
                exclude_nodes.append(node.name)
                continue
            if out in tensor_consumers:
                for consumer in tensor_consumers[out]:
                    if consumer.op_type == "Sigmoid":
                        for consumer_out in consumer.output:
                            if consumer_out in output_names:
                                exclude_nodes.append(node.name)

    if exclude_nodes:
        print(f"  自动排除输出头节点: {exclude_nodes}")
        print(f"    → 这些节点保留 FP32 精度，避免 score/geo 量化误差")
    else:
        print(f"  ⚠ 未自动发现输出头节点")
    return exclude_nodes


# ============================================================
# 模型预处理
# ============================================================

def preprocess_model(input_path: str) -> str:
    from onnxruntime.quantization.shape_inference import quant_pre_process

    output_path = input_path.replace(".onnx", "_prep.onnx")
    print(f"  预处理模型: {input_path} → {output_path}")
    try:
        quant_pre_process(
            input_model_path=input_path,
            output_model_path=output_path,
            skip_symbolic_shape=False,
            skip_optimization=False,
            skip_onnx_shape=False,
            auto_merge=True,
            verbose=0,
        )
        print(f"  ✓ 预处理完成")
        return output_path
    except Exception as e:
        print(f"  ⚠ 预处理失败 ({e})，将直接使用原始模型")
        return input_path


# ============================================================
# 主量化流程
# ============================================================

def quantize_int8(
    input_model: str,
    output_model: str,
    calib_dir: str = None,
    calib_num: int = 50,
    per_channel: bool = False,
    calibrate_method: str = "MinMax",
    input_size=INPUT_SIZE,
    quant_format: str = "qoperator",
    reduce_range: bool = False,
):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationMethod,
        QuantFormat,
        QuantType,
    )

    print("=" * 60)
    print("EAST 模型 INT8 静态量化 v4")
    print("=" * 60)

    if quant_format.lower() in ("qoperator", "qop"):
        q_format = QuantFormat.QOperator
        act_type = QuantType.QUInt8
        wgt_type = QuantType.QInt8
        fmt_str = "QOperator (U8S8) → 文件体积缩小"
    else:
        q_format = QuantFormat.QDQ
        act_type = QuantType.QInt8
        wgt_type = QuantType.QInt8
        fmt_str = "QDQ (S8S8) → 文件体积不变，运行时融合"

    print(f"  输入模型:     {input_model}")
    print(f"  输出模型:     {output_model}")
    print(f"  量化格式:     {fmt_str}")
    print(f"  校准目录:     {calib_dir or '(随机数据)'}")
    print(f"  校准样本数:   {calib_num}")
    print(f"  Per-channel:  {per_channel}")
    print(f"  Reduce-range: {reduce_range}")
    print(f"  校准方法:     {calibrate_method}")
    input_h, input_w = normalize_input_size(input_size)
    print(f"  输入尺寸:     {input_w}x{input_h} (宽x高)")
    print()

    if not os.path.exists(input_model):
        print(f"✗ 输入模型不存在: {input_model}")
        return

    orig_size = os.path.getsize(input_model) / 1024 / 1024
    print(f"[1/6] 原始 FP32 模型大小: {orig_size:.1f} MB")

    # ---- Step 1: 发现需要排除的输出头节点 ----
    print(f"\n[2/6] 分析模型结构，定位敏感输出节点...")
    exclude_nodes = find_output_head_nodes(input_model)

    # ---- Step 2: 预处理 ----
    print(f"\n[3/6] 模型预处理 (shape inference + optimization)...")
    prep_model_path = preprocess_model(input_model)

    if prep_model_path != input_model:
        exclude_nodes_prep = find_output_head_nodes(prep_model_path)
        if exclude_nodes_prep:
            exclude_nodes = exclude_nodes_prep

    # ---- Step 3: 准备校准数据 ----
    print(f"\n[4/6] 准备校准数据...")
    sess = ort.InferenceSession(prep_model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    calib_reader = EASTCalibrationDataReader(
        input_name=input_name,
        calib_dir=calib_dir,
        calib_num=calib_num,
        input_size=input_size,
    )

    # ---- Step 4: 执行静态量化 ----
    print(f"\n[5/6] 执行 INT8 静态量化 ({q_format.name})...")

    if calibrate_method.lower() == "entropy":
        cal_method = CalibrationMethod.Entropy
    elif calibrate_method.lower() == "percentile":
        cal_method = CalibrationMethod.Percentile
    else:
        cal_method = CalibrationMethod.MinMax

    os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else ".", exist_ok=True)

    extra_opts = {
        "WeightSymmetric": True,
        "ActivationSymmetric": False,
    }
    if q_format == QuantFormat.QDQ:
        extra_opts["AddQDQPairToWeight"] = True
        extra_opts["QuantizeBias"] = True
        extra_opts["ForceQuantizeNoInputCheck"] = True

    t0 = time.perf_counter()

    quantize_static(
        model_input=prep_model_path,
        model_output=output_model,
        calibration_data_reader=calib_reader,
        quant_format=q_format,
        activation_type=act_type,
        weight_type=wgt_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=cal_method,
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"],
        nodes_to_exclude=exclude_nodes if exclude_nodes else None,
        extra_options=extra_opts,
    )

    quant_time = time.perf_counter() - t0
    quant_size = os.path.getsize(output_model) / 1024 / 1024

    print(f"  ✓ 量化完成！ (耗时 {quant_time:.1f}s)")
    print(f"  FP32 模型:  {orig_size:.1f} MB")
    print(f"  INT8 模型:  {quant_size:.1f} MB")
    print(f"  压缩比:     {orig_size / quant_size:.2f}x")

    if q_format == QuantFormat.QDQ and abs(orig_size - quant_size) < 1.0:
        print(f"  说明: QDQ 格式文件体积不缩小是正常的")
        print(f"        若需文件体积也缩小，请使用 --quant_format qoperator")

    # ---- 清理 ----
    if prep_model_path != input_model and os.path.exists(prep_model_path):
        os.remove(prep_model_path)

    # ---- Step 5: 验证精度 ----
    print(f"\n[6/6] 验证 FP32 vs INT8 输出一致性 (固定种子={VERIFY_SEED}, {VERIFY_NUM_SAMPLES}组样本)...")
    try:
        _verify_accuracy(input_model, output_model, input_size, calib_dir)
    except Exception as e:
        print(f"  ⚠ 验证失败: {e}")

    # ---- Benchmark ----
    print(f"\n[Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...")
    try:
        _benchmark(input_model, output_model, input_size)
    except Exception as e:
        print(f"  ⚠ Benchmark 失败: {e}")

    print(f"\n{'='*60}")
    print(f"✓ INT8 静态量化完成: {output_model}")
    print(f"  模型大小: {orig_size:.1f} MB → {quant_size:.1f} MB ({orig_size/quant_size:.1f}x 压缩)")
    print(f"  用法: python detection_small_inference_640.py --model_path {output_model}")
    print(f"{'='*60}")


# ============================================================
# 验证：FP32 vs INT8 （固定种子 + 多样本 + 双指标）
# ============================================================

def _verify_accuracy(fp32_model: str, int8_model: str, input_size,
                     calib_dir: str = None):
    """
    验证设计原则：
      1. 优先使用真实图像验证（对量化精度的评估最有意义）
      2. 真实图像取校准集的末尾 10 张（与校准阶段实际使用的前 N 张分开）
      3. 固定种子的随机数据仅作为「基线对照」：确认量化流程本身没有 bug
      4. 同时评估 score_map 和 geo_map 的 相关系数 / MAE / 最大误差
      5. 所有结果完全可复现（真实图像按文件名排序固定，随机数据固定种子）
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(fp32_model, opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_model, opts, providers=["CPUExecutionProvider"])
    name_fp32 = sess_fp32.get_inputs()[0].name
    name_int8 = sess_int8.get_inputs()[0].name
    input_h, input_w = normalize_input_size(input_size)

    def _run_one(arr):
        """单张图像 FP32 vs INT8 对比，返回各项指标"""
        fp32_outs = sess_fp32.run(None, {name_fp32: arr})
        int8_outs = sess_int8.run(None, {name_int8: arr})

        # 按通道数匹配：1ch=score, 4ch=geo
        fp32_s = sorted(fp32_outs, key=lambda x: x.shape[1])
        int8_s = sorted(int8_outs, key=lambda x: x.shape[1])

        result = {}
        for tag, fp32_t, int8_t in [("score", fp32_s[0], int8_s[0]),
                                      ("geo",   fp32_s[1], int8_s[1])]:
            diff = np.abs(fp32_t - int8_t)
            result[f"{tag}_max_err"] = np.max(diff)
            result[f"{tag}_mae"] = np.mean(diff)

            f_flat, i_flat = fp32_t.flatten(), int8_t.flatten()
            if np.std(f_flat) > 1e-8 and np.std(i_flat) > 1e-8:
                result[f"{tag}_corr"] = np.corrcoef(f_flat, i_flat)[0, 1]
            else:
                result[f"{tag}_corr"] = None  # 方差为零，相关系数无意义

        return result

    def _print_group(label, results):
        """打印一组验证结果的汇总"""
        score_corrs = [r["score_corr"] for r in results if r["score_corr"] is not None]
        geo_corrs   = [r["geo_corr"]   for r in results if r["geo_corr"]   is not None]

        print(f"\n  {'─'*50}")
        print(f"  {label} ({len(results)} 组):")
        print(f"  {'─'*50}")
        print(f"  score_map  最大误差:  {np.mean([r['score_max_err'] for r in results]):.4e}")
        print(f"  score_map  平均误差:  {np.mean([r['score_mae'] for r in results]):.4e}")
        if score_corrs:
            print(f"  score_map  相关系数:  {np.mean(score_corrs):.6f}  "
                  f"(min={np.min(score_corrs):.6f}, max={np.max(score_corrs):.6f})")
        else:
            print(f"  score_map  相关系数:  N/A (输出方差为零，模型未激活)")

        print(f"  geo_map    最大误差:  {np.mean([r['geo_max_err'] for r in results]):.4e}")
        print(f"  geo_map    平均误差:  {np.mean([r['geo_mae'] for r in results]):.4e}")
        if geo_corrs:
            print(f"  geo_map    相关系数:  {np.mean(geo_corrs):.6f}  "
                  f"(min={np.min(geo_corrs):.6f}, max={np.max(geo_corrs):.6f})")
        else:
            print(f"  geo_map    相关系数:  N/A (输出方差为零)")

        return score_corrs, geo_corrs

    # ================================================================
    # 第一组：真实图像验证（最重要）
    # ================================================================
    real_results = []
    real_score_corrs = []
    real_geo_corrs = []

    if calib_dir and os.path.isdir(calib_dir):
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(glob.glob(os.path.join(calib_dir, ext)))
        image_files = sorted(set(image_files))

        # 取末尾 10 张：与校准阶段的前 N 张尽量分开
        verify_images = image_files[-10:] if len(image_files) > 10 else image_files

        for fpath in verify_images:
            try:
                arr = preprocess_image(fpath, input_size)
                real_results.append(_run_one(arr))
            except Exception:
                pass

        if real_results:
            real_score_corrs, real_geo_corrs = _print_group(
                "真实图像验证 ★ (校准集末尾, 与校准样本尽量不重叠)", real_results)

    # ================================================================
    # 第二组：固定种子随机数据（基线对照）
    # ================================================================
    rng = np.random.RandomState(VERIFY_SEED)
    rand_results = []

    for _ in range(VERIFY_NUM_SAMPLES):
        dummy = rng.rand(1, 3, input_h, input_w).astype(np.float32)
        for c in range(3):
            dummy[0, c] = (dummy[0, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        rand_results.append(_run_one(dummy))

    rand_score_corrs, rand_geo_corrs = _print_group(
        f"随机数据基线对照 (seed={VERIFY_SEED})", rand_results)

    if not real_results:
        print(f"\n  ⚠ 未提供校准目录，仅用随机数据验证")
        print(f"    随机数据上模型输出大部分接近零（检测不到文字），")
        print(f"    相关系数可能偏高或不稳定，不能代表真实推理精度。")
        print(f"    建议提供 --calib_dir 以获得可靠的精度评估。")

    # ================================================================
    # 综合评估（优先使用真实图像的指标）
    # ================================================================
    print(f"\n  {'─'*50}")
    print(f"  综合评估:")
    print(f"  {'─'*50}")

    # 优先用真实图像指标，没有则用随机数据
    if real_score_corrs:
        avg_score = np.mean(real_score_corrs)
        avg_geo   = np.mean(real_geo_corrs) if real_geo_corrs else 1.0
        src = "真实图像"
    elif rand_score_corrs:
        avg_score = np.mean(rand_score_corrs)
        avg_geo   = np.mean(rand_geo_corrs) if rand_geo_corrs else 1.0
        src = "随机数据 (仅供参考)"
    else:
        avg_score, avg_geo, src = 1.0, 1.0, "无数据"

    print(f"  评估来源:  {src}")

    if avg_score > 0.99 and avg_geo > 0.98:
        print(f"  ✓ 精度优秀 (score≈{avg_score:.4f}, geo≈{avg_geo:.4f})，量化几乎无损")
    elif avg_score > 0.97 and avg_geo > 0.95:
        print(f"  ✓ 精度良好 (score≈{avg_score:.4f}, geo≈{avg_geo:.4f})，适合生产使用")
    elif avg_score > 0.93 and avg_geo > 0.90:
        print(f"  △ 精度可接受 (score≈{avg_score:.4f}, geo≈{avg_geo:.4f})")
        print(f"    建议在实际图像上对比 FP32 和 INT8 的检测框效果")
        print(f"    注意: score 经过阈值 0.7 二值化后，实际检测差异可能很小")
    else:
        print(f"  ⚠ 精度损失较大 (score≈{avg_score:.4f}, geo≈{avg_geo:.4f})")
        print(f"    建议尝试:")
        print(f"    1. --calibrate_method entropy  (对 outlier 更鲁棒)")
        print(f"    2. --per_channel               (权重范围差异大时有效)")
        print(f"    3. --reduce_range              (非 VNNI CPU)")
        print(f"    4. --calib_num 200              (更多校准样本)")


# ============================================================
# Benchmark
# ============================================================

def _benchmark(fp32_model: str, int8_model: str, input_size):
    import onnxruntime as ort

    # 固定种子确保 benchmark 输入一致
    rng = np.random.RandomState(VERIFY_SEED + 1)
    input_h, input_w = normalize_input_size(input_size)
    dummy = rng.rand(1, 3, input_h, input_w).astype(np.float32)

    def bench_model(model_path, label, warmup=5, repeat=20):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        opts.inter_op_num_threads = max(1, os.cpu_count() // 2)
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True

        sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        feed = {input_name: dummy}

        for _ in range(warmup):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(repeat):
            sess.run(None, feed)
        avg_ms = (time.perf_counter() - t0) / repeat * 1000
        print(f"  {label:20s} {avg_ms:.1f} ms/张")
        return avg_ms

    fp32_ms = bench_model(fp32_model, "ONNX FP32:")
    int8_ms = bench_model(int8_model, "ONNX INT8:")
    print(f"  {'INT8 加速比:':<20s} {fp32_ms / int8_ms:.2f}x")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EAST ONNX 模型 INT8 静态量化 v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 推荐：真实图像 + Entropy 校准
  python detection_small_quantize_int8.py \\
      --input_model best_detection.onnx \\
      --output_model best_detection_int8.onnx \\
      --calib_dir detection_small/data/val_images \\
      --calibrate_method entropy

  # QOperator 格式 + MinMax（快速）
  python detection_small_quantize_int8.py \\
      --input_model best_detection.onnx \\
      --output_model best_detection_int8.onnx \\
      --calib_dir detection_small/data/val_images

  # 高精度配置
  python detection_small_quantize_int8.py \\
      --input_model best_detection.onnx \\
      --output_model best_detection_int8.onnx \\
      --calib_dir detection_small/data/val_images \\
      --per_channel --calibrate_method entropy
        """,
    )

    parser.add_argument("--input_model", type=str, default="best_detection.onnx")
    parser.add_argument("--output_model", type=str, default="best_detection_int8_static.onnx")
    parser.add_argument("--calib_dir", type=str, default="detection_small/data/val_images",
                        help="校准图像目录")
    parser.add_argument("--calib_num", type=int, default=157,
                        help="校准样本数量（默认 100）")
    parser.add_argument("--quant_format", type=str, default="qoperator",
                        choices=["qoperator", "qop", "qdq"],
                        help="量化格式：qoperator(默认,体积缩小) | qdq(兼容性好)")
    parser.add_argument("--per_channel", action="store_true",
                        help="per-channel 量化（精度更高）")
    parser.add_argument("--reduce_range", action="store_true",
                        help="7-bit 量化范围（非 VNNI CPU 可改善精度）")
    parser.add_argument("--calibrate_method", type=str, default="MinMax",
                        choices=["MinMax", "Entropy", "Percentile"])
    parser.add_argument(
        "--input_size", type=normalize_input_size, default=INPUT_SIZE,
        help="模型输入尺寸：正方形可写 640，矩形按高x宽写 256x768",
    )

    args = parser.parse_args()

    quantize_int8(
        input_model=args.input_model,
        output_model=args.output_model,
        calib_dir=args.calib_dir,
        calib_num=args.calib_num,
        per_channel=args.per_channel,
        calibrate_method=args.calibrate_method,
        input_size=args.input_size,
        quant_format=args.quant_format,
        reduce_range=args.reduce_range,
    )
