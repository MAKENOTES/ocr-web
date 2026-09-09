"""CRNN 静态 INT8 量化：共享 RGB 预处理、动态宽度、真实标注 CER 门禁。

默认只量化 CNN，保留 GRU 与分类层 FP32。--quantize_head 可探索更高压缩率，
必须用独立带标注数据验证。--accuracy_control 使用 NNCF 的 1-CER 指标恢复精度，
并在保存前再次检查 CER 增幅；max_drop=0.001 表示最多增加 0.1 个百分点。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from Levenshtein import distance
from torch.utils.data import DataLoader

from recognition.components.converter import LabelConverter
from recognition.components.dataset import (
    InferenceDataset, TextRecognitionDataset, dynamic_padding_collate_fn,
)
from recognition_benchmark_cpu import evaluate_backend, select_indices
from recognition_inference import OpenVINOBackend, normalize_output_to_tbc
from recognition.utils.openvino import make_gru_length_safe


class CRNNCalibrationDataset:
    """与部署完全相同的完整文字框预处理；默认不限宽，不压缩长文本。"""

    def __init__(self, image_dir, img_height=32, max_width=None, num_channels=3,
                 subset_size=300, seed=20260908):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        paths = sorted(str(path) for path in Path(image_dir).rglob('*')
                       if path.is_file() and path.suffix.lower() in extensions)
        if not paths:
            raise FileNotFoundError(f"未找到校准图片: {image_dir}")
        indices = select_indices(len(paths), subset_size, seed)
        self.image_paths = [paths[i] for i in indices]
        self.dataset = InferenceDataset(self.image_paths, img_height, num_channels,
                                        img_width=max_width)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 在添加 batch 维之前复用真实推理 collate，保留每图 32px 右侧保护区。
        images, _, widths = dynamic_padding_collate_fn([self.dataset[index]])
        return images[0].numpy(), np.int64(widths[0])


def create_calibration_dataloader(image_dir, img_height=32, max_width=None,
                                  subset_size=300, num_channels=3, seed=20260908):
    dataset = CRNNCalibrationDataset(image_dir, img_height, max_width, num_channels,
                                     subset_size, seed)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def transform_calibration(data_item):
    images, widths = data_item
    return {'input': np.asarray(images, dtype=np.float32),
            'img_widths': np.asarray(widths, dtype=np.int64).reshape(-1)}


def model_channels(model):
    image_input = next((port for port in model.inputs if len(port.partial_shape) == 4), None)
    if image_input is None or image_input.partial_shape[1].is_dynamic:
        raise ValueError("模型必须包含通道数固定的四维图片输入")
    channels = image_input.partial_shape[1].get_length()
    if channels not in (1, 3):
        raise ValueError(f"不支持的输入通道数: {channels}")
    if not any(len(port.partial_shape) == 1 for port in model.inputs):
        raise ValueError("模型缺少 img_widths；请重新导出长度感知模型")
    return channels


def ov_feeds(model, images, widths):
    # 转换后的宽度输入可能不再名为 img_widths；按 rank 找端口。
    return {port.any_name: images if len(port.partial_shape) == 4 else widths
            for port in model.inputs}


def verify_openvino_conversion(onnx_path, model, calibration_loader, max_samples=16):
    """量化前先校对 ONNX 与 OpenVINO，防止两个错误 IR 彼此一致而蒙混过关。"""
    import onnxruntime as ort
    import openvino as ov
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    reference = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    converted = ov.Core().compile_model(model, 'CPU', {
        'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 1, 'NUM_STREAMS': 1})
    max_error = 0.0
    count = 0
    for item in calibration_loader:
        feeds = transform_calibration(item)
        expected = reference.run(None, feeds)[0]
        actual = np.asarray(converted(ov_feeds(model, feeds['input'], feeds['img_widths']))[0])
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=1e-3):
            raise RuntimeError(f"OpenVINO FP32 与 ONNX 不一致，禁止继续量化；校准样本 {count}")
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        count += 1
        if count >= max_samples:
            break
    print(f"OpenVINO 转换校验: {count} 张，最大误差 {max_error:.3e}", flush=True)
    return dict(samples=count, max_absolute_error=max_error)


def make_validation_fn(converter):
    def validation_fn(compiled_model, data_items):
        # NNCF 传入的是 CompiledModel 和原始数据项的 iterable。
        # 返回全局字符加权的 1-CER，禁止用受 -1e9 padding 主导的余弦相似度。
        edits = characters = 0
        for images, labels, widths in data_items:
            result = compiled_model(ov_feeds(compiled_model, images.numpy(),
                                            widths.numpy().astype(np.int64)))[0]
            output = normalize_output_to_tbc(np.asarray(result), len(labels))
            texts = converter.decode(torch.from_numpy(output), layout='TBC')
            for label, text in zip(labels, texts):
                edits += distance(label, text)
                characters += len(label)
        if not characters:
            raise ValueError("精度控制验证集没有有效文字")
        return 1.0 - edits / characters, None
    return validation_fn


def quantize_via_openvino_ir(onnx_model_path, calibration_dir, output_dir,
                             img_height=32, max_width=None, subset_size=300,
                             accuracy_control=False, validation_dir=None,
                             max_drop=0.001, keys_path='recognition/keys.txt',
                             validation_annotation=None, validation_size=1000,
                             seed=20260908, quantize_head=False, num_threads=4):
    import nncf
    import openvino as ov
    if subset_size < 1 or validation_size < 0 or max_drop < 0 or num_threads < 1:
        raise ValueError("subset_size/num_threads 必须为正，validation_size/max_drop 不能为负")
    if accuracy_control and not validation_annotation:
        raise ValueError("--accuracy_control 必须提供 --validation_annotation 真实标注；图片目录无法计算 CER")
    if max_width is not None:
        raise ValueError("部署默认不限宽；校准不能单独设置 max_width，避免改变长文本形状")
    torch.set_num_threads(num_threads)
    model = make_gru_length_safe(ov.convert_model(onnx_model_path))
    channels = model_channels(model)
    target = Path(output_dir)
    fp32_path = target / 'fp32/crnn_fp32.xml'
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, fp32_path, compress_to_fp16=False)
    loader = create_calibration_dataloader(calibration_dir, img_height, None,
                                           subset_size, channels, seed)
    conversion_check = verify_openvino_conversion(onnx_model_path, model, loader)
    def transform(item):
        data = transform_calibration(item)
        return ov_feeds(model, data['input'], data['img_widths'])
    calibration = nncf.Dataset(loader, transform)
    types = ['GRUSequence', 'GRUCell']
    if not quantize_head:
        types.append('MatMul')
    present_types = {op.get_type_name() for op in model.get_ordered_ops()}
    ignored = nncf.IgnoredScope(types=[name for name in types if name in present_types])
    options = dict(subset_size=len(loader.dataset), preset=nncf.QuantizationPreset.MIXED,
                   target_device=nncf.TargetDevice.CPU, fast_bias_correction=False,
                   ignored_scope=ignored)
    manifest = dict(onnx_model=str(onnx_model_path), seed=seed, num_channels=channels,
                    calibration_paths=loader.dataset.image_paths, quantize_head=quantize_head,
                    accuracy_control=accuracy_control, max_drop=max_drop,
                    validation_annotation=validation_annotation, validation_size=validation_size,
                    conversion_check=conversion_check)
    if accuracy_control:
        from torch.utils.data import Subset
        validation = TextRecognitionDataset(validation_annotation, img_height, None, channels, is_train=False)
        indices = select_indices(len(validation), validation_size, seed)
        calibration_paths = {Path(path).resolve() for path in loader.dataset.image_paths}
        overlap = calibration_paths.intersection(Path(validation.image_paths[i]).resolve() for i in indices)
        if overlap:
            raise ValueError(f"校准与精度控制数据重叠: {next(iter(overlap))}")
        validation_loader = DataLoader(Subset(validation, indices), batch_size=1,
                                       collate_fn=dynamic_padding_collate_fn)
        validation_data = nncf.Dataset(validation_loader, lambda item: ov_feeds(
            model, item[0].numpy(), item[2].numpy().astype(np.int64)))
        metric = make_validation_fn(LabelConverter(keys_path))
        quantized = nncf.quantize_with_accuracy_control(
            model, calibration, validation_data, metric, max_drop=max_drop,
            drop_type=nncf.DropType.ABSOLUTE, **options)
        core = ov.Core()
        config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32',
                  'INFERENCE_NUM_THREADS': num_threads, 'NUM_STREAMS': 1}
        before = metric(core.compile_model(model, 'CPU', config), validation_loader)[0]
        after = metric(core.compile_model(quantized, 'CPU', config), validation_loader)[0]
        manifest.update(fp32_cer=1-before, int8_cer=1-after,
                        validation_paths=[validation.image_paths[i] for i in indices])
        if before - after > max_drop + 1e-12:
            raise RuntimeError(f"INT8 未通过 CER 门禁: 增加 {before-after:.6f} > {max_drop}")
    else:
        quantized = nncf.quantize(model, calibration, **options)
    int8_path = target / 'int8/crnn_int8.xml'
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, int8_path, compress_to_fp16=False)
    (target / 'quantization.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"FP32: {fp32_path}\nINT8: {int8_path}", flush=True)
    return str(fp32_path), str(int8_path)


def quantize_onnx_directly(onnx_model_path, calibration_dir, output_dir,
                           img_height=32, max_width=None, subset_size=300,
                           seed=20260908, quantize_head=False):
    """可选 NNCF ONNX 路径，共享与 OpenVINO 相同的校准输入和排除策略。"""
    import nncf
    import onnx
    import openvino as ov
    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters, QuantizationParameters
    if max_width is not None:
        raise ValueError("校准必须与部署相同，不能单独限制 max_width")
    model = onnx.load(onnx_model_path)
    channels = model_channels(ov.convert_model(onnx_model_path))
    loader = create_calibration_dataloader(calibration_dir, img_height, None, subset_size, channels, seed)
    types = ['GRU'] + ([] if quantize_head else ['MatMul', 'Gemm'])
    present_types = {node.op_type for node in model.graph.node}
    quantized = nncf.quantize(
        model, nncf.Dataset(loader, transform_calibration), subset_size=len(loader.dataset),
        preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.CPU,
        fast_bias_correction=False,
        # ORT CPU 的 QLinearConv 不支持逐通道激活零点；权重仍可逐通道量化。
        advanced_parameters=AdvancedQuantizationParameters(
            activations_quantization_params=QuantizationParameters(per_channel=False)),
        ignored_scope=nncf.IgnoredScope(types=[name for name in types if name in present_types]))
    target = Path(output_dir) / 'int8_onnx/crnn_int8.onnx'
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(quantized, target)
    onnx.checker.check_model(str(target))
    from recognition_inference import ONNXBackend
    runtime = ONNXBackend(str(target), num_threads=1)
    # 实际执行，不能把“能保存 ONNX 文件”当成 ORT 可部署。
    for index, item in enumerate(loader):
        feeds = transform_calibration(item)
        runtime.infer(torch.from_numpy(feeds['input']), torch.from_numpy(feeds['img_widths']))
        if index >= 3:
            break
    ir_path = target.with_suffix('.xml')
    ov.save_model(make_gru_length_safe(ov.convert_model(target)), ir_path, compress_to_fp16=False)
    return str(target), str(ir_path)


def validate_and_benchmark(fp32_model_path, int8_model_path, calibration_dir,
                            keys_path, img_height=32, max_width=None,
                            validation_annotation=None, limit=1000, num_threads=4,
                            output_path=None):
    if not validation_annotation:
        raise ValueError("验证准确率必须提供 --validation_annotation，或用 --skip_benchmark 仅生成模型")
    torch.set_num_threads(1)
    converter = LabelConverter(keys_path)
    reports, all_predictions = {}, {}
    for name, path in [('fp32', fp32_model_path), ('int8', int8_model_path)]:
        backend = OpenVINOBackend(path, num_threads)
        dataset = TextRecognitionDataset(validation_annotation, img_height, None,
                                          backend.num_channels, is_train=False)
        metrics, predictions = evaluate_backend(backend, dataset, converter, limit=limit)
        reports[name] = metrics
        all_predictions[name] = predictions
        del backend
    reports['cer_increase'] = reports['int8']['cer'] - reports['fp32']['cer']
    reports['text_agreement'] = sum(a['prediction'] == b['prediction'] for a, b in
                                    zip(all_predictions['fp32'], all_predictions['int8'])) / reports['fp32']['samples']
    if output_path:
        Path(output_path).write_text(json.dumps(dict(metrics=reports, predictions=all_predictions),
                                                ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return reports


def inspect_model_nodes(onnx_model_path):
    import openvino as ov
    for op in ov.convert_model(onnx_model_path).get_ordered_ops():
        if op.get_type_name() in {'Convolution', 'GroupConvolution', 'MatMul', 'GRUSequence', 'GRUCell'}:
            print(op.get_type_name(), op.get_friendly_name())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx_model', default='best_recognition.onnx')
    parser.add_argument('--keys_path', default='recognition/keys.txt')
    parser.add_argument('--calibration_dir', default='recognition/data/train_images')
    parser.add_argument('--output_dir', default='outputs/recognition_quantized_model')
    parser.add_argument('--subset_size', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--max_width', type=int, default=None, help='仅用于显式拦截旧参数；当前禁止校准单独限宽')
    parser.add_argument('--accuracy_control', action='store_true')
    parser.add_argument('--validation_annotation', help='精度控制/评估用真实标注文件，路径相对标注文件')
    parser.add_argument('--validation_dir', help='旧参数；不能替代带文字标签的 validation_annotation')
    parser.add_argument('--validation_size', type=int, default=1000, help='0 为全部，否则固定种子随机抽样')
    parser.add_argument('--max_drop', type=float, default=0.001, help='CER 最大绝对增幅，0.001 = 0.1 个百分点')
    parser.add_argument('--quantize_head', action='store_true', help='额外量化分类层，必须实测 CER')
    parser.add_argument('--also_onnx', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--skip_benchmark', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect_model_nodes(args.onnx_model)
        return
    if (args.accuracy_control or not args.skip_benchmark) and not args.validation_annotation:
        parser.error('需要 --validation_annotation 计算真实 CER；仅量化可加 --skip_benchmark')
    if args.accuracy_control and args.also_onnx:
        parser.error('--also_onnx 不支持精度恢复，不能与 --accuracy_control 混用')
    fp32, int8 = quantize_via_openvino_ir(
        args.onnx_model, args.calibration_dir, args.output_dir, max_width=args.max_width,
        subset_size=args.subset_size, accuracy_control=args.accuracy_control,
        max_drop=args.max_drop, keys_path=args.keys_path,
        validation_annotation=args.validation_annotation, validation_size=args.validation_size,
        seed=args.seed, quantize_head=args.quantize_head, num_threads=args.num_threads)
    if args.also_onnx:
        quantize_onnx_directly(args.onnx_model, args.calibration_dir, args.output_dir,
                               subset_size=args.subset_size, seed=args.seed,
                               quantize_head=args.quantize_head)
    if not args.skip_benchmark:
        validate_and_benchmark(fp32, int8, args.calibration_dir, args.keys_path,
                                validation_annotation=args.validation_annotation,
                                limit=args.validation_size, num_threads=args.num_threads,
                                output_path=Path(args.output_dir) / 'validation.json')


if __name__ == '__main__':
    main()
