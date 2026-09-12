"""独立验证文档检测的固定数值契约、数据隔离和整图推理。"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from detection_larger.components.dataset import OCRDataset, read_boxes
from detection_larger.components.eval_metrics import match_counts, summarize_counts
from detection_larger.components.ground_truth import generate_ground_truth
from detection_larger.components.loss import EASTLoss
from detection_larger.components.transforms import get_transforms, normalize_input_size, prepare_image, restore_boxes
from detection_larger.inference import DocumentDetector, decode_maps, suppress_boxes
from detection_larger.models.east_model import checkpoint_config, create_model, load_checkpoint, save_checkpoint


@pytest.mark.parametrize('use_directory,save_crops', [(False, False), (True, True)])
@pytest.mark.parametrize('has_boxes', [True, False])
def test_inference_main_uses_constants(tmp_path, monkeypatch, use_directory, save_crops, has_boxes):
    from detection_larger import inference

    source = tmp_path / 'images'
    source.mkdir()
    image_path = source / 'page.png'
    Image.new('RGB', (40, 30), 'white').save(image_path)
    (source / 'notes.txt').write_text('非图片文件', encoding='utf-8')
    output = tmp_path / 'output'
    model_path = tmp_path / 'model.pth'
    for name, value in {
        'MODEL_PATH': model_path, 'IMAGE_PATH': source if use_directory else image_path,
        'OUTPUT_DIR': output, 'CROP_DIR': tmp_path / 'crops', 'DEVICE': 'cpu', 'SCORE_THRESHOLD': .6,
        'NMS_IOU_THRESHOLD': .2, 'SAVE_CROPS': save_crops,
    }.items():
        monkeypatch.setattr(inference, name, value)

    class FakeDetector:
        def __init__(self, path, device):
            assert (path, device) == (model_path, 'cpu')

        def predict(self, image, threshold, iou_threshold):
            assert image.size == (40, 30)
            assert (threshold, iou_threshold) == (.6, .2)
            return [{'box': [2.25, 3, 20, 15], 'score': .9}] if has_boxes else []

    monkeypatch.setattr(inference, 'DocumentDetector', FakeDetector)
    inference.main()
    assert sorted(path.name for path in output.iterdir()) == ['images', 'labels']
    with Image.open(output / 'images' / 'page.png') as image:
        assert image.size == (40, 30)
        assert image.getpixel((2, 3)) == ((255, 0, 0) if has_boxes else (255, 255, 255))
    dataset = OCRDataset(output / 'images', output / 'labels')
    assert dataset.label_paths == [output / 'labels' / 'gt_page.txt']
    assert sorted(path.name for path in (output / 'labels').iterdir()) == ['gt_page.txt']
    boxes, ignored = read_boxes(dataset.label_paths[0], (40, 30))
    assert boxes.tolist() == ([[2.25, 3, 20, 15]] if has_boxes else [])
    assert not ignored.any()
    if not has_boxes:
        assert dataset.label_paths[0].read_text() == ''
    crop = tmp_path / 'crops' / 'page.png' / 'crop_0000.png'
    assert crop.exists() == (save_crops and has_boxes)
    if save_crops and has_boxes:
        with Image.open(crop) as image:
            assert image.size == (18, 12)
    if not save_crops:
        assert not (tmp_path / 'crops').exists()

    with pytest.raises(FileExistsError, match=str(output)):
        inference.main()
    if save_crops:
        monkeypatch.setattr(inference, 'OUTPUT_DIR', tmp_path / 'new_output')
        with pytest.raises(FileExistsError, match='Crop output already exists'):
            inference.main()
        assert not (tmp_path / 'new_output').exists()


def test_inference_main_rejects_empty_directory(tmp_path, monkeypatch):
    from detection_larger import inference

    source = tmp_path / 'empty'
    source.mkdir()
    output = tmp_path / 'output'
    monkeypatch.setattr(inference, 'IMAGE_PATH', source)
    monkeypatch.setattr(inference, 'OUTPUT_DIR', output)
    with pytest.raises(ValueError, match=str(source)):
        inference.main()
    assert not output.exists()


def test_inference_rejects_duplicate_label_stems(tmp_path, monkeypatch):
    from detection_larger import inference

    source = tmp_path / 'images'
    source.mkdir()
    for name in ('page.jpg', 'page.png'):
        Image.new('RGB', (40, 30), 'white').save(source / name)
    monkeypatch.setattr(inference, 'IMAGE_PATH', source)
    monkeypatch.setattr(inference, 'OUTPUT_DIR', tmp_path / 'output')
    with pytest.raises(ValueError, match='Duplicate image stem for labels:.*page.jpg.*page.png'):
        inference.main()
    assert not (tmp_path / 'output').exists()


def test_decode_maps_keeps_all_regions_above_threshold():
    score = np.zeros((1, 284, 284), dtype=np.float32)
    for y in range(0, 284, 4):
        for x in range(0, 284, 4):
            score[0, y:y + 2, x:x + 2] = .8
    geometry = np.full((4, 284, 284), np.log1p(1), dtype=np.float32)
    boxes, confidence = decode_maps(score, geometry, threshold=.7)
    assert len(boxes) == 71 * 71
    np.testing.assert_allclose(confidence, .8)
    assert len(decode_maps(score, geometry, threshold=.9)[0]) == 0


@pytest.mark.parametrize('threshold', [0, 1, -.1])
def test_decode_maps_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='threshold must be'):
        decode_maps(np.zeros((1, 4, 4)), np.zeros((4, 4, 4)), threshold)


@pytest.mark.parametrize('available,requested,expected', [
    (True, 'auto', 'cuda'), (False, 'auto', 'cpu'), (True, 'cpu', 'cpu'),
    (True, 'cuda:0', 'cuda:0'),
])
def test_detector_selects_pytorch_device(monkeypatch, available, requested, expected):
    from detection_larger import inference

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    calls = []

    def fake_load(path, device):
        calls.append((path, device))
        return object(), {'config': checkpoint_config(64)}

    monkeypatch.setattr(inference, 'load_checkpoint', fake_load)
    detector = DocumentDetector('model.pth', requested)
    assert detector.device == expected
    assert calls == [('model.pth', expected)]


def make_source(tmp_path, count=4):
    images, labels = tmp_path / 'images', tmp_path / 'labels'
    images.mkdir()
    labels.mkdir()
    for index in range(count):
        Image.new('RGB', (96, 64), (index * 20, 100, 200)).save(images / f'{index}.png')
        (labels / f'gt_{index}.txt').write_text('8 8 40 24\n', encoding='utf-8')
    return {'name': 'pdf', 'images': str(images), 'labels': str(labels)}


@pytest.mark.parametrize('line', ['NaN 2 20 30', '20 2 10 30', '0 0 101 20', 'bad label', '1 2 3'])
def test_invalid_labels_include_path_and_line(tmp_path, line):
    path = tmp_path / 'bad.txt'
    path.write_text(line)
    with pytest.raises(ValueError, match=r'bad.txt:1'):
        read_boxes(path, (100, 100))


def test_labels_preserve_numeric_text_ignore_empty_and_explicit_quads(tmp_path):
    path = tmp_path / 'labels.txt'
    path.write_text('1,2,30,40,"123,456"\n2 3 10 20 ###\n')
    boxes, ignored = read_boxes(path, (100, 100))
    assert boxes.shape == (2, 4)
    assert ignored.tolist() == [False, True]
    path.write_text('1 2 30 40 hello,world\n')
    assert read_boxes(path, (100, 100))[0].tolist() == [[1, 2, 30, 40]]
    path.write_text('1,2,30,2,30,40,1,40,text\n')
    assert read_boxes(path, (100, 100), 'quad8')[0].tolist() == [[1, 2, 30, 40]]
    path.write_text('1,2,30,3,30,40,1,40,text\n')
    with pytest.raises(ValueError, match='rotated'):
        read_boxes(path, (100, 100), 'quad8')
    path.write_text('')
    assert read_boxes(path, (100, 100))[0].shape == (0, 4)


def test_nms_keeps_adjacent_cells_and_removes_duplicates():
    boxes = np.array([[0, 0, 30, 20], [1, 0, 31, 20], [32, 0, 62, 20]], dtype=np.float32)
    assert suppress_boxes(boxes, np.array([.9, .8, .7])).tolist() == [0, 2]
    decoded, scores = decode_maps(np.zeros((1, 16, 16)), np.zeros((4, 16, 16)))
    assert decoded.shape == (0, 4) and scores.shape == (0,)


def test_evaluation_counts_blanks_duplicates_misses_and_ignore():
    boxes = np.array([[0, 0, 20, 20], [30, 0, 50, 20]], dtype=np.float32)
    predictions = [{'box': box, 'score': .9} for box in [[0, 0, 20, 20], [0, 0, 20, 20], [35, 2, 45, 18]]]
    assert match_counts(predictions, boxes, np.array([False, True])).tolist() == [1, 1, 0]
    assert match_counts([], boxes, np.array([False, True])).tolist() == [0, 0, 1]
    assert summarize_counts([0, 1, 0])['f1'] == 0
    assert summarize_counts([0, 0, 0])['f1'] == 1


@pytest.mark.parametrize('size', [(32, 17), (2632, 1114), (2480, 3509), (64, 400)])
def test_whole_image_preprocess_matches_inference_and_coordinates_roundtrip(size):
    image = Image.new('RGB', size, (243, 240, 238))
    boxes = np.array([[1, 1, size[0] - 1, size[1] - 1]], dtype=np.float32)
    quads = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    actual = get_transforms(False, 1024)({'image': image, 'quads': quads.copy(), 'ignored': np.array([False])})
    tensor, metadata = prepare_image(image, 1024)
    torch.testing.assert_close(tensor, actual['image'], rtol=0, atol=0)
    scale, offset = metadata
    transformed = boxes * scale + np.array([*offset, *offset])
    np.testing.assert_allclose(restore_boxes(transformed, metadata, size), boxes, atol=0.001)


@pytest.mark.parametrize('input_size', [63, 100, (64, 63), (64,), (64, 96, 128)])
def test_invalid_input_size(input_size):
    with pytest.raises(ValueError):
        normalize_input_size(input_size)


def test_ground_truth_and_loss_match_reference_values_and_gradients():
    quads = np.array([[[8, 8], [72, 8], [72, 32], [8, 32]],
                      [[16, 48], [40, 48], [40, 64], [16, 64]]], dtype=np.float32)
    score, geometry = generate_ground_truth((96, 96), quads)
    # 固定迁移前的数值样本，不再导入另一检测子系统作为运行时参考。
    with np.load(Path(__file__).parent / 'fixtures' / 'detection_larger_reference.npz') as data:
        np.testing.assert_array_equal(score, data['score'])
        np.testing.assert_array_equal(geometry, data['geometry'])
        expected = [torch.from_numpy(data[f'result_{index}']) for index in range(5)]
    truth_score, truth_geo = torch.from_numpy(score)[None, None], torch.from_numpy(geometry)[None]
    pred_score = torch.full_like(truth_score, .7, requires_grad=True)
    pred_geo = torch.full_like(truth_geo, .3, requires_grad=True)
    actual = EASTLoss(lambda_iou=3)(truth_score, pred_score, truth_geo, pred_geo, torch.ones_like(truth_score))
    actual_grads = torch.autograd.grad(actual[0], (pred_score, pred_geo))
    for reference, result in zip(expected, (*actual, *actual_grads)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_ignored_regions_have_no_loss_gradient_and_blank_is_finite():
    sample = get_transforms(False, 64)({
        'image': Image.new('RGB', (64, 64), 'white'),
        'quads': np.array([[[8, 8], [24, 8], [24, 24], [8, 24]]], dtype=np.float32),
        'ignored': np.array([True]),
    })
    assert sample['training_mask'][2:6, 2:6].eq(0).all()
    assert sample['score_map'].sum() == 0
    score = torch.full((1, 1, 16, 16), .8, requires_grad=True)
    geo = torch.zeros((1, 4, 16, 16), requires_grad=True)
    loss = EASTLoss(3)(sample['score_map'][None, None], score, sample['geo_map'][None],
                       geo, sample['training_mask'][None])[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert score.grad[:, :, 2:6, 2:6].eq(0).all()
    assert score.grad.sum() > 0
    assert geo.grad.eq(0).all()


def test_explicit_input_size_matches_training_preprocess_and_checkpoint():
    from detection_larger.train import INPUT_SIZE

    image = Image.new('RGB', (96, 64), 'white')
    tensor, _ = prepare_image(image, INPUT_SIZE)
    sample = get_transforms(is_train=False, input_size=INPUT_SIZE)({
        'image': image, 'quads': np.empty((0, 4, 2), dtype=np.float32),
        'ignored': np.empty(0, dtype=bool),
    })
    assert tensor.shape[-2:] == (INPUT_SIZE, INPUT_SIZE)
    assert checkpoint_config(INPUT_SIZE)['input_size'] == [INPUT_SIZE, INPUT_SIZE]
    torch.testing.assert_close(sample['image'], tensor, rtol=0, atol=0)


def test_whole_image_inference_restores_long_lines_and_cells():
    image = Image.new('RGB', (320, 128), 'white')
    boxes = np.array([[8, 16, 310, 48], [32, 80, 64, 112], [160, 80, 192, 112]], dtype=np.float32)
    quads = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    target = get_transforms(False, 256)({'image': image, 'quads': quads, 'ignored': np.zeros(3, dtype=bool)})
    calls = []

    class PerfectModel:
        def __call__(self, tensor):
            calls.append(tensor.shape)
            return target['score_map'][None, None], target['geo_map'][None]

    detector = DocumentDetector.__new__(DocumentDetector)
    detector.config = checkpoint_config(256)
    detector.device, detector.session, detector.model = 'cpu', None, PerfectModel()
    result = detector.predict(image)
    assert len(calls) == 1
    assert len(result) == 3
    assert match_counts(result, boxes, np.zeros(3, dtype=bool)).tolist() == [3, 0, 0]
    np.testing.assert_allclose(result[0]['box'], boxes[0], atol=0.001)


def test_model_backward_and_checkpoint_contract(tmp_path):
    from detection_larger.components.training_options import load_initial_weights
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model()
        image = torch.randn(2, 3, 64, 64)
        score, geometry = model(image)
        assert score.shape == (2, 1, 16, 16) and geometry.shape == (2, 4, 16, 16)
        loss = EASTLoss(3)(torch.ones_like(score), score, torch.zeros_like(geometry), geometry)[0]
        loss.backward()
        assert torch.isfinite(loss) and model.score_conv.weight.grad.abs().sum() > 0
        path = tmp_path / 'model.pth'
        save_checkpoint(model, path, 64)
        loaded, _ = load_checkpoint(path)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(image)[0], model(image)[0])
        torch.save(model.state_dict(), path)
        with torch.no_grad():
            loaded.score_conv.weight.zero_()
        load_initial_weights(loaded, path)
        torch.testing.assert_close(loaded.score_conv.weight, model.score_conv.weight, rtol=0, atol=0)
        torch.save({'config': {'format': 'document-east-resnet18-v1'}, 'model': model.state_dict()}, path)
        with pytest.raises(ValueError, match='old tiled checkpoints'):
            load_checkpoint(path)
    finally:
        torch.set_num_threads(old_threads)


def test_training_entry_runs_reference_loop_and_saves_top_k(tmp_path, monkeypatch):
    from detection_larger import train as training
    source = make_source(tmp_path)
    val_root = tmp_path / 'validation'
    val_root.mkdir()
    validation = make_source(val_root, count=2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, 'VAL_INTERVAL', 1)
    monkeypatch.setattr(training, 'CHECKPOINT_INTERVAL', 1)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, value in {
            'TRAIN_IMAGE_DIR': source['images'], 'TRAIN_LABEL_DIR': source['labels'],
            'VAL_IMAGE_DIR': validation['images'], 'VAL_LABEL_DIR': validation['labels'],
            'INPUT_SIZE': 64, 'BATCH_SIZE': 2, 'EPOCHS': 1, 'NUM_WORKERS': 0,
            'DEVICE': 'cpu', 'PRETRAINED': False, 'LOG_GRAPH': False,
        }.items():
            monkeypatch.setattr(training, name, value)
        training.main(['--output', str(tmp_path / 'run')])
        assert (tmp_path / 'run' / 'detection_epoch_1.pth').exists()
        best = list((tmp_path / 'run').glob('best_detection_epoch_*.pth'))
        assert len(best) == 1
        model, checkpoint = load_checkpoint(best[0])
        assert checkpoint['config']['input_size'] == [64, 64]
        with torch.no_grad():
            assert torch.isfinite(model(torch.rand(1, 3, 64, 64))[0]).all()
    finally:
        torch.set_num_threads(old_threads)


def test_directory_loader_includes_mixed_formats_and_precise_labels(tmp_path):
    source = make_source(tmp_path, count=1)
    root = Path(source['images'])
    Image.new('RGB', (96, 64), 'white').save(root / 'pdf.JPG')
    Image.new('RGBA', (96, 64), (0, 0, 0, 0)).save(root / 'web.PNG')
    for name in ('pdf', 'web'):
        (Path(source['labels']) / f'{name}.txt').write_text('8 8 40 24')
    dataset = OCRDataset(source['images'], source['labels'], get_transforms(False, 64))
    assert {p.name for p in dataset.image_paths} == {'0.png', 'pdf.JPG', 'web.PNG'}
    for index, path in enumerate(dataset.image_paths):
        assert dataset[index]['image'].shape == (3, 64, 64)
        assert dataset[index]['score_map'].sum() > 0
    raw = OCRDataset(source['images'], source['labels'])
    assert all(raw[i]['image'].mode == 'RGB' for i in range(len(raw)))


def test_directory_loader_rejects_missing_or_ambiguous_labels(tmp_path):
    source = make_source(tmp_path, count=1)
    label_dir = Path(source['labels'])
    (label_dir / '0.txt').write_text('8 8 40 24')
    with pytest.raises(ValueError, match='exactly one label'):
        OCRDataset(source['images'], source['labels'])
    (label_dir / '0.txt').unlink()
    (label_dir / 'gt_0.txt').unlink()
    with pytest.raises(ValueError, match='0.png'):
        OCRDataset(source['images'], source['labels'])


def test_directory_data_uses_same_training_and_inference_preprocess(tmp_path):
    source = make_source(tmp_path, count=1)
    dataset = OCRDataset(source['images'], source['labels'], get_transforms(False, 64))
    expected, _ = prepare_image(Image.open(dataset.image_paths[0]).convert('RGB'), 64)
    torch.testing.assert_close(dataset[0]['image'], expected, rtol=0, atol=0)


def test_evaluation_reads_directories(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from detection_larger.components import eval_metrics

    source = make_source(tmp_path, count=2)

    class PerfectDetector:
        def predict(self, image, threshold, nms_iou):
            return [{'box': [8, 8, 40, 24], 'score': 0.9}]

    monkeypatch.setattr(eval_metrics, 'DocumentDetector', lambda *args: PerfectDetector())
    monkeypatch.setattr(sys, 'argv', ['eval_metrics.py', '--model', 'unused.pth',
                                    '--image-dir', source['images'], '--label-dir', source['labels']])
    eval_metrics.main()
    report = json.loads(capsys.readouterr().out)
    assert report['overall']['tp'] == 2
    assert report['overall']['fp'] == report['overall']['fn'] == 0
    assert report['formats']['.png']['f1'] == 1


def test_training_options_only_accept_run_paths(tmp_path):
    from detection_larger.components.training_options import parse_args

    args = parse_args(['--weights', 'initial.pth', '--output', str(tmp_path / 'new')])
    assert args.weights == Path('initial.pth')
    assert args.output == tmp_path / 'new'
    for option in ('--epochs', '--input-size', '--train-images', '--label-format', '--device'):
        with pytest.raises(SystemExit):
            parse_args([option, '1'])
    (tmp_path / 'keep.pth').write_text('existing')
    with pytest.raises(FileExistsError, match='must be empty'):
        parse_args(['--output', str(tmp_path)])
    assert (tmp_path / 'keep.pth').read_text() == 'existing'
