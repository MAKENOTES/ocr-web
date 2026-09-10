"""错误定向合成的标签、可见边界、退化与输出契约。"""
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from recognition.utils.annotation import split_annotation_fields
from recognition.utils.synthesis import error_common as common
from recognition.utils.synthesis import generate_edge as edge
from recognition.utils.synthesis import generate_degraded as degraded
from recognition.utils.synthesis import generate_confusable as confusable
from recognition.utils.synthesis.style import Style

STYLE = Style((255, 255, 255), (0, 0, 0), 32, .8, .1, 3, 'test')


def body():
    image = Image.new('RGB', (80, 32), 'white')
    draw = ImageDraw.Draw(image)
    for x in (5, 28, 51):
        draw.rectangle((x, 4, x + 18, 28), outline='black', width=2)
    return image


@pytest.mark.parametrize('pattern', sorted(set(edge.PATTERNS) - {'fragment'}))
def test_edge_keeps_at_least_95_percent_of_ink_and_touches_requested_edge(pattern):
    source = body()
    output, detail = edge.apply_edge(source, '甲乙甲', STYLE, pattern, random.Random(7))
    before = (np.asarray(source).mean(axis=2) < 100).sum()
    after = (np.asarray(output).mean(axis=2) < 100).sum()
    assert after >= before * .95
    pixels = np.asarray(output).mean(axis=2)
    if pattern.endswith('right'):
        assert pixels[:, -1].min() == 0
    elif pattern.endswith('left'):
        assert pixels[:, 0].min() == 0
    assert output.tobytes() == source.crop(detail['crop_box']).tobytes()


def test_punctuation_is_not_cut_and_fragment_preserves_complete_body():
    source = body()
    output, detail = edge.apply_edge(source, '甲乙;', STYLE, 'clip_right', random.Random(1))
    assert detail['cut_ink_pixels'] == 0
    assert output.width == 70
    output, detail = edge.apply_edge(source, '甲乙甲', STYLE, 'fragment', random.Random(1), source)
    assert output.crop(detail['body_box']).tobytes() == source.tobytes()
    assert detail['fragment_width'] == 1
    assert np.asarray(output)[:, 0].min() == 0


@pytest.mark.parametrize('pattern', degraded.PATTERNS)
def test_degradation_is_visible_reproducible_and_preserves_size(pattern):
    source = body()
    output, detail = degraded.degrade(source, STYLE, pattern, random.Random(3))
    repeat, second = degraded.degrade(source, STYLE, pattern, random.Random(3))
    assert output.size == source.size and output.mode == 'RGB'
    assert output.tobytes() != source.tobytes()
    assert output.tobytes() == repeat.tobytes() and detail == second
    assert (np.asarray(output).mean(axis=2) < 180).sum() > 100


def test_confusion_alignment_keeps_both_han_characters_without_symbols():
    targets, groups = common.confusion_targets([
        {'text': '年已续约', 'prediction': '年己续约'},
        {'text': '家禽', 'prediction': '家&'},
        {'text': '互', 'prediction': '-'},
    ])
    assert targets == list('已己禽互')
    assert groups == [['已', '己'], ['禽'], ['互']]


@pytest.fixture
def setup_inputs(tmp_path):
    font = Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    if not font.exists():
        pytest.skip('Chinese test font unavailable')
    source = Image.new('RGB', (100, 32), 'white')
    ImageDraw.Draw(source).text((4, 0), '甲已乙', font=ImageFont.truetype(str(font), 24, index=2), fill='black')
    source.save(tmp_path / 'ref.png')
    analysis = tmp_path / 'analysis.md'
    analysis.write_text('\n'.join(f'## {title}（1）\n| [ref](ref.png) | 33 | 甲已乙 | 甲己乙 | 可辨 |'
                                  for title in common.TITLES.values()), encoding='utf-8')
    (tmp_path / 'keys.txt').write_text('甲\n乙\n已\n己\n', encoding='utf-8')
    (tmp_path / 'corpus.txt').write_text('a.png\t甲乙甲\nb.png\t乙甲乙\n', encoding='utf-8')
    return ['--analysis', str(analysis), '--keys', str(tmp_path / 'keys.txt'),
            '--corpus', str(tmp_path / 'corpus.txt'), '--fonts', str(font),
            '--images', '10', '--output-dir', str(tmp_path / 'images'),
            '--annotation', str(tmp_path / 'labels.txt'), '--report-dir', str(tmp_path / 'report')]


@pytest.mark.parametrize('module', [edge, degraded, confusable])
def test_outputs_have_correct_labels_counts_references_and_no_overwrite(module, setup_inputs):
    args = module.parse_args(setup_inputs)
    before = (args.analysis.parent / 'ref.png').read_bytes()
    report = module.generate(args)
    records = [json.loads(s) for s in (args.report_dir / 'samples.jsonl').read_text().splitlines()]
    labels = [split_annotation_fields(row, args.annotation, i)
              for i, row in enumerate(args.annotation.read_text().splitlines(), 1)]
    assert report['annotation'] == str(args.annotation)
    assert 'training_annotation' not in report
    assert not args.annotation.with_name('labels_training.txt').exists()
    assert not args.annotation.with_name('labels.txt.partial').exists()
    assert len(labels) == len(records) == len(list(args.output_dir.glob('*.png'))) == 10
    for (relative, text), record in zip(labels, records):
        assert relative == record['file'] and text == record['text']
        assert text != '甲已乙'
        with Image.open(args.annotation.parent / relative) as image:
            image.load()
            assert image.mode == 'RGB'
        if module is confusable:
            assert record['effect']['target'] in text
    if module is confusable:
        assert report['targets'] == {'已': 5, '己': 5}
    assert (args.analysis.parent / 'ref.png').read_bytes() == before
    with pytest.raises(FileExistsError):
        module.generate(args)


def test_bad_input_and_dry_run_do_not_create_outputs(setup_inputs):
    args = edge.parse_args(setup_inputs + ['--dry-run'])
    assert edge.generate(args)['dry_run']
    assert not args.output_dir.exists() and not args.report_dir.exists()
    args.images = 0
    with pytest.raises(ValueError, match='positive'):
        edge.generate(args)
    args.images = 10
    (args.analysis.parent / 'ref.png').unlink()
    with pytest.raises(FileNotFoundError, match='ref.png'):
        edge.generate(args)
    assert not args.output_dir.exists()


def test_failure_does_not_publish_training_labels(setup_inputs):
    args = edge.parse_args(setup_inputs)
    def fail(*unused):
        raise RuntimeError('render failed')
    with pytest.raises(RuntimeError, match='render failed'):
        common.run(args, fail)
    assert not args.annotation.exists()
    assert not args.annotation.with_name('labels_training.txt').exists()


def test_actual_analysis_reference_coverage():
    for kind, count in [('edge', 37), ('degraded', 18), ('confusable', 43)]:
        assert len(common.read_references(common.ANALYSIS, kind)) == count
