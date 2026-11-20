"""通过真实 TextRecognitionDataset 入口审计新合成批次；不修改去章实现。"""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime

ROOT = Path('/home/weiqiang/AI/ocr-web')
sys.path.insert(0, str(ROOT))
source_path = ROOT / 'recognition/components/dataset.py'
source_bytes = source_path.read_bytes()
from recognition.components import dataset as module
from recognition.tools.synthesis.generate_red_stamp import red_mask as synthesis_red_mask
import numpy as np
import torch
from PIL import Image, ImageDraw

assert source_path.read_bytes() == source_bytes, 'Dataset source changed during import'
torch.set_num_threads(1)
annotation = ROOT / 'recognition/data/synthesis_red_20260909_v2_annotation.txt'
generation_report = ROOT / 'recognition/utils/synthesis/synthesis_report/synthesis_red_20260909_v2'
report_dir = generation_report / 'red_seal_audit'
report_dir.mkdir(exist_ok=False)
processed_dir = report_dir / 'processed'
processed_dir.mkdir()
(report_dir / 'dataset_source.py').write_bytes(source_bytes)
(report_dir / 'audit_red_seal.py').write_bytes(Path(__file__).read_bytes())
source_lines = source_bytes.decode().splitlines()
metadata = [json.loads(line) for line in (generation_report / 'samples.jsonl').read_text().splitlines()]
by_path = {(annotation.parent / item['file']).resolve(): item for item in metadata}
dataset = module.TextRecognitionDataset(annotation, img_height=32, img_width=None,
                                         num_channels=3, is_train=False)
assert len(dataset) == len(metadata) == 1000
original = module.remove_red_seal
rows = []
active_path = None

def audited_remove(image):
    captured = {}
    def profile(frame, event, arg):
        if frame.f_code is original.__code__ and event == 'return':
            captured['red_mask'] = frame.f_locals['red_mask']
            captured['return_line'] = frame.f_lineno
    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = original(image)
    finally:
        sys.setprofile(previous_profile)
    before = np.asarray(image)
    after = np.asarray(result)
    assert before.shape == after.shape
    changed = np.any(before != after, axis=2)
    candidates = captured['red_mask']
    candidate_count = int(candidates.sum())
    changed_count = int(changed.sum())
    assert not (changed & ~candidates).any(), f'Changed non-candidate pixels: {active_path}'
    condition = source_lines[captured['return_line'] - 2].strip()
    if changed_count:
        reason = 'triggered'
    elif candidate_count == 0:
        reason = 'no_candidate_red'
    elif candidates.mean() > 0.35:
        reason = 'red_fraction_above_35_percent'
    elif 'paper_mask.mean()' in condition:
        reason = 'paper_background_gate'
    elif 'dark_y.size' in condition:
        reason = 'insufficient_dark_body'
    elif 'text_height <' in condition:
        reason = 'text_height_below_6'
    elif 'erase_mask.any()' in condition:
        reason = 'local_overlap_alignment_or_shape_protection'
    else:
        raise AssertionError(f'Unclassified exit: {captured["return_line"]}: {condition}')
    item = by_path[active_path]
    record = {
        'file': item['file'], 'scene': item['stamp']['scene'],
        'text': item['text'], 'width': image.width, 'height': image.height,
        'triggered': bool(changed_count), 'changed_pixels': changed_count,
        'changed_fraction': changed_count / changed.size,
        'candidate_red_pixels': candidate_count,
        'candidate_removed_fraction': changed_count / candidate_count if candidate_count else 0,
        'visible_red_before': int(synthesis_red_mask(before).sum()),
        'visible_red_after': int(synthesis_red_mask(after).sum()),
        'reason': reason, 'return_line': captured['return_line'],
        'processed': f'processed/{active_path.name}',
    }
    rows.append(record)
    result.save(processed_dir / active_path.name)
    return result

module.remove_red_seal = audited_remove
try:
    for index, path in enumerate(dataset.image_paths):
        active_path = Path(path).resolve()
        tensor, label = dataset[index]
        assert tensor.shape[0] == 3 and tensor.shape[1] == 32 and torch.isfinite(tensor).all()
        assert label == by_path[active_path]['text'].strip()
        if (index + 1) % 200 == 0:
            print(f'Audited {index + 1}/1000', flush=True)
finally:
    module.remove_red_seal = original
assert len(rows) == 1000

def aggregate(items):
    count = len(items)
    triggered = sum(row['triggered'] for row in items)
    before = sum(row['visible_red_before'] for row in items)
    after = sum(row['visible_red_after'] for row in items)
    candidates = sum(row['candidate_red_pixels'] for row in items)
    changed = sum(row['changed_pixels'] for row in items)
    return {
        'images': count, 'triggered': triggered, 'not_triggered': count - triggered,
        'trigger_rate': triggered / count,
        'changed_at_least_16_pixels': sum(row['changed_pixels'] >= 16 for row in items),
        'removed_at_least_half_candidates': sum(row['candidate_removed_fraction'] >= 0.5 for row in items),
        'changed_pixels': changed, 'candidate_red_pixels': candidates,
        'candidate_pixel_removal_rate': changed / candidates if candidates else 0,
        'visible_red_before': before, 'visible_red_after': after,
        'visible_red_pixel_reduction_rate': 1 - after / before if before else 0,
        'no_visible_red_remaining': sum(row['visible_red_after'] == 0 for row in items),
    }

summary = {
    'created_at': datetime.now().astimezone().isoformat(),
    'definition': 'Triggered iff remove_red_seal changes at least one RGB pixel before resize/augmentation.',
    'entry_point': 'TextRecognitionDataset.__getitem__ -> _load_and_process_image -> remove_red_seal',
    'mode': 'is_train=False; removal is shared with training and runs before random augmentation',
    'annotation': str(annotation), 'generation_report': str(generation_report),
    'dataset_source_sha256': hashlib.sha256(source_bytes).hexdigest(),
    'dataset_source_changed_during_audit': source_path.read_bytes() != source_bytes,
    'generation_script_sha256': hashlib.sha256((ROOT/'recognition/utils/synthesis/generate_red_stamp.py').read_bytes()).hexdigest(),
    'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
    'overall': aggregate(rows),
    'by_scene': {scene: aggregate([r for r in rows if r['scene'] == scene]) for scene in sorted({r['scene'] for r in rows})},
    'exit_reasons': dict(Counter(row['reason'] for row in rows)),
}
(report_dir/'samples.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
(report_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
lines = [
    '# 新生成 1000 张红章图：dataset.py 去章触发审计', '',
    f"- 标注：`{annotation}`", f"- 测试时间：{summary['created_at']}",
    f"- dataset.py SHA-256：`{summary['dataset_source_sha256']}`",
    f"- 审计期间源文件改变：{summary['dataset_source_changed_during_audit']}",
    '- 入口为真实 TextRecognitionDataset（验证模式）；在缩放前记录去章结果。训练也在随机增强前执行同一函数。',
    '- 触发定义：至少一个 RGB 像素被修改。函数无随机开关，比例是本批样本命中率，不是运行时随机概率。',
    '- 触发不等于整章去净，也不代表 OCR 准确率；未调用识别模型。', '',
    f"**触发 {summary['overall']['triggered']}/1000（{summary['overall']['trigger_rate']:.1%}）。**", '',
    '| 场景 | 张数 | 触发 | 未触发 | 触发率 | 候选红像素移除率（像素加权） |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for scene, value in summary['by_scene'].items():
    lines.append(f"| {scene} | {value['images']} | {value['triggered']} | {value['not_triggered']} | {value['trigger_rate']:.1%} | {value['candidate_pixel_removal_rate']:.1%} |")
lines += ['', '## 像素统计', '',
    f"- 改动至少 16 像素：{summary['overall']['changed_at_least_16_pixels']}/1000。",
    f"- 移除至少一半候选红像素：{summary['overall']['removed_at_least_half_candidates']}/1000。",
    f"- 全批候选红像素移除率：{summary['overall']['candidate_pixel_removal_rate']:.2%}。",
    f"- 较宽松可见红阈值（生成器 red_mask）下像素减少：{summary['overall']['visible_red_pixel_reduction_rate']:.2%}。",
    '- 已逐图断言未修改任何 dataset.py 候选红色之外的像素；这不是正文恢复准确率。', '',
    '## 返回分支统计', '', '```json', json.dumps(summary['exit_reasons'], ensure_ascii=False, indent=2), '```', '',
    '逐图记录见 `samples.jsonl`，去章后原尺寸 PNG 在 `processed/`，代码快照见 `dataset_source.py`。',
]
(report_dir/'report.md').write_text('\n'.join(lines)+'\n')
# 每种场景各选一张触发与一张未触发图，按中位移除率展示，原图与去章后上下对照。
chosen = []
for scene in summary['by_scene']:
    for triggered in (True, False):
        matches = sorted([r for r in rows if r['scene'] == scene and r['triggered'] == triggered],
                         key=lambda r: r['candidate_removed_fraction'])
        if matches:
            chosen.append(matches[len(matches)//2])
canvas = Image.new('RGB', (1600, len(chosen)*150), '#dedede')
draw = ImageDraw.Draw(canvas)
for index, row in enumerate(chosen):
    y = index*150
    draw.text((5, y+2), f"{row['scene']} | triggered={row['triggered']} | removed={row['candidate_removed_fraction']:.1%} | {Path(row['file']).name}", fill='black')
    for offset, image_path in ((24, annotation.parent/row['file']), (86, report_dir/row['processed'])):
        with Image.open(image_path) as im:
            im.thumbnail((1580, 56)); canvas.paste(im, (5,y+offset))
canvas.save(report_dir/'before_after.png')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
print('Report:', report_dir/'report.md', flush=True)
