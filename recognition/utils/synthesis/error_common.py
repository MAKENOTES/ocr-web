"""三类错误合成共享输入、风格采样和产物保存；类别算法由各入口显式传入。"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import replace
from datetime import datetime
from difflib import SequenceMatcher
import html
import json
import os
from pathlib import Path
import random
import re

from PIL import Image, ImageDraw

from recognition.utils.annotation import format_annotation_line
from recognition.utils.character_stats import load_keys
from recognition.utils.synthesis.generate import (
    ROOT, DATA, default_font_paths, ensure_empty_outputs, load_fonts, render_image, sha256,
)
from recognition.utils.synthesis.generate_lines import measure_style
from recognition.utils.synthesis.generate_red_stamp import load_corpus, text_units
from recognition.utils.synthesis.naming import REPORT_ROOT

ANALYSIS = ROOT / 'recognition/datadoc/analysisdata/errdata20260909.md'  # 默认历史错误分析，提供类别、参考图链接及形近目标。
TITLES = {'edge': '裁切或边缘残字', 'degraded': '低清或笔画粘连', 'confusable': '汉字形近误识'}  # 各合成类别在错误分析中的二级标题，按精确标题匹配。


def read_references(path: Path, kind: str) -> list[dict]:
    """只读取指定二级分类的 PNG 链接和文本，不把验证图作为训练底图。"""
    rows, active = [], False
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            active = line.startswith('## ' + TITLES[kind] + '（')
        if not active or not re.search(r'\]\([^)]*\.png\)', line):
            continue
        fields = [part.strip() for part in line.strip('|').split('|')]
        link = re.search(r'\]\(([^)]+\.png)\)', fields[0]).group(1)
        image_path = (path.parent / link).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f'Reference PNG does not exist: {image_path}')
        if len(fields) != 5:
            raise ValueError(f'Invalid analysis table row in {path}: {line}')
        rows.append({'path': image_path, 'text': html.unescape(fields[2]),
                     'prediction': html.unescape(fields[3])})
    if not rows:
        raise ValueError(f'No references for {TITLES[kind]} in {path}')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate reference PNG in {path}: {kind}')
    return rows


def confusion_targets(rows: list[dict]) -> tuple[list[str], list[list[str]]]:
    """对齐替换段，收录真实汉字与误识汉字；非汉字符号只保留真实汉字。"""
    pairs, targets = [], []
    for row in rows:
        a, b = row['text'], row['prediction']
        for tag, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            chars = list(dict.fromkeys(c for c in a[i:j] + b[k:l] if '\u4e00' <= c <= '\u9fff'))
            if chars:
                pairs.append(chars)
                targets.extend(chars)
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise ValueError('No Chinese confusion targets found in analysis')
    return targets, pairs


def parse_args(kind: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f'参考错误分析合成{TITLES[kind]}，不覆盖已有产物',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', type=int, default=1000, help='生成图片数量，张')
    parser.add_argument('--seed', type=int, default=20260910, help='随机种子')
    parser.add_argument('--analysis', type=Path, default=ANALYSIS, help='包含三类分类及 PNG 链接的错误分析')
    parser.add_argument('--data-dir', type=Path, default=DATA, help='输出数据根目录')
    parser.add_argument('--corpus', type=Path, default=DATA / 'train_annotation.txt', help='正文语料，路径加 Tab 加文字')
    parser.add_argument('--keys', type=Path, default=ROOT / 'recognition/keys.txt', help='模型字典')
    parser.add_argument('--fonts', type=Path, nargs='+', help='字体文件，默认本机中文字体')
    parser.add_argument('--output-dir', type=Path, help='图片目录，默认数据根目录下按类别和日期命名')
    parser.add_argument('--annotation', type=Path, help='训练用 Tab 标注路径，只生成此文件')
    parser.add_argument('--report-dir', type=Path, help='报告目录，默认 synthesis_report 下按类别和日期命名')
    parser.add_argument('--dry-run', action='store_true', help='校验输入及字体、统计参考风格，不生成产物')
    args = parser.parse_args(argv)
    args.kind = kind
    args.date = datetime.now().strftime('%Y%m%d')
    prefix = f'synthesis_{kind}_{args.date}'
    args.output_dir = (args.output_dir or args.data_dir / f'{prefix}_images').resolve()
    args.annotation = (args.annotation or args.data_dir / f'{prefix}_annotation.txt').resolve()
    args.report_dir = (args.report_dir or REPORT_ROOT / prefix).resolve()
    return args


class Inputs:
    """一次加载各入口共用的字体、训练语料和逐图风格。"""
    def __init__(self, args):
        self.rows = read_references(args.analysis, args.kind)
        self.styles = [replace(measure_style(r['path']),
                               length=max(2, min(64, round(text_units(r['text']))))) for r in self.rows]
        self.keys = set(load_keys(args.keys))
        self.targets, self.pairs = confusion_targets(self.rows) if args.kind == 'confusable' else ([], [])
        missing = set(self.targets) - self.keys
        if missing:
            raise ValueError(f'Confusion targets absent from {args.keys}: {sorted(missing)}')
        visible = sorted(c for c in self.keys if c.isprintable() and not c.isspace())
        self.fonts = load_fonts(args.fonts or default_font_paths(), visible)
        self.corpus = load_corpus(args.corpus, self.keys, self.fonts)
        self.lengths = [row[0] for row in self.corpus]
        self.forbidden = {r['text'].replace(' ', '') for kind in TITLES
                          for r in read_references(args.analysis, kind)}

    def body(self, style, rng, target=None, bold=False):
        """按参考长度抽取训练文本；目标字替换一个汉字位置，标签跟随实际渲染文本。"""
        lo = bisect_left(self.lengths, max(1, style.length * .75))
        hi = bisect_right(self.lengths, max(3, style.length * 1.25))
        if lo == hi:
            lo, hi = 0, len(self.corpus)
        for _ in range(200):
            _, text, line, options = self.corpus[rng.randrange(lo, hi)]
            if target is not None:
                positions = [i for i, c in enumerate(text) if '\u4e00' <= c <= '\u9fff']
                if not positions:
                    continue
                pos = rng.choice(positions)
                text = text[:pos] + target + text[pos + 1:]
                options = [i for i in options if target in self.fonts[i].chars]
            if not options or text.replace(' ', '') in self.forbidden:
                continue
            weights = []
            for i in options:
                face = self.fonts[i]
                serif = 'Serif' in face.name or 'UMing' in face.name
                weight = 3 if serif else 1
                weight *= (3 if bold else .3) if 'Bold' in face.path.stem else 1
                weights.append(weight)
            face = self.fonts[rng.choices(options, weights=weights, k=1)[0]]
            image = render_image(text, face, style, rng)
            return image, text, face, {'corpus_line': line, 'target': target}
        raise ValueError(f'No eligible training text for target={target!r}, reference={style.reference}')


def run(args, make_sample) -> dict:
    """入口传入其 make_sample；这里仅负责共用校验、遍历、Tab 标注和报告。"""
    if args.images <= 0:
        raise ValueError('--images must be positive')
    ensure_empty_outputs(args.output_dir, args.annotation, args.report_dir)
    partial = args.annotation.with_name(args.annotation.name + '.partial')
    if partial.exists():
        raise FileExistsError(f'Output already exists: {partial}')
    inputs = Inputs(args)
    summary = {'images': args.images, 'kind': args.kind, 'seed': args.seed,
               'reference_count': len(inputs.rows), 'output_dir': str(args.output_dir),
               'annotation': str(args.annotation), 'annotation_format': 'relative/path.png<TAB>text',
               'analysis_sha256': sha256(args.analysis), 'keys_sha256': sha256(args.keys),
               'corpus': str(args.corpus.resolve()), 'corpus_sha256': sha256(args.corpus),
               'confusion_groups': inputs.pairs,
               'font_sha256': {str(f.path): sha256(f.path) for f in inputs.fonts},
               'reference_sha256': {str(r['path']): sha256(r['path']) for r in inputs.rows}}
    if args.dry_run:
        return dict(summary, dry_run=True)
    rng = random.Random(args.seed)
    references = [i % len(inputs.styles) for i in range(args.images)]
    rng.shuffle(references)
    for directory in (args.output_dir, args.annotation.parent, args.report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    preview = Image.new('RGB', (1500, 1200), '#dedede')
    draw = ImageDraw.Draw(preview)
    counts, targets, ref_counts = Counter(), Counter(), Counter()
    preview_count = 0
    with partial.open('x', encoding='utf-8') as labels, \
            (args.report_dir / 'samples.jsonl').open('x', encoding='utf-8') as records:
        for i, reference in enumerate(references):
            style = inputs.styles[reference]
            image, text, face, detail = make_sample(inputs, style, rng, i)
            if not text or not set(text) <= inputs.keys or image.mode != 'RGB':
                raise ValueError(f'Invalid generated sample {i + 1}: {text!r}')
            path = args.output_dir / f'synthesis_{args.kind}_{args.date}_{i + 1:05d}.png'
            with path.open('xb') as handle:
                image.save(handle, format='PNG')
            relative = Path(os.path.relpath(path, args.annotation.parent)).as_posix()
            labels.write(format_annotation_line(relative, text))
            records.write(json.dumps({'file': relative, 'text': text, 'font': str(face.path),
                                      'font_index': face.index, 'style': style.to_dict(),
                                      'effect': detail}, ensure_ascii=False) + '\n')
            counts[detail['pattern']] += 1
            ref_counts[style.reference] += 1
            if detail.get('target'):
                targets[detail['target']] += 1
            # 首 12 张保留连续模式，其余均匀抽样；长行只缩略显示，原 PNG 保持原尺寸。
            if i < 12 or i % max(1, args.images // 36) == 0:
                slot = preview_count
                if slot < 48:
                    thumb = image.copy()
                    thumb.thumbnail((735, 30))
                    x, y = (slot % 2) * 750 + 5, (slot // 2) * 50
                    draw.text((x, y), f'{i + 1:05d} {detail["pattern"]}', fill='black')
                    preview.paste(thumb, (x, y + 17))
                    preview_count += 1
            if (i + 1) % 200 == 0:
                print(f'{args.kind}: {i + 1}/{args.images}', flush=True)
    summary.update(patterns=dict(counts), targets=dict(targets), references=dict(ref_counts))
    preview.save(args.report_dir / 'preview.png')
    (args.report_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.link(partial, args.annotation)
    partial.unlink()
    return summary
