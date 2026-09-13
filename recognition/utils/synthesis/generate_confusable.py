#!/usr/bin/env python3
"""从形近误识表提取目标字，参考小字号姓名、机构名和正文风格合成 1000 张。"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录，由当前文件位置推导，不依赖运行目录。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.utils.synthesis import error_common as common


def make_sample(inputs, style, rng, index):
    # 按目标字均分图片配额，混淆双方都渲染其正确字形；保留正常清晰度。
    target = inputs.targets[index % len(inputs.targets)]
    body, text, face, detail = inputs.body(style, rng, target=target)
    return body, text, face, dict(detail, pattern='confusable')


def parse_args(argv=None):
    return common.parse_args('confusable', argv)


def generate(args):
    return common.run(args, make_sample)


def main():
    try:
        report = generate(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    print(f'{"Validated" if report.get("dry_run") else "Saved"} {report["images"]} images; {report["annotation"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
