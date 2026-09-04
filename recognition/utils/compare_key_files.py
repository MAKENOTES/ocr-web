#!/usr/bin/env python3
"""提取 keys-new.txt 中相对 keys.txt 新增的字符。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RECOGNITION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KEYS = RECOGNITION_DIR / "keys.txt"
DEFAULT_NEW_KEYS = RECOGNITION_DIR / "keys-new.txt"
DEFAULT_OUTPUT = RECOGNITION_DIR / "keys-append.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取 keys-new.txt 中存在、但 keys.txt 中不存在的字符。"
    )
    parser.add_argument(
        "--keys",
        type=Path,
        default=DEFAULT_KEYS,
        help="当前字符字典（默认：recognition/keys.txt）。",
    )
    parser.add_argument(
        "--keys-new",
        type=Path,
        default=DEFAULT_NEW_KEYS,
        help="待比较的新字符字典（默认：recognition/keys-new.txt）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="新增字符输出文件（默认：recognition/keys-append.txt）。",
    )
    return parser.parse_args()


def load_key_entries(keys_path: str | Path) -> list[str]:
    """按字典顺序读取唯一字符；忽略空行并保留空格字符。"""
    path = Path(keys_path)
    if not path.is_file():
        raise FileNotFoundError(f"字典文件不存在: {path}")

    entries: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            char = line.rstrip("\r\n")
            if not char:
                continue
            if len(char) != 1:
                raise ValueError(
                    f"字典每个非空行必须只有一个字符: "
                    f"{path}:{line_number}: {char!r}"
                )
            if char not in seen:
                entries.append(char)
                seen.add(char)

    return entries


def write_append_keys(
    keys_file: str | Path = DEFAULT_KEYS,
    new_keys_file: str | Path = DEFAULT_NEW_KEYS,
    output_file: str | Path = DEFAULT_OUTPUT,
) -> int:
    """写入新字典相对当前字典的有序差集，并返回新增字符数。"""
    current_keys = set(load_key_entries(keys_file))
    append_keys = [
        char for char in load_key_entries(new_keys_file) if char not in current_keys
    ]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{char}\n" for char in append_keys),
        encoding="utf-8",
    )
    return len(append_keys)


def main() -> int:
    args = parse_args()
    try:
        append_count = write_append_keys(args.keys, args.keys_new, args.output)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"比较完成！发现 {append_count} 个新增字符。")
    print(f"新增字符已保存至: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
