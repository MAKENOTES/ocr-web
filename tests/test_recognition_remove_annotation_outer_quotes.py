from pathlib import Path

import pytest

from recognition.utils.remove_annotation_outer_quotes import (
    remove_outer_quotes_content,
)


def test_removes_only_one_outer_quote_pair() -> None:
    path = Path("annotations.txt")
    content = (
        'images/a.png\t"响应文件"\n'
        'images/b.png\t"他说：\\"可以\\""\n'
        'images/c.png\t"以一组"正常"细胞作为参考"\n'
    )

    converted, count = remove_outer_quotes_content(content, path)

    assert converted == (
        "images/a.png\t响应文件\n"
        'images/b.png\t他说：\\"可以\\"\n'
        'images/c.png\t以一组"正常"细胞作为参考\n'
    )
    assert count == 3


def test_is_idempotent_for_unquoted_text() -> None:
    path = Path("annotations.txt")
    content = "images/a.png\t响应文件\n"

    converted, count = remove_outer_quotes_content(content, path)

    assert converted == content
    assert count == 0


def test_rejects_empty_text_after_removing_quotes() -> None:
    path = Path("annotations.txt")

    with pytest.raises(ValueError, match=r"annotations\.txt:1"):
        remove_outer_quotes_content('images/a.png\t""\n', path)
