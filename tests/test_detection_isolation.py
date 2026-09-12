"""防止两个检测子系统通过 Python 导入重新产生相互依赖。"""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize('subsystem,forbidden', [
    ('detection', 'detection_larger'), ('detection_larger', 'detection'),
])
def test_detection_subsystems_do_not_import_each_other(subsystem, forbidden):
    root = Path(__file__).resolve().parents[1]
    paths = list((root / subsystem).rglob('*.py'))
    paths += [path for path in root.glob(f'{subsystem}_*.py')
              if subsystem != 'detection' or not path.name.startswith('detection_larger_')]
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            else:
                continue
            for module in modules:
                top = module.split('.')[0]
                blocked = top == forbidden or top.startswith(forbidden + '_')
                if forbidden == 'detection' and top.startswith('detection_larger'):
                    blocked = False
                assert not blocked, f'{path}:{node.lineno} imports {module}'
